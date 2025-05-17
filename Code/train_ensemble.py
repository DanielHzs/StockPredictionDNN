import os
import glob
import json
import argparse
import pathlib
import multiprocessing as mp
import tensorflow as tf
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# 1. Configuration (defaults; CLI overrides applied in main)
# ────────────────────────────────────────────────────────────────────────────────
CONFIG = {
    'data_dir':      os.environ.get('DATA_DIR', '/gpfs/home/zh283/StockPredictionDNN/Data'),
    'model_dir':     '/gpfs/home/zh283/StockPredictionDNN/chkpts_script',
    'variant':       None,
    'label':         None,
    'weight':        None,
    'loss':          None,
    'feature_dim':   153,
    'train_years':   list(range(1971, 1997)),
    'val_years':     list(range(1998, 2002)),
    'predict_years': list(range(2003, 2024)),
    'num_layers':        5,
    'neurons_per_layer': 128,
    'dropout':           0,
    'use_batch_norm':    True,
    'huber_delta':       1.0,
    'learning_rate':     1e-3,
    'lr_scheduler':      'reduce_on_plateau',
    'decay_steps':       10000,
    'decay_rate':        0.96,
    'reduce_patience':   5,
    'reduce_factor':     0.5,
    'min_lr':            1e-6,
    'train_batch_size': 2048,
    'eval_batch_size':  32976,
    'epochs':           200,
    'ensemble_size':    10,
}

# ────────────────────────────────────────────────────────────────────────────────
# 2. Dataset utilities
# ────────────────────────────────────────────────────────────────────────────────
@tf.autograph.experimental.do_not_convert
def _parse_example_fn(feature_dim: int, mode: str = 'train'):
    desc = {
        'feat': tf.io.FixedLenFeature([feature_dim], tf.float32),
        CONFIG['label']: tf.io.FixedLenFeature([], tf.float32),
        CONFIG['weight']: tf.io.FixedLenFeature([], tf.float32),
    }
    if mode == 'predict':
        desc.update({
            'permno':           tf.io.FixedLenFeature([], tf.int64),
            'eom':              tf.io.FixedLenFeature([], tf.int64),
            'me':               tf.io.FixedLenFeature([], tf.float32),
            'size_grp':         tf.io.FixedLenFeature([], tf.string),
            'crsp_exchcd':      tf.io.FixedLenFeature([], tf.int64),
            'ret':              tf.io.FixedLenFeature([], tf.float32),
            'ret_exc':          tf.io.FixedLenFeature([], tf.float32),
            'ret_exc_lead1m':   tf.io.FixedLenFeature([], tf.float32),
        })
    def _parse(record):
        ex = tf.io.parse_single_example(record, desc)
        feat = ex.pop('feat')
        y    = ex.pop(CONFIG['label'])
        w    = ex.pop(CONFIG['weight'])
        return (feat, y, w, ex) if mode == 'predict' else (feat, y, w)
    return _parse


def make_dataset(years, mode='train'):
    tfrecord_dir = CONFIG['tfrecord_dir']
    # since tfrecord_dir already includes variant, match year files directly
    pattern = os.path.join(tfrecord_dir, "*.tfrecord")
    files = [f for f in glob.glob(pattern)
             if int(os.path.basename(f).split('year')[1].split('.')[0]) in years]
    ds_files = tf.data.Dataset.from_tensor_slices(sorted(files))
    cycle = min(16, len(files)) if mode=='train' and len(files)>1 else 1
    if mode=='train' and len(files)>1:
        ds_files = ds_files.shuffle(len(files))
    ds = ds_files.interleave(tf.data.TFRecordDataset,
                             cycle_length=cycle,
                             num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_parse_example_fn(CONFIG['feature_dim'], mode),
                num_parallel_calls=tf.data.AUTOTUNE)
    if mode=='train':
        ds = ds.cache().shuffle(10_000)
        bsz = CONFIG['train_batch_size']
    elif mode=='val':
        ds = ds.cache()
        bsz = CONFIG['eval_batch_size']
    else:  # predict
        # no cache, just batch
        bsz = CONFIG['eval_batch_size']
    return ds.batch(bsz).prefetch(tf.data.AUTOTUNE)

# ────────────────────────────────────────────────────────────────────────────────
# 3. Model factory
# ────────────────────────────────────────────────────────────────────────────────
LOSS_MAP = {
    'mse':   tf.keras.losses.MeanSquaredError(),
    'mae':   tf.keras.losses.MeanAbsoluteError(),
    'huber': tf.keras.losses.Huber(delta=CONFIG['huber_delta']),
}
LR_SCHEDULES = {
    'none': lambda: CONFIG['learning_rate'],
    'exponential_decay': lambda: tf.keras.optimizers.schedules.ExponentialDecay(
        CONFIG['learning_rate'], CONFIG['decay_steps'], CONFIG['decay_rate'], staircase=True)
}

def build_model():
    inp = tf.keras.Input(shape=(CONFIG['feature_dim'],))
    x = inp
    for _ in range(CONFIG['num_layers']):
        x = tf.keras.layers.Dense(CONFIG['neurons_per_layer'])(x)
        if CONFIG['use_batch_norm']:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        if CONFIG['dropout'] > 0:
            x = tf.keras.layers.Dropout(CONFIG['dropout'])(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    lr = LR_SCHEDULES.get(CONFIG['lr_scheduler'], lambda: CONFIG['learning_rate'])()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=LOSS_MAP[CONFIG['loss']], metrics=['mae'])
    return model

# ────────────────────────────────────────────────────────────────────────────────
# 4. Worker
# ────────────────────────────────────────────────────────────────────────────────
def train_single(idx: int):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_ds = make_dataset(CONFIG['train_years'], 'train')
    val_ds   = make_dataset(CONFIG['val_years'],   'val')

    model = build_model()
    cbs = [tf.keras.callbacks.EarlyStopping('val_loss', patience=15, restore_best_weights=True)]
    if CONFIG['lr_scheduler'] == 'reduce_on_plateau':
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=CONFIG['reduce_patience'],
            factor=CONFIG['reduce_factor'], verbose=1, min_lr=CONFIG['min_lr']
        ))
    cbs.append(tf.keras.callbacks.TensorBoard(log_dir=str(CONFIG['log_dir'] / f"run_{idx}"), update_freq='epoch'))

    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG['epochs'], callbacks=cbs, verbose=2)
    model.save(CONFIG['base_dir'] / f"run_{idx}.keras", save_format='keras_v3')

# ────────────────────────────────────────────────────────────────────────────────
# 5. Entry point
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Train an ensemble of models for a given hyperparameter combo.")
    p.add_argument('--variant', required=True, choices=['raw','pct','z','invn'])
    p.add_argument('--label',   required=True, choices=['ret_exc_lead1m','ret_pct','ret_z','ret_invn'])
    p.add_argument('--weight',  required=True, choices=['w_ew','w_vw'])
    p.add_argument('--loss',    required=True, choices=['mse','mae','huber'])
    p.add_argument('--replica', required=True, type=int, help='Ensemble index (0-based)')
    args = p.parse_args()

    # override config from CLI
    CONFIG.update({
        'variant': args.variant,
        'label':   args.label,
        'weight':  args.weight,
        'loss':    args.loss,
    })

    # derived paths now that CLI is set
    CONFIG['tfrecord_dir'] = os.path.join(CONFIG['data_dir'], 'tfrecords', args.variant)
    base = pathlib.Path(CONFIG['model_dir']) / f"{args.variant}_{args.label}_{args.weight}_{args.loss}"
    CONFIG['base_dir'] = base
    CONFIG['log_dir'] = base / 'logs'
    base.mkdir(parents=True, exist_ok=True)
    CONFIG['log_dir'].mkdir(exist_ok=True)

    mp.set_start_method('spawn', force=True)
    train_single(args.replica)
