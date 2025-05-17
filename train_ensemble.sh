#!/usr/bin/env bash
#SBATCH --job-name=ensem             # base name; overridden by launcher
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-10                 # 10 replicas

set -eo pipefail                    # exit on errors, but allow unset vars

# log exactly how we were invoked
echo "$(date -Is) — Called as: $0 $@" >> ~/batch_submissions.log

module load cuda/12.2
eval "$(conda shell.bash hook)"
conda activate tf_env

# parse hyperparam flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --variant) VARIANT="$2"; shift 2;;
    --label)   LABEL="$2";   shift 2;;
    --weight)  WEIGHT="$2";  shift 2;;
    --loss)    LOSS="$2";    shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

: "${VARIANT:?must set --variant}"
: "${LABEL:?must set --label}"
: "${WEIGHT:?must set --weight}"
: "${LOSS:?must set --loss}"

python /gpfs/home/zh283/StockPredictionDNN/Code/train_ensemble.py \
  --variant "$VARIANT" \
  --label   "$LABEL" \
  --weight  "$WEIGHT" \
  --loss    "$LOSS" \
  --replica "$SLURM_ARRAY_TASK_ID"