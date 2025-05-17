# launch_all.sh
#!/usr/bin/env bash
set -euo pipefail

LOG=~/batch_submissions.log
REPLICATES=${1:-10}
MAXIDX=$(( REPLICATES - 1 ))

submit() {
  out=$(sbatch "$@")
  jid=$(echo "$out" | awk '{print $4}')
  printf '%s\t%s\t%s\n' \
    "$(date -Is)" \
    "$jid" \
    "$(printf '%q ' "$@")" \
  >> "$LOG"
}

# clean and recreate log directory
rm -rf logs/*
mkdir -p logs

# define your sweep lists here
variants=(raw pct z invn)
labels=(ret_exc_lead1m ret_pct ret_z ret_invn)
weights=(w_ew w_vw)
losses=(mse mae)

# iterate
for variant in "${variants[@]}"; do
  for label in "${labels[@]}"; do
    for weight in "${weights[@]}"; do
      for loss in "${losses[@]}"; do
        submit \
          -J ens_${variant}_${label}_${weight}_${loss} \
          --partition=normal \
          --gres=gpu:1 \
          --cpus-per-task=8 \
          --mem=16G \
          --time=1-00:00:00 \
          --array=0-${MAXIDX} \
          -o logs/ens_${variant}_${label}_${weight}_${loss}_%A_%a.out \
          -e logs/ens_${variant}_${label}_${weight}_${loss}_%A_%a.err \
          train_ensemble.sh \
            --variant "$variant" \
            --label   "$label" \
            --weight  "$weight" \
            --loss    "$loss"
      done
    done
  done
done