# rerun_failed.sh
#!/usr/bin/env bash
set -euo pipefail

LOG=~/batch_submissions.log
TMP_FAILED=~/failed_tasks.txt

# 1) extract all jobIDs from the log
awk '{print $2}' "$LOG" | sort -u | paste -sd, - > all_jobs.csv

# 2) query SLURM for their final states
sacct -n -P \
  --jobs="$(cat all_jobs.csv)" \
  --format=JobID,State \
  --allocations | \
awk -F'|' '$2!="COMPLETED"' > "$TMP_FAILED"

# 3) group failed array-tasks by parent job and re-submit only those indices
awk -F_ '{ failed[$1] = failed[$1] ? failed[$1]","$2 : $2 }
          END { for (j in failed) print j, failed[j] }' "$TMP_FAILED" | \
while read parent indices; do
  # grab the last sbatch line for this parent
  orig=$(grep -P "\t${parent}\t" "$LOG" | tail -1)
  # strip existing --array=
  cmd=$(echo "$orig" | sed -E 's/--array=[^ ]+//')
  echo "Re-submitting $parent array indices $indices"
  sbatch --array="$indices" $cmd
done