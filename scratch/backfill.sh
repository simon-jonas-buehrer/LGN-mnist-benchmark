#!/bin/bash
# Keep the GPU budget saturated: whenever the account has headroom (<18 GPU jobs) and this
# session's experiment jobs (hg*/reg*/gen* tags) are under 8, submit the next spec from
# scratch/queue.txt (lines of "tag::extra args"; consumed top-down). Emits one line per
# submit; exits when the queue is empty.
cd "$(dirname "$0")/.." || exit 1
Q=scratch/queue.txt
source scratch/base.env
while true; do
  [ -s "$Q" ] || { echo "backfill queue empty -- done"; exit 0; }
  in_use=$(squeue -u "$USER" -h -o "%b" 2>/dev/null | grep -c gpu)
  mine=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -cE '^(hg|reg|gen)' || true)
  if [ "$in_use" -lt 18 ] && [ "$mine" -lt 10 ]; then
    spec=$(head -1 "$Q")
    tag="${spec%%::*}"; extra="${spec#*::}"
    out=$(sbatch --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=04:00:00 \
      --job-name="$tag" --output="scratch/${tag}.out" \
      --wrap="bash scratch/cd.sh --ckpt scratch/${tag}.pt $BASE $extra" 2>&1 | tail -1)
    if echo "$out" | grep -q Submitted; then
      sed -i 1d "$Q"
      echo "backfilled $tag ($out; account $((in_use+1))/18, session $((mine+1))/8)"
    else
      echo "backfill $tag sbatch FAILED: $out"; sleep 900
    fi
  fi
  sleep 300
done
