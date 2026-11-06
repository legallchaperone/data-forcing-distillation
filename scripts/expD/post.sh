#!/bin/bash
# After both Exp D trainings finished: generate held-out teacher paths (different seed), evaluate every checkpoint of
# both runs on them, decode 2 videos per run from the last checkpoint. Re-runnable.
source /tmp/fg/env.sh; cd /tmp/fg/src
HELD=/tmp/fg/out/DATA/wan1.3b-kd-paths-heldout
echo "== POST START $(date -u)"
if [ $(ls $HELD/*.tar 2>/dev/null | wc -l) -lt 2 ]; then
  $PY scripts/data_gen/generate_wan_kd_paths.py --num-samples 16 --seed 7 --shard-size 8 --batch-size 2 --num-steps 30 --out-dir $HELD > /tmp/fg/gen_heldout.log 2>&1
  echo "== HELDOUT GEN EXIT=$? $(date -u)"
fi
CFG=fastgen/configs/experiments/WanT2V/config_kd_causal_swift.py
COMMON="log_config.wandb_mode=disabled"
$PY scripts/expD/eval_checkpoints.py --config $CFG --heldout-dir $HELD --videos 2 --out /tmp/fg/out/evalD_baseline.json -- $COMMON model.swift_alpha=0.0  log_config.name=kd_causal_baseline  > /tmp/fg/evalD_baseline.log 2>&1
echo "== EVAL baseline EXIT=$? $(date -u)"
$PY scripts/expD/eval_checkpoints.py --config $CFG --heldout-dir $HELD --videos 2 --out /tmp/fg/out/evalD_swift.json    -- $COMMON model.swift_alpha=0.25 log_config.name=kd_causal_swift_a025 > /tmp/fg/evalD_swift.log 2>&1
echo "== EVAL swift EXIT=$? $(date -u)"
echo "== POST DONE $(date -u)"
