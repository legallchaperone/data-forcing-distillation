#!/bin/bash
# Exp D pipeline (resumable): teacher path generation until a wall-clock deadline, then causal KD baseline + SWIFT.
# Re-running resumes generation at the last complete shard and resumes training from the latest checkpoint.
# Env knobs: GEN_HOURS (default 3) generation budget from first launch; TARGET (default 400) max samples;
#            MAX_ITER (default 2000), SAVE_EVERY (default 250), SNAPSHOT (default MAX_ITER/4).
source /tmp/fg/env.sh; cd /tmp/fg/src
DATA=/tmp/fg/out/DATA/wan1.3b-kd-paths; SHARD=8; TARGET=${TARGET:-400}
MAX_ITER=${MAX_ITER:-2000}; SAVE_EVERY=${SAVE_EVERY:-250}; SNAPSHOT=${SNAPSHOT:-$((MAX_ITER/4))}
mkdir -p $DATA
# the deadline is fixed at first launch so that re-runs do not extend the generation budget
[ -f $DATA/.deadline ] || date -u -d "+${GEN_HOURS:-3} hours" +%s > $DATA/.deadline
GEN_DEADLINE=$(cat $DATA/.deadline)
complete_shards() { n=0; for t in $(ls $DATA/*.tar 2>/dev/null | sort); do c=$(tar tf $t 2>/dev/null | wc -l); if [ "$c" -eq $((SHARD*4)) ]; then n=$((n+1)); else rm -f $t; echo "removed incomplete shard $t ($c entries)" >&2; fi; done; echo $n; }
echo "== EXPD START $(date -u)  deadline $(date -u -d @$GEN_DEADLINE)"
if [ ! -f $DATA/.gen_done ]; then
  while [ $(date +%s) -lt $GEN_DEADLINE ]; do
    N=$(complete_shards); START=$((N*SHARD))
    if [ $START -ge $TARGET ]; then break; fi
    echo "== GEN from sample $START ($N complete shards) $(date -u)"
    $PY scripts/data_gen/generate_wan_kd_paths.py --num-samples $TARGET --start-idx $START --shard-size $SHARD --batch-size 2 --num-steps 30 --out-dir $DATA >> /tmp/fg/gen_paths.log 2>&1 &
    GENPID=$!
    while kill -0 $GENPID 2>/dev/null && [ $(date +%s) -lt $GEN_DEADLINE ]; do sleep 30; done
    if kill -0 $GENPID 2>/dev/null; then echo "== GEN DEADLINE, stopping $(date -u)"; kill $GENPID; sleep 5; kill -9 $GENPID 2>/dev/null; break; fi
    wait $GENPID; echo "== GEN EXIT=$? $(date -u)"; sleep 2
  done
  N=$(complete_shards); echo "== GEN DONE: $N complete shards = $((N*SHARD)) samples $(date -u)"; touch $DATA/.gen_done
fi
COMMON="trainer.max_iter=$MAX_ITER trainer.save_ckpt_iter=$SAVE_EVERY trainer.validation_iter=100000 trainer.logging_iter=25 model.swift_snapshot_iter=$SNAPSHOT trainer.callbacks.ema_swift.snapshot_iter=$SNAPSHOT log_config.wandb_mode=disabled"
CFG=fastgen/configs/experiments/WanT2V/config_kd_causal_swift.py
echo "== TRAIN LAUNCH $(date -u)"
nohup $PY train.py --config $CFG - $COMMON model.swift_alpha=0.0  log_config.name=kd_causal_baseline  >> /tmp/fg/trainD_baseline.log 2>&1 &
sleep 20
nohup $PY train.py --config $CFG - $COMMON model.swift_alpha=0.25 log_config.name=kd_causal_swift_a025 >> /tmp/fg/trainD_swift.log 2>&1 &
echo "== TRAIN LAUNCHED $(date -u)"
