#!/usr/bin/env bash
# Unattended, resumable experiment chain for a fresh sandbox. Each stage is skipped when its output already exists.
# Usage (after setup_sandbox.sh):  nohup bash run_pipeline.sh > /marimo/work/pipeline.log 2>&1 &
set -uo pipefail
WORK=${WORK:-/marimo/work}
PY=${PY:-/tmp/uv-venv/bin/python}
E=$(cd "$(dirname "$0")" && pwd)
D=$WORK/data
N_TEACHER=${N_TEACHER:-32}
STEPS=${STEPS:-400}
cd "$WORK/sf" || exit 1
stage() { echo "=== $(date +%T) $*"; }

if [ -n "${HF_REPO:-}" ]; then
  $PY -c "from huggingface_hub import HfApi; print('HF user:', HfApi().whoami()['name'])" || { echo "HF_REPO set but not logged in (run: hf auth login)"; exit 1; }
  echo "mirroring to hf://$HF_REPO"
  stage "restore from hf://$HF_REPO"
  $PY -c "from huggingface_hub import HfApi, snapshot_download; HfApi().create_repo('$HF_REPO', private=True, exist_ok=True); snapshot_download(repo_id='$HF_REPO', local_dir='$D')" 2>&1 | tail -1
  mkdir -p $D; ls $D
fi


stage teacher
[ "$(ls $D/teacher/*.pt 2>/dev/null | wc -l)" -ge "$N_TEACHER" ] || \
  $PY $E/gen_teacher.py --out $D/teacher --n $N_TEACHER > $WORK/gen_full.log 2>&1

[ -n "${HF_REPO:-}" ] && $PY $E/hf_mirror.py $D/teacher

stage measure
[ -f $D/kappa/rows.jsonl ] && [ "$(wc -l < $D/kappa/rows.jsonl)" -ge $((N_TEACHER * 12)) ] || \
  $PY $E/measure_kappa.py --data $D/teacher --out $D/kappa > $WORK/kappa_full.log 2>&1

ft() {  # name, extra args...
  local name=$1; shift
  stage "finetune $name"
  [ -f $D/$name/student.pt ] || \
    $PY $E/finetune.py --data $D/teacher --cgen $D/kappa/cgen --out $D/$name --steps $STEPS --eval_every $STEPS --save_ckpt "$@" > $WORK/$name.log 2>&1
}
roll() {  # name, ckpt
  stage "rollout $1"
  [ -f $D/roll_$1/rows.jsonl ] && [ "$(wc -l < $D/roll_$1/rows.jsonl)" -ge 240 ] || \
    $PY $E/rollout_eval.py --ckpt "$2" --data $D/teacher --out $D/roll_$1 --decode 2 > $WORK/roll_$1.log 2>&1
}

roll init checkpoints/ode_init.pt
ft ft_a0 --alpha 0
roll a0 $D/ft_a0/student.pt
ft ft_t3_sched --mode T3 --alpha_schedule 625:1.9,833:1.3,937:0.8,1000:0.4
roll t3_sched $D/ft_t3_sched/student.pt
ft ft_t3_a03 --mode T3 --alpha 0.3
roll t3_a03 $D/ft_t3_a03/student.pt
ft ft_t2_a1 --mode T2 --alpha 1.0
roll t2_a1 $D/ft_t2_a1/student.pt
stage PIPELINE_DONE
