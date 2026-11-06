# Exp D: causal Wan 1.3B KD (ODE init), plain vs SWIFT

Scripts for running the comparison on a single-GPU sandbox (see `fastgen/methods/knowledge_distillation/README.md`
for the SWIFT option itself).

```bash
nohup bash bootstrap.sh > /tmp/fg/bootstrap.log 2>&1 &   # clone, venv, Wan2.1-1.3B weights (~3 min on a fast link)
nohup bash /tmp/fg/src/scripts/expD/run.sh > /tmp/fg/expD.log 2>&1 &   # generation (until deadline), then both trainings
```

`run.sh` is idempotent: re-running it resumes path generation at the last complete shard (the deadline is fixed at
the first launch) and resumes both trainings from their latest checkpoints. Logs: `/tmp/fg/gen_paths.log`,
`/tmp/fg/trainD_{baseline,swift}.log`. Checkpoints: `/tmp/fg/out/fastgen/wan_kd_causal_swift/<run>/checkpoints/`.
Measured on an RTX Pro 6000: about 70 s per path sample (30 UniPC steps, CFG 5, 480p x 81 frames).
