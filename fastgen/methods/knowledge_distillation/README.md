# Knowledge Distillation Methods

Learn from pre-computed teacher denoising trajectories.

## KD (Knowledge Distillation)

**File:** [`KD.py`](KD.py) | **Reference:** [Luhman & Luhman, 2021](https://arxiv.org/abs/2101.02388)

MSE loss between student prediction and teacher output: `L = ||f(x_t, t) - x_0^teacher||²`

**Data Requirements:**
- Single-step: `{"real": clean, "noise": noise, "condition": cond}`
- Multi-step: `{"real": clean, "path": [B, steps, C, H, W], "condition": cond}`

**Key Parameters:**
- `student_sample_steps`: Number of student steps
- `sample_t_cfg.t_list`: Timesteps (must align with path)

**Configs:** [`WanT2V/config_kd.py`](../../configs/experiments/WanT2V/config_kd.py), [`SDXL/config_kd.py`](../../configs/experiments/SDXL/config_kd.py), [`CogVideoX/config_kd.py`](../../configs/experiments/CogVideoX/config_kd.py)

---

## CausalKD

**File:** [`KD.py`](KD.py) | **Reference:** [Yin et al., 2024](https://arxiv.org/abs/2412.07772)

KD for causal video models with inhomogeneous timesteps and autoregressive generation.

**Data Requirements:**
- `{"real": [B,C,T,H,W], "path": [B,steps,C,T,H,W], "condition": cond}`

**Key Parameters:**
- `context_noise`: Noise for cached context
- See also key parameters of KD above

**Configs:** [`WanT2V/config_kd_path.py`](../../configs/experiments/WanT2V/config_kd_path.py)

---

## SWIFT target extrapolation (KD and CausalKD)

**Reference:** [SWIFT, Self-guided Weak-to-strong dIFfusion models via Target extrapolation](https://arxiv.org/abs/2610.25844)

Optional for both models above. The regression target `x_0^teacher` is replaced by

`x_0^teacher + swift_alpha * (x_0^teacher - G_weak(x_t, t))`

where `G_weak` is a frozen snapshot of the student taken at `swift_snapshot_iter` (the plain target is used before
that). The weak network is held like an EMA network (full copy per rank, not FSDP-sharded, checkpointed) and is
filled by an `EMACallback(type="snapshot", snapshot_iter=..., ema_name=swift_weak_name)`.

**Key Parameters (`KDModelConfig`):**
- `swift_alpha`: extrapolation strength, `0` disables (default)
- `swift_snapshot_iter`: total iteration of the snapshot (e.g. 1/4 of `max_iter`), must match the callback
- `swift_weak_name`: name of the weak network, must start with `ema` (default `ema_swift`)

**Logged:** `recon_loss` is the training loss against the extrapolated target; `recon_loss_plain` is the MSE against
the un-extrapolated teacher output and is the quantity to compare against a plain-KD run.

**Configs:** [`WanT2V/config_kd_causal_swift.py`](../../configs/experiments/WanT2V/config_kd_causal_swift.py),
[`EDM/config_kd_cifar10.py`](../../configs/experiments/EDM/config_kd_cifar10.py) (baseline) /
[`EDM/config_kd_cifar10_swift.py`](../../configs/experiments/EDM/config_kd_cifar10_swift.py)

**CIFAR-10 reference run** (single seed, 4000 iters, batch 256, EDM cond-vp teacher pairs from
`scripts/data_gen/generate_edm_kd_pairs.py`; raw curves in `assets/experiments/cifar10_kd_swift/results.json`):

| FID-10k at iteration | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
|---|---|---|---|---|---|---|---|---|
| plain KD | 28.2 | 21.6 | 18.6 | 16.5 | 15.3 | 14.8 | 13.9 | 13.4 |
| KD + SWIFT (alpha 0.25, snapshot at 1000) | 27.8 | 22.8 | 16.5 | 14.8 | 12.7 | 11.9 | 10.7 | 10.5 |

The two runs are identical before the snapshot. After it, the un-extrapolated regression loss is about 4% lower and
FID is lower at every checkpoint; SWIFT at 2500 iterations already matches plain KD at 4000.

**Causal Wan 1.3B reference run** (`scripts/expD/`, single seed, reduced resolution 320x576 x 33 frames, 240 teacher
paths, 1200 iters, batch 2, snapshot at 300; raw curves in `assets/experiments/wan_causal_kd_swift/results.json`):
held-out plain MSE on 16 fresh teacher paths is flat at about 0.072 for both runs across all checkpoints (the
one-step-from-noise term at t=0.999 dominates it), yet same-noise samples differ strongly: the plain-KD student
collapses to a blurry posterior-mean-like video while the SWIFT student keeps texture and colors close to the teacher
(`teacher_vs_students_same_noise.jpg`). MSE is the wrong metric for this comparison; judge with FVD or VBench.
