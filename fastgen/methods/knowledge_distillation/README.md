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

**Configs:** [`WanT2V/config_kd_causal_swift.py`](../../configs/experiments/WanT2V/config_kd_causal_swift.py)
