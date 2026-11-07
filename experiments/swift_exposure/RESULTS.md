# Swift-style extrapolation against exposure bias: results so far (2026-11-07)

Setting: Wan2.1-T2V-1.3B causal student, Self-Forcing `ode_init.pt` (4-step, timesteps 1000/937/833/625), the same
starting point as Rolling Forcing. Pseudo-ground-truth: 32 clips sampled with the bidirectional 1.3B teacher
(50-step UniPC, CFG 5), 21 latent frames = 7 blocks of 3. 26 clips train, 6 held out. All fine-tunes: 400 steps,
batch 2, lr 1e-5, AdamW, loss only on the block after the drifted context, offline self-rollouts from `ode_init`.

Targets. alpha=0 baseline: y = x0. T3 (ours): y = x0 + alpha (F(x_t|c_gt) - F(x_t|c_gen)). T2 (Swift original):
y = x0 + alpha (x0 - F(x_t|c_gen)). F = frozen init.

## Step 2a: is the extrapolation direction right? (32 clips, 384 points)

Prefix 2 GT blocks, roll the student 3 blocks. Latent drift vs the distance to an unrelated clip: 13% / 26% / 35%
at depth 1/2/3, so the GT frame is still a valid label. Drifted context doubles the prediction error at t=625.

beta = <x0 - F(c_gen), d> / |d|^2 with d = F(c_gt) - F(c_gen):

| t | m=1 | m=2 | m=3 |
| --- | --- | --- | --- |
| 625 | 1.19 ± 0.01 | 1.17 ± 0.01 | 1.14 ± 0.01 |
| 833 | 1.19 ± 0.02 | 1.19 ± 0.01 | 1.17 ± 0.01 |
| 937 | 1.11 ± 0.04 | 1.21 ± 0.02 | 1.19 ± 0.02 |
| 1000 | 0.65 ± 0.11 | 1.11 ± 0.07 | 1.18 ± 0.05 |

Per-clip beta > 0 in 100% of cases except (m=1, t=1000): 81%. cos 0.55-0.74. Direction passes.

## Step 2b: under-correction kappa after the alpha=0 fine-tune (6 held-out clips)

kappa = sum <S - F, x0 - F> / sum |x0 - F|^2 at (x_t | c_gen). Saturated between step 200 and 400.

| t | kappa @200 | kappa @400 | err_student / err_init @400 | proj of update on d @400 | |update| / |residual| |
| --- | --- | --- | --- | --- | --- |
| 625 | 0.33-0.35 | 0.33-0.35 | 0.75-0.77 | 0.46-0.56 | 0.65-0.68 |
| 833 | 0.40-0.43 | 0.42-0.44 | 0.64-0.66 | 0.55-0.64 | 0.70-0.73 |
| 937 | 0.50-0.52 | 0.52-0.55 | 0.49-0.56 | 0.74-0.79 | 0.77 |
| 1000 | 0.70-0.73 | 0.70-0.74 | 0.26-0.30 | 0.44-1.11 | 0.83-0.86 |

Reading: the learned update is mostly along d (context correction) and moves far (|u|/|r| ~ 0.7-0.85) while
projecting only 0.34-0.55 on the residual: a large component is regression toward the mean, which for a few-step
sampler shows up as frozen, washed-out video. The t=1000 row is dominated by that effect, not by exposure-bias
correction. Implied alpha* = 1/kappa - 1: 1.9 / 1.3 / 0.8 / 0.4 for t = 625 / 833 / 937 / 1000.

T3 alpha=1 kappa @400: 0.41 / 0.55 / 0.70 / 0.90 (by construction, not a gain).

## Long rollouts: 6 held-out clips x 2 seeds, 2 GT blocks + 20 generated blocks (~15 s), local attention 21 frames

| metric | init | alpha=0 | T3 alpha=1 | teacher data |
| --- | --- | --- | --- | --- |
| latent MSE to GT, generated block 1 / 5 | 0.37 / 0.76 | 0.09 / 0.24 | 0.13 / 0.53 | 0 |
| statistics distance to teacher data, block 0 / 9 / 19 | 3.0 / 4.9 / 5.9 | 2.4 / 2.3 / 2.3 | 2.2 / 2.1 / 3.0 | low and flat |
| per-channel latent std, block 0 / 19 | 0.58 / 0.27 | 0.69 / 0.84 | 0.74 / 1.21 | stable |
| motion (consecutive-frame MSE), blocks 1-9 | 0.09-0.19 | 0.02-0.04 | 0.09-0.12 | 0.083 |
| motion, blocks 16-19 | 0.11-0.13 | 0.013-0.017 | 0.04 | 0.083 |

Per-clip late-horizon (blocks 15-19) motion: alpha=0 0.006-0.030 on all 6 clips (frozen); T3 0.017-0.069.

Visual (clip 26, 1.6 s / 8 s / 15.5 s): `results/grid26_3.jpg`. init: noise-like drift to a flat colour.
alpha=0: structure survives longer, then freezes and smears into stripes. T3 alpha=1: motion and structure survive
to ~8 s, but colours over-saturate and streak (CFG-style over-extrapolation compounding over the rollout).

## Interpretation

* Direction (beta > 1) and under-correction (kappa 0.34-0.55 at the noise levels that matter) both hold on the real
  model, so the mechanism identified in the toy is present.
* alpha=0 trades noise-drift for freeze: MSE regression on GT frames from drifted contexts pulls the few-step
  sampler toward the conditional mean ("do not move").
* T3 alpha=1 pushes the model off the mean (motion restored, early statistics closest to the teacher) but overshoots:
  channel std climbs to 1.21 and the drift returns after ~10 blocks. A single alpha=1 is 2.5x the implied alpha*
  at t=1000, where the context dominates the prediction.
* Open experiment (queued, lost twice to sandbox termination): T3 with alpha(t) = 1.9 / 1.3 / 0.8 / 0.4, and T2
  alpha=1 (its extrapolation also covers the sample-vs-mean residual, so it is the candidate against freezing;
  the toy found T2 better than T3 when branching is small, which is the regime here: drift is 35% of unrelated).

## Caveats

6 held-out clips; the 400-step regression fine-tunes are an order of magnitude below Rolling Forcing's DMD
post-training in compute and are not comparable to it; only the relative question "does the extrapolation term add
value over alpha=0" is addressed.

## Reproduce

```
bash setup_sandbox.sh                      # ~8 min, one-shot environment
nohup bash run_pipeline.sh > /marimo/work/pipeline.log 2>&1 &   # resumable chain; ~12 h on one RTX PRO 6000
bash pull_results.sh <url> <token>         # from the Claude side, copies jsonl results into results/
```
