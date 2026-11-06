"""Step 2a: measure the exposure-bias signal on the real model (Self-Forcing ode_init.pt, Wan2.1-1.3B causal).

For each teacher video z (21 latent frames = 7 blocks of 3):
  * prefix = first 2 blocks of z (ground truth)
  * roll the causal 4-step student for 3 blocks from the prefix -> c_gen (depth m = 1, 2, 3 are its prefixes)
  * for the block right after depth m, at each of the 4 student timesteps t, with ONE shared noise draw:
        F_gt  = F(x_t | c_gt)   teacher-forced on the true video
        F_gen = F(x_t | c_gen)  teacher-forced on the drifted context
        d     = F_gt - F_gen
    and record  err_gt, err_gen, |d|^2, beta = <x0 - F_gen, d>/<d, d>, cos, and the latent drift |c_gen - c_gt|^2.
Also saves c_gen latents for the alpha=0 fine-tune.

Run from the Self-Forcing repo root:
    cd /marimo/work/sf && python /path/to/measure_kappa.py --data /marimo/work/data/teacher --out /marimo/work/data/kappa
"""
import argparse
import glob
import json
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from pipeline import CausalInferencePipeline  # noqa: E402

FRAME_SEQ = 1560
BLK = 3


def load_pipeline(ckpt="checkpoints/ode_init.pt", device="cuda"):
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                             OmegaConf.load("configs/self_forcing_dmd.yaml"))
    pipe = CausalInferencePipeline(config, device=device)
    state = torch.load(ckpt, map_location="cpu")
    pipe.generator.load_state_dict(state["generator"])
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device)
    pipe.generator.to(device)
    return pipe, config


@torch.no_grad()
def rollout(pipe, cond, initial_latent, num_blocks, gen):
    """Mirror of CausalInferencePipeline.inference (video-extension mode) without VAE decode.
    initial_latent: [1, 3k, C, H, W] clean context. Returns the generated [1, 3*num_blocks, C, H, W]."""
    B, dev, dt = 1, initial_latent.device, initial_latent.dtype
    pipe._initialize_kv_cache(B, dt, dev)
    pipe._initialize_crossattn_cache(B, dt, dev)
    steps = pipe.denoising_step_list
    cur = 0
    zeros = torch.zeros([B, BLK], device=dev, dtype=torch.int64)
    for _ in range(initial_latent.shape[1] // BLK):
        pipe.generator(noisy_image_or_video=initial_latent[:, cur:cur + BLK], conditional_dict=cond, timestep=zeros,
                       kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache, current_start=cur * FRAME_SEQ)
        cur += BLK
    outs = []
    shape = [B, BLK] + list(initial_latent.shape[2:])
    for _ in range(num_blocks):
        noisy = torch.randn(shape, device=dev, dtype=dt, generator=gen)
        for i, t in enumerate(steps):
            ts = torch.ones([B, BLK], device=dev, dtype=torch.int64) * t
            _, x0 = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts,
                                   kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                                   current_start=cur * FRAME_SEQ)
            if i < len(steps) - 1:
                nxt = steps[i + 1] * torch.ones([B * BLK], device=dev, dtype=torch.long)
                noisy = pipe.scheduler.add_noise(x0.flatten(0, 1), torch.randn(x0.flatten(0, 1).shape, device=dev, dtype=dt, generator=gen), nxt).unflatten(0, (B, BLK))
        outs.append(x0)
        pipe.generator(noisy_image_or_video=x0, conditional_dict=cond, timestep=zeros,
                       kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache, current_start=cur * FRAME_SEQ)
        cur += BLK
    return torch.cat(outs, dim=1)


@torch.no_grad()
def tf_predict(pipe, cond, clean_video, noisy_video, t):
    """Teacher-forced x0 prediction for every block of noisy_video given clean_video as context."""
    B, F = clean_video.shape[:2]
    ts = torch.full([B, F], int(t), device=clean_video.device, dtype=torch.int64)
    _, pred = pipe.generator(noisy_image_or_video=noisy_video, conditional_dict=cond, timestep=ts, clean_x=clean_video)
    return pred


def mse(a, b):
    return ((a.float() - b.float()) ** 2).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--prefix_blocks", type=int, default=2)
    ap.add_argument("--depths", type=str, default="1,2,3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "cgen"), exist_ok=True)
    depths = [int(x) for x in args.depths.split(",")]
    max_depth = max(depths)

    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    pipe, config = load_pipeline(device=dev)
    steps = [int(t) for t in pipe.denoising_step_list.tolist()]
    print("student timesteps:", steps, flush=True)

    files = sorted(glob.glob(os.path.join(args.data, "*.pt")))[: args.n]
    assert len(files) >= 2, "need at least two teacher videos"
    all_latents = [torch.load(f)["latent"] for f in files]
    rows_path = os.path.join(args.out, "rows.jsonl")
    done = set()
    if os.path.exists(rows_path):
        done = {json.loads(l)["idx"] for l in open(rows_path)}
    fout = open(rows_path, "a")

    for vi, f in enumerate(files):
        idx = int(os.path.basename(f).split(".")[0])
        if idx in done:
            continue
        t0 = time.time()
        item = torch.load(f)
        z = item["latent"].to(dev, torch.bfloat16).unsqueeze(0)  # [1,21,C,H,W]
        cond = pipe.text_encoder(text_prompts=[item["prompt"]])
        other = all_latents[(vi + 1) % len(all_latents)].to(dev, torch.bfloat16).unsqueeze(0)

        p = args.prefix_blocks * BLK
        gen = torch.Generator(device=dev).manual_seed(args.seed * 100000 + idx)
        cgen = rollout(pipe, cond, z[:, :p], max_depth, gen)  # [1, 3*max_depth, ...]
        torch.save(cgen[0].to(torch.float16).cpu(), os.path.join(args.out, "cgen", f"{idx:05d}.pt"))

        for t in steps:
            noise = torch.randn(z.shape, device=dev, dtype=z.dtype, generator=gen)
            tt = torch.full([z.shape[0] * z.shape[1]], t, device=dev, dtype=torch.long)
            noisy = pipe.scheduler.add_noise(z.flatten(0, 1), noise.flatten(0, 1), tt).unflatten(0, z.shape[:2])
            pred_gt = tf_predict(pipe, cond, z, noisy, t)
            for m in depths:
                tb = p + m * BLK  # target block start frame
                if tb + BLK > z.shape[1]:
                    continue
                clean_gen = torch.cat([z[:, :p], cgen[:, : m * BLK], z[:, tb:]], dim=1)
                pred_gen = tf_predict(pipe, cond, clean_gen, noisy, t)
                x0 = z[:, tb:tb + BLK].float()
                Fgt = pred_gt[:, tb:tb + BLK].float()
                Fgen = pred_gen[:, tb:tb + BLK].float()
                d = Fgt - Fgen
                r = x0 - Fgen
                dd = (d * d).sum().item()
                rd = (r * d).sum().item()
                row = dict(
                    idx=idx, m=m, t=t,
                    err_gt=mse(Fgt, x0), err_gen=mse(Fgen, x0), d2=(d * d).mean().item(),
                    beta=rd / max(dd, 1e-12),
                    cos=rd / max((d * d).sum().sqrt().item() * (r * r).sum().sqrt().item(), 1e-12),
                    drift=mse(cgen[:, (m - 1) * BLK: m * BLK], z[:, tb - BLK: tb]),
                    drift_cum=mse(cgen[:, : m * BLK], z[:, p:tb]),
                    unrelated=mse(other[:, tb - BLK: tb], z[:, tb - BLK: tb]),
                    x0_var=(x0 * x0).mean().item(),
                    err_gt_allblocks=mse(pred_gt[:, BLK:], z[:, BLK:]),
                )
                fout.write(json.dumps(row) + "\n")
        fout.flush()
        print(f"[{idx}] {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
