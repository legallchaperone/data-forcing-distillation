"""Long-rollout drift evaluation for a causal student checkpoint (init, alpha=0, or T3).

For each held-out teacher video: feed the first `prefix_blocks` GT blocks, then roll `blocks` blocks with the 4-step
student (KV cache with local attention so the rollout can exceed 21 frames). Per block we record
  * err_gt      : latent MSE to the teacher video while GT frames exist (first 7 blocks of the teacher clip)
  * ch_mean/std : per-channel latent mean / std of the block (drift of global statistics = colour / exposure drift)
  * stat_dist   : distance of (ch_mean, ch_std) to the teacher-dataset statistics, normalised by the dataset spread
  * motion      : MSE between consecutive latent frames inside the block (collapse to static or flicker)
Optionally decodes a few rollouts to mp4 for eyeballing.

Run from the Self-Forcing repo root:
    cd /marimo/work/sf && python /path/to/rollout_eval.py --ckpt checkpoints/ode_init.pt --ckpt_key generator \
        --data /marimo/work/data/teacher --out /marimo/work/data/roll_init --blocks 20
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


def load_pipeline(ckpt, key, local_attn, sink, dev):
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                             OmegaConf.load("configs/self_forcing_dmd.yaml"))
    config.model_kwargs = {"timestep_shift": 5.0, "local_attn_size": local_attn, "sink_size": sink}
    pipe = CausalInferencePipeline(config, device=dev)
    state = torch.load(ckpt, map_location="cpu")
    state = state[key] if key in state else state
    pipe.generator.load_state_dict(state)
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(dev)
    pipe.generator.to(dev)
    pipe.vae.to(dev)
    return pipe


@torch.no_grad()
def rollout(pipe, cond, initial_latent, num_blocks, gen):
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
                                   kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache, current_start=cur * FRAME_SEQ)
            if i < len(steps) - 1:
                nxt = steps[i + 1] * torch.ones([B * BLK], device=dev, dtype=torch.long)
                noisy = pipe.scheduler.add_noise(x0.flatten(0, 1), torch.randn(x0.flatten(0, 1).shape, device=dev, dtype=dt, generator=gen), nxt).unflatten(0, (B, BLK))
        outs.append(x0)
        pipe.generator(noisy_image_or_video=x0, conditional_dict=cond, timestep=zeros,
                       kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache, current_start=cur * FRAME_SEQ)
        cur += BLK
    return torch.cat(outs, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt_key", default="generator")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=26, help="first teacher video index to use (held-out split)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--prefix_blocks", type=int, default=2)
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--local_attn", type=int, default=21)
    ap.add_argument("--sink", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--decode", type=int, default=0, help="decode this many rollouts to mp4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    pipe = load_pipeline(args.ckpt, args.ckpt_key, args.local_attn, args.sink, dev)

    files = sorted(glob.glob(os.path.join(args.data, "*.pt")))
    all_z = torch.stack([torch.load(f)["latent"].float() for f in files])        # [N,21,C,H,W]
    ds_mean = all_z.mean(dim=(0, 1, 3, 4))                                      # [C]
    ds_std = all_z.std(dim=(0, 1, 3, 4))
    blk_means = all_z.unflatten(1, (7, BLK)).mean(dim=(2, 4, 5))                # [N,7,C]
    blk_stds = all_z.unflatten(1, (7, BLK)).std(dim=(2, 4, 5))
    spread_mean = blk_means.std(dim=(0, 1)) + 1e-6                              # natural per-block spread
    spread_std = blk_stds.std(dim=(0, 1)) + 1e-6
    sel = [f for f in files if args.start <= int(os.path.basename(f).split(".")[0]) < args.start + args.n]

    fout = open(os.path.join(args.out, "rows.jsonl"), "a")
    p = args.prefix_blocks * BLK
    for f in sel:
        idx = int(os.path.basename(f).split(".")[0])
        item = torch.load(f)
        z = item["latent"].to(dev, torch.bfloat16).unsqueeze(0)
        cond = pipe.text_encoder(text_prompts=[item["prompt"]])
        for s in range(args.seeds):
            t0 = time.time()
            gen = torch.Generator(device=dev).manual_seed(1000 * s + idx)
            roll = rollout(pipe, cond, z[:, :p], args.blocks, gen).float()        # [1, 3*blocks, C, H, W]
            full = torch.cat([z[:, :p].float(), roll], dim=1)
            for b in range(args.blocks):
                blk = roll[0, b * BLK:(b + 1) * BLK]                                # [3,C,H,W]
                gb = args.prefix_blocks + b
                err = ((blk - z[0, gb * BLK:(gb + 1) * BLK].float()) ** 2).mean().item() if gb < 7 else None
                cm_ = blk.mean(dim=(0, 2, 3)); cs_ = blk.std(dim=(0, 2, 3))
                stat_dist = (((cm_.cpu() - ds_mean) / spread_mean) ** 2).mean().sqrt().item() + \
                            (((cs_.cpu() - ds_std) / spread_std) ** 2).mean().sqrt().item()
                prev = full[0, p + b * BLK - 1]
                motion = ((blk[0] - prev) ** 2).mean().item()
                intra = ((blk[1:] - blk[:-1]) ** 2).mean().item()
                fout.write(json.dumps(dict(idx=idx, seed=s, block=b, err_gt=err, stat_dist=stat_dist,
                                           ch_mean_abs=cm_.abs().mean().item(), ch_std=cs_.mean().item(),
                                           motion=motion, intra_motion=intra)) + "\n")
            fout.flush()
            if args.decode > 0 and s == 0 and idx < args.start + args.decode:
                video = pipe.vae.decode_to_pixel(full.to(torch.bfloat16))[0]       # [F,3,H,W] in [-1,1]
                video = ((video * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                import imageio
                imageio.mimwrite(os.path.join(args.out, f"{idx:05d}.mp4"), video, fps=16, codec="libx264", quality=7)
            print(f"[{idx} seed {s}] {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
