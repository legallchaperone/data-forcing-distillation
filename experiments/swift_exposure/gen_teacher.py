"""Generate pseudo-ground-truth video latents with the bidirectional Wan2.1-T2V-1.3B teacher.

Run from inside the Self-Forcing repo root (paths like wan_models/... are relative):
    cd /marimo/work/sf && python /path/to/gen_teacher.py --out /marimo/work/data/teacher --n 64

Each output file  <out>/<idx:05d>.pt  holds {"latent": [21,16,60,104] fp16, "prompt": str, "seed": int}.
"""
import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from pipeline import BidirectionalDiffusionInferencePipeline  # noqa: E402


def sample_latents(pipe, noise, prompt, config):
    """Same loop as BidirectionalDiffusionInferencePipeline.inference, without the VAE decode."""
    cond = pipe.text_encoder(text_prompts=[prompt])
    uncond = pipe.text_encoder(text_prompts=[config.negative_prompt])
    latents = noise
    sched = pipe._initialize_sample_scheduler(noise)
    for t in sched.timesteps:
        timestep = t * torch.ones([latents.shape[0], 21], device=noise.device, dtype=torch.float32)
        flow_c, _ = pipe.generator(latents, cond, timestep)
        flow_u, _ = pipe.generator(latents, uncond, timestep)
        flow = flow_u + config.guidance_scale * (flow_c - flow_u)
        latents = sched.step(flow.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False)[0].squeeze(0)
    return latents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="prompts/MovieGenVideoBench.txt")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                             OmegaConf.load("configs/self_forcing_dmd.yaml"))
    config.guidance_scale = args.guidance
    config.model_kwargs = {"timestep_shift": 8.0}

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    pipe = BidirectionalDiffusionInferencePipeline(config, device=device)
    pipe.sampling_steps = args.steps
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device)
    pipe.generator.to(device)
    pipe.vae.to(device)

    prompts = [l.strip() for l in open(args.prompts) if l.strip()]
    for idx in range(args.start, args.start + args.n):
        path = os.path.join(args.out, f"{idx:05d}.pt")
        if os.path.exists(path):
            continue
        prompt = prompts[idx % len(prompts)]
        g = torch.Generator(device=device).manual_seed(args.seed * 100000 + idx)
        noise = torch.randn([1, 21, 16, 60, 104], device=device, dtype=torch.bfloat16, generator=g)
        t0 = time.time()
        latents = sample_latents(pipe, noise, prompt, config)
        torch.save({"latent": latents[0].to(torch.float16).cpu(), "prompt": prompt, "seed": idx}, path)
        print(f"[{idx}] {time.time() - t0:.1f}s  {prompt[:60]}", flush=True)


if __name__ == "__main__":
    main()
