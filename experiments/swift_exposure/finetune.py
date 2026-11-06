"""Step 2b: fine-tune the causal student on self-rollout (drifted) contexts with ground-truth labels, and measure kappa.

Targets (per toy notebook):
  alpha = 0        baseline  y = x0
  --mode T3        ours      y = x0 + alpha * (F(x_t|c_gt) - F(x_t|c_gen))      (F = frozen init)
  --mode T2        Swift     y = x0 + alpha * (x0 - F(x_t|c_gen))
Only the block right after the drifted context carries loss. Contexts come from measure_kappa.py (offline rollouts).

kappa on held-out videos:  kappa = sum <S - F, x0 - F> / sum |x0 - F|^2   with everything evaluated at (x_t | c_gen).

Run from the Self-Forcing repo root:
    cd /marimo/work/sf && python /path/to/finetune.py --data /marimo/work/data/teacher --cgen /marimo/work/data/kappa/cgen \
        --out /marimo/work/data/ft_a0 --alpha 0 --steps 400
"""
import argparse
import copy
import glob
import json
import os
import random
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from pipeline import CausalInferencePipeline  # noqa: E402

BLK = 3


def build(args, dev):
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                             OmegaConf.load("configs/self_forcing_dmd.yaml"))
    pipe = CausalInferencePipeline(config, device=dev)
    state = torch.load("checkpoints/ode_init.pt", map_location="cpu")
    pipe.generator.load_state_dict(state["generator"])
    pipe.text_encoder.to(dev, torch.bfloat16)
    pipe.generator.to(dev)  # fp32 master weights, bf16 autocast in forward
    steps = [int(t) for t in pipe.denoising_step_list.tolist()]
    return pipe, steps


def load_data(args, dev, pipe):
    files = sorted(glob.glob(os.path.join(args.data, "*.pt")))
    items = []
    for f in files:
        idx = int(os.path.basename(f).split(".")[0])
        cg = os.path.join(args.cgen, f"{idx:05d}.pt")
        if not os.path.exists(cg):
            continue
        d = torch.load(f)
        items.append(dict(idx=idx, z=d["latent"].to(torch.bfloat16), cgen=torch.load(cg).to(torch.bfloat16), prompt=d["prompt"]))
    with torch.no_grad():
        for it in items:
            it["cond"] = {k: v.to(torch.bfloat16).cpu() for k, v in pipe.text_encoder(text_prompts=[it["prompt"]]).items()}
    pipe.text_encoder.cpu()
    torch.cuda.empty_cache()
    n_hold = max(4, len(items) // 5)
    return items[:-n_hold], items[-n_hold:]


def make_batch(it, m, p, dev):
    z = it["z"].to(dev).unsqueeze(0)
    cg = it["cgen"].to(dev).unsqueeze(0)
    tb = p + m * BLK
    clean_gen = torch.cat([z[:, :p], cg[:, : m * BLK], z[:, tb:]], dim=1)
    cond = {k: v.to(dev) for k, v in it["cond"].items()}
    return z, clean_gen, cond, tb


def tf_pred(gen, cond, clean, noisy, t):
    ts = torch.full(noisy.shape[:2], int(t), device=noisy.device, dtype=torch.int64)
    _, pred = gen(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts, clean_x=clean)
    return pred


@torch.no_grad()
def eval_kappa(student, frozen, sched, items, steps, p, dev, depths, n_noise=1, seed=123):
    g = torch.Generator(device=dev).manual_seed(seed)
    acc = {}
    for it in items:
        for m in depths:
            z, clean_gen, cond, tb = make_batch(it, m, p, dev)
            x0 = z[:, tb:tb + BLK].float()
            for t in steps:
                for _ in range(n_noise):
                    noise = torch.randn(z.shape, device=dev, dtype=z.dtype, generator=g)
                    tt = torch.full([z.shape[1]], t, device=dev, dtype=torch.long)
                    noisy = sched.add_noise(z.flatten(0, 1), noise.flatten(0, 1), tt).unflatten(0, z.shape[:2])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        S = tf_pred(student, cond, clean_gen, noisy, t)[:, tb:tb + BLK].float()
                        F = tf_pred(frozen, cond, clean_gen, noisy, t)[:, tb:tb + BLK].float()
                        Fgt = tf_pred(frozen, cond, z, noisy, t)[:, tb:tb + BLK].float()
                    r, u, d = x0 - F, S - F, Fgt - F
                    a = acc.setdefault((m, t), dict(ur=0.0, rr=0.0, uu=0.0, ud=0.0, dd=0.0, errS=0.0, errF=0.0, n=0))
                    a["ur"] += (u * r).sum().item(); a["rr"] += (r * r).sum().item(); a["uu"] += (u * u).sum().item()
                    a["ud"] += (u * d).sum().item(); a["dd"] += (d * d).sum().item()
                    a["errS"] += ((S - x0) ** 2).mean().item(); a["errF"] += (r ** 2).mean().item(); a["n"] += 1
    rows = []
    for (m, t), a in acc.items():
        rows.append(dict(m=m, t=t, kappa=a["ur"] / max(a["rr"], 1e-12), proj_d=a["ud"] / max(a["dd"], 1e-12),
                         rel_move=(a["uu"] / max(a["rr"], 1e-12)) ** 0.5,
                         err_student=a["errS"] / a["n"], err_init=a["errF"] / a["n"], n=a["n"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cgen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--mode", default="T3", choices=["T3", "T2"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--prefix_blocks", type=int, default=2)
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_ckpt", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    depths = [int(x) for x in args.depths.split(",")]
    p = args.prefix_blocks * BLK

    pipe, steps = build(args, dev)
    train_items, hold_items = load_data(args, dev, pipe)
    print(f"train videos {len(train_items)}  held-out {len(hold_items)}  timesteps {steps}", flush=True)

    student = pipe.generator
    frozen = copy.deepcopy(student).to(torch.bfloat16).eval().requires_grad_(False)
    student.model.requires_grad_(True)
    student.enable_gradient_checkpointing()
    sched = pipe.scheduler
    opt = torch.optim.AdamW([q for q in student.parameters() if q.requires_grad], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    kap = open(os.path.join(args.out, "kappa.jsonl"), "a")

    def do_eval(step):
        student.eval()
        rows = eval_kappa(student, frozen, sched, hold_items, steps, p, dev, depths)
        for r in rows:
            r["step"] = step; kap.write(json.dumps(r) + "\n")
        kap.flush()
        km = sum(r["kappa"] for r in rows) / len(rows)
        print(f"[eval step {step}] mean kappa {km:.3f}  " + "  ".join(f"m{r['m']}t{r['t']}:{r['kappa']:.2f}" for r in sorted(rows, key=lambda r: (r['m'], -r['t']))), flush=True)
        student.train()

    do_eval(0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.accum):
            it = random.choice(train_items); m = random.choice(depths); t = random.choice(steps)
            z, clean_gen, cond, tb = make_batch(it, m, p, dev)
            noise = torch.randn_like(z)
            tt = torch.full([z.shape[1]], t, device=dev, dtype=torch.long)
            noisy = sched.add_noise(z.flatten(0, 1), noise.flatten(0, 1), tt).unflatten(0, z.shape[:2])
            x0 = z[:, tb:tb + BLK].float()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if args.alpha == 0:
                    y = x0
                elif args.mode == "T3":
                    Fgen = tf_pred(frozen, cond, clean_gen, noisy, t)[:, tb:tb + BLK].float()
                    Fgt = tf_pred(frozen, cond, z, noisy, t)[:, tb:tb + BLK].float()
                    y = x0 + args.alpha * (Fgt - Fgen)
                else:
                    Fgen = tf_pred(frozen, cond, clean_gen, noisy, t)[:, tb:tb + BLK].float()
                    y = x0 + args.alpha * (x0 - Fgen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = tf_pred(student, cond, clean_gen, noisy, t)[:, tb:tb + BLK].float()
            loss = torch.nn.functional.mse_loss(pred, y) / args.accum
            loss.backward()
            tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0).item()
        opt.step()
        log.write(json.dumps(dict(step=step, loss=tot, grad_norm=gn, t=t, m=m)) + "\n")
        if step % 10 == 0:
            log.flush()
            print(f"step {step} loss {tot:.4f} gn {gn:.2f} {(time.time() - t0) / step:.1f}s/it", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            do_eval(step)
    if args.save_ckpt:
        torch.save({"generator": {k: v.to(torch.bfloat16) for k, v in student.state_dict().items()}}, os.path.join(args.out, "student.pt"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
