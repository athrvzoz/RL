"""
Experiments: turn one knob at a time and watch what changes (10,000 steps per run).

  --part noise   DDPG exploration noise σ = 0 (none) or 0.5 (a lot)        ~2.5 min
  --part tau     DDPG τ = 1.0 (NO slow target network) or 0.05             ~2.5 min
  --part alpha   SAC temperature α fixed at 0 (no curiosity) or 1.0        ~3 min
  --part seeds   DDPG and SAC with seeds 1 and 2: was seed 0 just lucky?   ~5 min

The "normal" setting of each knob (σ = 0.1, τ = 0.005, α automatic, seed 0) is NOT
re-trained: it is the first 10,000 steps of the main run, read from results.json
(same code and seed; saves CPU time).

Run:  python pendulum_experiment.py      (first, to make results.json)
      python knobs.py --part noise
      python knobs.py --part alpha --no-show
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small memory footprint: other
os.environ.setdefault("OMP_NUM_THREADS", "2")        # programs share this computer
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

from common import train
from ddpg_td3 import DDPG
from sac import SAC

sys.stdout.reconfigure(encoding="utf-8")

STEPS = 10_000
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]

PARTS = {
    "noise": ("DDPG", "σ = 0.1 (normal)", [("σ = 0 (no noise)", lambda s: DDPG(noise=0.0, seed=s)),
                                           ("σ = 0.5 (lots)", lambda s: DDPG(noise=0.5, seed=s))]),
    "tau": ("DDPG", "τ = 0.005 (normal)", [("τ = 0.05", lambda s: DDPG(tau=0.05, seed=s)),
                                           ("τ = 1 (no target net)", lambda s: DDPG(tau=1.0, seed=s))]),
    "alpha": ("SAC", "α automatic (normal)", [("α = 0 (no curiosity)", lambda s: SAC(alpha=0.0, seed=s)),
                                              ("α = 1 (very curious)", lambda s: SAC(alpha=1.0, seed=s))]),
    "seeds": (None, None, [("DDPG seed 1", lambda s: DDPG(seed=1)), ("DDPG seed 2", lambda s: DDPG(seed=2)),
                           ("SAC seed 1", lambda s: SAC(seed=1)), ("SAC seed 2", lambda s: SAC(seed=2))]),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=list(PARTS), required=True)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    base_algo, base_label, variants = PARTS[args.part]
    with open("results.json") as f:
        main_run = json.load(f)
    curves = {}
    if base_algo:     # the normal setting: reuse the main run
        r = main_run[base_algo]
        curves[base_label] = (r["steps"][:10], r["eval"][:10])
    else:
        for algo in ("DDPG", "SAC"):
            r = main_run[algo]
            curves[f"{algo} seed 0"] = (r["steps"][:10], r["eval"][:10])

    for label, make in variants:
        seed = int(label[-1]) if args.part == "seeds" else 0
        print(f"\n{label}: {STEPS:,} steps")
        log = train(make(seed), total_steps=STEPS, seed=seed)
        curves[label] = (log["step"], log["eval"])

    print("\n" + "=" * 72)
    print(f"PART '{args.part}': test score (avg of 5 games) at each 1,000 steps")
    print("=" * 72)
    for label, (steps, scores) in curves.items():
        print(f"  {label:24}" + "".join(f"{v:7.0f}" for v in scores))
    print(f"  {'':24}" + "".join(f"{s // 1000:6}k" for s in curves[next(iter(curves))][0]))

    out = {k: {"steps": v[0], "eval": v[1]} for k, v in curves.items()}
    with open(f"knobs_{args.part}.json", "w") as f:
        json.dump(out, f, indent=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    for c, (label, (steps, scores)) in zip(COLORS, curves.items()):
        ax.plot(steps, scores, lw=2, marker="o", ms=4, color=c, label=label)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Test score (avg of 5 games)")
    ax.set_title(f"Pendulum-v1, changing one knob: {args.part}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(f"knobs_{args.part}.png", dpi=120, bbox_inches="tight")
    print(f"\nSaved knobs_{args.part}.png and knobs_{args.part}.json")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
