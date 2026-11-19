"""
What does the clip actually buy? CartPole, 2 seeds × 100,000 steps per variant.

    PPO (clip 0.2)          the normal setting
    no clip (ε = 100)       the fence is so wide it never bites: plain "reuse the batch 10 times"
    no clip + lr ×10        same, with big steps: one bad update can wreck the policy
    clip 0.2 + lr ×10       big steps, but the fence still holds

Run:  python ablation.py            (about 6-10 minutes)
      python ablation.py --no-show
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from ppo import evaluate, smooth, train

sys.stdout.reconfigure(encoding="utf-8")

VARIANTS = {
    "PPO (clip 0.2)": dict(),
    "no clip": dict(clip=100.0),
    "no clip, lr ×10": dict(clip=100.0, lr=3e-3),
    "clip 0.2, lr ×10": dict(lr=3e-3),
}
COLORS = ["#2C4BD1", "#C47A00", "#C2413A", "#1E8A5A"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    results = {}
    for name, kw in VARIANTS.items():
        runs = []
        for seed in range(args.seeds):
            agent, hist, eps = train("CartPole-v1", args.steps, seed=seed, verbose=False, **kw)
            exam = np.mean(evaluate("CartPole-v1", agent, games=20))
            last = np.mean([r for _, r in eps[-50:]])
            max_kl = max(h["approx_kl"] for h in hist)
            biggest = max(max(h["max_ratio"], 1 / max(h["min_ratio"], 1e-6)) for h in hist)
            runs.append((hist, eps, exam, last))
            print(f"  {name:<18} seed {seed}: last-50 avg {last:6.1f} | exam {exam:6.1f} | "
                  f"largest approx-KL {max_kl:.3f} | largest prob change ×{biggest:.1f}")
        results[name] = runs
    print("\nSummary (mean of seeds)")
    for name, runs in results.items():
        print(f"  {name:<18} last-50 avg {np.mean([r[3] for r in runs]):6.1f}   exam {np.mean([r[2] for r in runs]):6.1f}"
              f"   avg approx-KL {np.mean([np.mean([h['approx_kl'] for h in r[0]]) for r in runs]):.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3))
    for (name, runs), c in zip(results.items(), COLORS):
        for k, (hist, eps, _, _) in enumerate(runs):
            s, r = zip(*eps)
            sm = smooth(r, 20)
            ax1.plot(s[len(s) - len(sm):], sm, color=c, lw=1.6, alpha=0.9,
                     ls="-" if k == 0 else "--", label=name if k == 0 else None)
            ax2.plot([h["step"] for h in hist], [h["approx_kl"] for h in hist], color=c, lw=1,
                     ls="-" if k == 0 else "--", label=name if k == 0 else None)
    ax1.set_title("Score (avg of 20 games); solid = seed 0, dashed = seed 1")
    ax1.set_xlabel("Environment steps")
    ax1.legend()
    ax2.set_yscale("log")
    ax2.set_title("approx-KL per update (log scale)")
    ax2.set_xlabel("Environment steps")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("ablation.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to ablation.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
