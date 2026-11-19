"""
Turning the knobs of A2C on CartPole-v1 (2 seeds each, 80,000 steps per run).

    python experiments.py nsteps     n_steps = 1, 5, 20   (how far to look before trusting the critic)
    python experiments.py entropy    ent_coef = 0, 0.01, 0.1 (how strongly to reward being unsure)
    python experiments.py shared     separate networks vs one shared body with two heads

Each group takes about 2-4 minutes and saves experiments_<group>.png.
Everything else stays at the defaults: 8 games side by side, lr 7e-4, gamma 0.99.

Add --no-show to skip the chart window.
"""
import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from a2c import evaluate, train_a2c

sys.stdout.reconfigure(encoding="utf-8")
STEPS = 80_000
SEEDS = [0, 1]
GROUPS = {
    "nsteps": [("n_steps = 1", {"n_steps": 1}), ("n_steps = 5", {"n_steps": 5}),
               ("n_steps = 20", {"n_steps": 20})],
    "entropy": [("ent_coef = 0", {"ent_coef": 0.0}), ("ent_coef = 0.01", {"ent_coef": 0.01}),
                ("ent_coef = 0.1", {"ent_coef": 0.1})],
    "shared": [("separate networks", {"shared": False}), ("shared body", {"shared": True})],
}


def curve(hist, grid, window=20):
    ends, ret = np.array(hist["ep_end_step"]), np.array(hist["ep_return"])
    return np.array([np.mean(ret[ends <= g][-window:]) if (ends <= g).any() else np.nan for g in grid])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    grid = np.arange(2_000, STEPS + 1, 2_000)

    print(f"{'setting':<20}{'seed':>5}{'updates':>9}{'last-50 avg':>13}{'exam':>7}"
          f"{'final entropy':>15}{'time':>7}")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    for k, (label, kw) in enumerate(GROUPS[args.group]):
        curves, ents = [], []
        for seed in SEEDS:
            t = time.time()
            model, h = train_a2c("CartPole-v1", total_steps=STEPS, seed=seed, **kw)
            exam = np.mean(evaluate(model, games=10))
            last = np.mean(h["ep_return"][-50:])
            print(f"{label:<20}{seed:>5}{len(h['update_step']):>9}{last:>13.1f}{exam:>7.1f}"
                  f"{np.mean(h['entropy'][-50:]):>15.3f}{time.time() - t:>6.0f}s")
            curves.append(curve(h, grid))
            ents.append((h["update_step"], h["entropy"]))
        color = f"C{k}"
        axs[0].plot(grid, np.nanmean(curves, axis=0), color=color, lw=2, label=f"{label} (mean of 2 seeds)")
        for c in curves:
            axs[0].plot(grid, c, color=color, lw=0.8, alpha=0.35)
        steps, e = ents[0]
        w = 50
        axs[1].plot(steps[w - 1:], np.convolve(e, np.ones(w) / w, mode="valid"), color=color, lw=1.5, label=label)
    axs[0].set(xlabel="Environment steps", ylabel="Average score, last 20 games",
               title=f"CartPole A2C: changing {args.group} (thin = single seeds)")
    axs[0].axhline(500, color="gray", ls="--", lw=1)
    axs[0].legend(loc="upper left", fontsize=9)
    axs[0].grid(alpha=0.3)
    axs[1].set(xlabel="Environment steps", ylabel="Entropy (seed 0, smoothed)", title="Actor entropy")
    axs[1].axhline(np.log(2), color="gray", ls="--", lw=1)
    axs[1].legend(fontsize=9)
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"experiments_{args.group}.png", dpi=110, bbox_inches="tight")
    print(f"\nChart saved to experiments_{args.group}.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
