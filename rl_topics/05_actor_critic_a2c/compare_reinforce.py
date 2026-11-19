"""
A2C vs plain REINFORCE on CartPole-v1, SAME budget of environment steps, 3 seeds each.

REINFORCE (one network, waits for the end of each game, weights each step by the
real return G_t) vs A2C (actor + critic, learns every 5 steps, weights each step
by the advantage = G - V(s)).

We measure:
    - the learning curves of every seed (how jumpy? how fast?)
    - the "weights" multiplying grad log pi: G_t for REINFORCE, the advantage for A2C.
      Their spread (standard deviation) is the noise in each learning signal.

Run:  python compare_reinforce.py           (about 4 minutes)
      python compare_reinforce.py --no-show
"""
import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from a2c import evaluate, train_a2c
from reinforce import train_reinforce

sys.stdout.reconfigure(encoding="utf-8")
STEPS = 120_000
SEEDS = [0, 1, 2]


def curve(hist, grid, window=20):
    """Average score of the last `window` games, read off at each step in `grid`."""
    ends, ret = np.array(hist["ep_end_step"]), np.array(hist["ep_return"])
    out = []
    for g in grid:
        done = ret[ends <= g][-window:]
        out.append(done.mean() if len(done) else np.nan)
    return np.array(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    grid = np.arange(2_000, STEPS + 1, 2_000)
    results = {"REINFORCE": [], "A2C": []}

    for seed in SEEDS:
        t = time.time()
        actor, h = train_reinforce("CartPole-v1", total_steps=STEPS, lr=1e-3, seed=seed)
        score = np.mean(evaluate(lambda o: (actor(o), None), games=20))
        results["REINFORCE"].append((h, score))
        print(f"REINFORCE seed {seed}: {len(h['ep_return']):>4} games, {len(h['ep_return'])} updates, "
              f"exam (greedy, 20 games) {score:5.1f}   [{time.time() - t:.0f}s]")
        t = time.time()
        model, h = train_a2c("CartPole-v1", total_steps=STEPS, seed=seed)
        score = np.mean(evaluate(model, games=20))
        results["A2C"].append((h, score))
        print(f"A2C       seed {seed}: {len(h['ep_return']):>4} games, {len(h['update_step'])} updates, "
              f"exam (greedy, 20 games) {score:5.1f}   [{time.time() - t:.0f}s]")

    print("\n" + "=" * 70)
    print(f"SUMMARY ({STEPS:,} environment steps each)")
    print("=" * 70)
    print("                        avg score of last 50 training games      exam (greedy)")
    print("                         seed 0   seed 1   seed 2   mean           mean")
    for name, runs in results.items():
        last = [np.mean(h["ep_return"][-50:]) for h, _ in runs]
        exams = [s for _, s in runs]
        print(f"  {name:<12}          " + "".join(f"{x:8.1f} " for x in last)
              + f"{np.mean(last):7.1f}        {np.mean(exams):7.1f}")
    # How jumpy is each curve? average absolute change of the 20-game average between checkpoints
    for name, runs in results.items():
        jumps = [np.nanmean(np.abs(np.diff(curve(h, grid)))) for h, _ in runs]
        print(f"  {name:<12} avg jump of the curve between 2k-step checkpoints: {np.mean(jumps):5.1f}")
    print("\n  Noise in the learning signal (std of the weight on grad log pi, average over updates):")
    r_std = np.mean([np.mean(h["weight_std"]) for h, _ in results["REINFORCE"]])
    a_std = np.mean([np.mean(h["adv_std"]) for h, _ in results["A2C"]])
    print(f"    REINFORCE  weight = G_t          std {r_std:6.2f}")
    print(f"    A2C        weight = advantage    std {a_std:6.2f}")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (name, runs), color in zip(axs, results.items(), ["tab:red", "tab:blue"]):
        for (h, _), seed in zip(runs, SEEDS):
            ax.plot(grid, curve(h, grid), color=color, alpha=0.4 + 0.2 * seed, lw=1.6, label=f"seed {seed}")
        ax.axhline(500, color="gray", ls="--", lw=1)
        ax.set(title=f"{name}: 3 seeds, {STEPS:,} steps each", xlabel="Environment steps",
               ylabel="Average score, last 20 games")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("compare_reinforce.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to compare_reinforce.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
