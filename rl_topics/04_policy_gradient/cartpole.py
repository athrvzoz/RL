"""
REINFORCE on CartPole: with and without a baseline (3 seeds each),
plus the "whole-game return" variant (no returns-to-go).

Prints learning progress, test scores (rolling the policy's dice, and greedy),
the policy's action probabilities in some states, and saves cartpole_reinforce.png.

Run:  python cartpole.py            (about 5-8 minutes)
      python cartpole.py --no-show
"""
import argparse
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from reinforce import evaluate, moving_average, train

sys.stdout.reconfigure(encoding="utf-8")

VARIANTS = {
    "no baseline": dict(baseline="none", rtg=True),
    "with baseline": dict(baseline="mean", rtg=True),
    "whole-game return": dict(baseline="none", rtg=False),
}
COLORS = {"no baseline": "#C47A00", "with baseline": "#2C4BD1", "whole-game return": "#9AA3B9"}
SHOW_STATES = {
    "balanced, still": [0.0, 0.0, 0.00, 0.0],
    "pole leaning RIGHT": [0.0, 0.0, 0.10, 0.0],
    "pole leaning LEFT": [0.0, 0.0, -0.10, 0.0],
    "pole falling right fast": [0.0, 0.0, 0.05, 1.0],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    env = gym.make("CartPole-v1")

    results = {}
    for name, kw in VARIANTS.items():
        runs = []
        for seed in range(args.seeds):
            policy, scores = train(env, episodes=args.episodes, seed=seed, **kw)
            test = evaluate(env, policy, games=20)
            test_greedy = evaluate(env, policy, games=20, greedy=True)
            runs.append((policy, scores, test, test_greedy))
            print(f"  {name:<18} seed {seed}: avg score games 1-100 {scores[:100].mean():6.1f} | "
                  f"last 100 {scores[-100:].mean():6.1f} | test (dice) {test:6.1f} | test (greedy) {test_greedy:6.1f}")
        results[name] = runs
        print()

    print("Summary (mean over seeds)")
    for name, runs in results.items():
        last = np.mean([r[1][-100:].mean() for r in runs])
        spread = np.mean([r[1][-100:].std() for r in runs])
        print(f"  {name:<18} last-100 avg {last:6.1f}  (game-to-game spread {spread:5.1f})   "
              f"test dice {np.mean([r[2] for r in runs]):6.1f}   greedy {np.mean([r[3] for r in runs]):6.1f}")

    best = max(results["with baseline"], key=lambda r: r[2])[0]
    print("\nAction probabilities of the best 'with baseline' policy:")
    with torch.no_grad():
        for name, st in SHOW_STATES.items():
            p = best(torch.tensor(st, dtype=torch.float32)).numpy()
            print(f"  {name:<26} P(left) {p[0]:.3f}   P(right) {p[1]:.3f}")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, runs in results.items():
        curve = np.mean([moving_average(r[1]) for r in runs], axis=0)
        ax.plot(np.arange(19, 19 + len(curve)), curve, lw=2, color=COLORS[name], label=name)
    ax.axhline(500, color="gray", ls="--", lw=1)
    ax.set_xlabel("Training game")
    ax.set_ylabel(f"Score (avg of last 20 games, mean of {args.seeds} seeds)")
    ax.set_title("REINFORCE on CartPole")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("cartpole_reinforce.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to cartpole_reinforce.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
