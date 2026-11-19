"""
Turning the knobs on CliffWalking (cheap variations of cliff_walking.py).

Experiment 1: how much the agent explores (epsilon = 0.01, 0.1, 0.2).
              The more SARSA explores, the further from the cliff it walks.
Experiment 2: epsilon shrinks to almost 0 during training.
              Once the agent stops exploring, the edge is no longer dangerous
              for it, so SARSA ALSO ends up on the short edge path.
Experiment 3: the learning rate alpha (0.1, 0.5, 1.0) for all three methods.

Every number is an average over 20 training runs (seeds 0..19), 500 games each.

Run:  python experiments.py            (about 1-2 minutes)
      python experiments.py --no-show
"""
import argparse
import sys
from collections import Counter

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from cliff_walking import COLORS, METHODS, NAMES, draw_paths
from td_agents import greedy_path, train

sys.stdout.reconfigure(encoding="utf-8")

RUNS = 20


def many_runs(env, method, **settings):
    """Train RUNS times. Returns (avg training reward per game, avg of last 100 games, path lengths)."""
    rewards, last100, lengths = [], [], []
    for seed in range(RUNS):
        Q, returns, _, _ = train(env, method, episodes=500, gamma=1.0, seed=seed, **settings)
        rewards.append(returns.mean())
        last100.append(returns[-100:].mean())
        lengths.append(len(greedy_path(env, Q)[1]) - 1)
    return float(np.mean(rewards)), float(np.mean(last100)), Counter(lengths)


def describe(lengths):
    return ", ".join(f"{n}×{k}" for n, k in sorted(lengths.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    env = gym.make("CliffWalking-v1")

    print("=" * 70)
    print(f"EXPERIMENT 1 · Exploration ε (α = 0.5, ε fixed), {RUNS} runs each")
    print("=" * 70)
    print("  greedy path lengths: 13 = cliff edge, 15 = one row up, 17 = top row, 100 = loops")
    eps_values = [0.01, 0.1, 0.2]
    for eps in eps_values:
        for m in ["sarsa", "qlearning"]:
            avg, last, lengths = many_runs(env, m, alpha=0.5, eps=eps)
            print(f"  ε = {eps:<5} {NAMES[m]:<11} training reward/game: all {avg:7.1f}, "
                  f"last 100 {last:7.1f}   paths: {describe(lengths)}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2 · ε starts at 0.1 and shrinks ×0.99 per game (≈0.0007 at the end)")
    print("=" * 70)
    for m in METHODS:
        avg, last, lengths = many_runs(env, m, alpha=0.5, eps=0.1, eps_decay=0.99)
        print(f"  {NAMES[m]:<15} training reward/game: all {avg:7.1f}, last 100 {last:7.1f}   "
              f"paths: {describe(lengths)}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 3 · Learning rate α (ε = 0.1 fixed)")
    print("=" * 70)
    alphas = [0.1, 0.5, 1.0]
    alpha_results = {}
    for m in METHODS:
        for alpha in alphas:
            avg, last, lengths = many_runs(env, m, alpha=alpha, eps=0.1)
            alpha_results[(m, alpha)] = avg
            print(f"  {NAMES[m]:<15} α = {alpha:<4} training reward/game: all {avg:7.1f}, "
                  f"last 100 {last:7.1f}   paths: {describe(lengths)}")

    # Chart 1: SARSA's greedy path for three amounts of exploration (seed 0)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5))
    for ax, eps in zip(axes, eps_values):
        Q, *_ = train(env, "sarsa", episodes=500, alpha=0.5, gamma=1.0, eps=eps, seed=0)
        total, path = greedy_path(env, Q)
        draw_paths(ax, Q, path, f"SARSA trained with ε = {eps}: greedy path, {len(path) - 1} steps",
                   COLORS["sarsa"])
    fig.tight_layout()
    fig.savefig("exp_eps_paths.png", dpi=110, bbox_inches="tight")

    # Chart 2: average training reward for each alpha
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.27
    for i, m in enumerate(METHODS):
        vals = [alpha_results[(m, a)] for a in alphas]
        bars = ax.bar(np.arange(3) + (i - 1) * width, vals, width - 0.03, color=COLORS[m], label=NAMES[m])
        ax.bar_label(bars, fmt="%.1f", fontsize=9)
    ax.set_xticks(np.arange(3), [f"α = {a}" for a in alphas])
    ax.set_ylabel("Avg training reward per game (500 games)")
    ax.set_title(f"Learning rate α on CliffWalking (ε = 0.1), average of {RUNS} runs")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("exp_alpha.png", dpi=110, bbox_inches="tight")
    print("\nCharts saved to exp_eps_paths.png and exp_alpha.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
