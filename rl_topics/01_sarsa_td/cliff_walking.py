"""
CliffWalking: SARSA vs Expected SARSA vs Q-learning (the classic experiment).

    row 0   .  .  .  .  .  .  .  .  .  .  .  .
    row 1   .  .  .  .  .  .  .  .  .  .  .  .
    row 2   .  .  .  .  .  .  .  .  .  .  .  .
    row 3   S  C  C  C  C  C  C  C  C  C  C  G       C = cliff

Every step costs -1. Stepping into the cliff costs -100 and sends you back to S
(the game does NOT end). Reaching G ends the game.
Actions: 0 up, 1 right, 2 down, 3 left.  State number = row * 12 + column.

The shortest path walks right along the cliff edge (13 steps, reward -13).
But the agent keeps exploring (epsilon = 0.1): 1 move in 10 is random,
and a random "down" on the edge means -100.

Q-learning learns the value of the BEST possible play, so it picks the edge.
SARSA learns the value of what it REALLY does (exploring included), so it
discovers that the edge is dangerous for an explorer and picks a safer path.

Part 1: one training run per method (seed 0): paths, test scores.
Part 2: 50 runs per method, averaged learning curves (the fair comparison).
Part 3: one real transition, three different targets.

Run:  python cliff_walking.py            (about 1 minute)
      python cliff_walking.py --no-show
"""
import argparse
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from td_agents import evaluate_with_eps, greedy_path, moving_average, policy_probs, train

sys.stdout.reconfigure(encoding="utf-8")

ROWS, COLS = 4, 12
ARROWS = "↑→↓←"                                      # CliffWalking: 0 up, 1 right, 2 down, 3 left
METHODS = ["sarsa", "expected_sarsa", "qlearning"]
NAMES = {"sarsa": "SARSA", "expected_sarsa": "Expected SARSA", "qlearning": "Q-learning"}
COLORS = {"sarsa": "#2C4BD1", "expected_sarsa": "#1E8A5A", "qlearning": "#C47A00"}
SETTINGS = dict(episodes=500, alpha=0.5, gamma=1.0, eps=0.1)


def is_cliff(s):
    return s // COLS == 3 and 1 <= s % COLS <= 10


def show_path(path):
    """Print the grid with the greedy path marked by *."""
    on_path = set(path)
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            s = r * COLS + c
            if s == 36:
                row.append("S")
            elif s == 47:
                row.append("G")
            elif is_cliff(s):
                row.append("C")
            else:
                row.append("*" if s in on_path else ".")
        print("     " + "  ".join(row))


def draw_paths(ax, Q, path, title, color):
    for r in range(ROWS):
        for c in range(COLS):
            s = r * COLS + c
            y = ROWS - 1 - r
            if is_cliff(s):
                ax.add_patch(plt.Rectangle((c, y), 1, 1, color="#5B3A29", ec="white"))
                ax.text(c + 0.5, y + 0.5, "cliff", ha="center", va="center", fontsize=7, color="white")
                continue
            face = "#E3F3EA" if s == 47 else "#F1F3F8"
            ax.add_patch(plt.Rectangle((c, y), 1, 1, color=face, ec="white"))
            if s == 47:
                ax.text(c + 0.5, y + 0.5, "G", ha="center", va="center", fontsize=12, weight="bold")
            else:
                ax.text(c + 0.5, y + 0.5, ARROWS[int(np.argmax(Q[s]))], ha="center", va="center",
                        fontsize=11, color="#9AA3B9")
    xs = [p % COLS + 0.5 for p in path]
    ys = [ROWS - 1 - p // COLS + 0.5 for p in path]
    ax.plot(xs, ys, color=color, lw=3, marker="o", ms=4)
    ax.text(0.5, 0.5, "S", ha="center", va="center", fontsize=12, weight="bold")
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def part1(env):
    print("=" * 66)
    print("PART 1 · One training run per method: 500 games, α=0.5, ε=0.1, γ=1")
    print("=" * 66)
    results = {}
    for m in METHODS:
        Q, returns, lengths, falls = train(env, m, seed=0, **SETTINGS)
        total, path = greedy_path(env, Q)
        avg_eps, falls_eps = evaluate_with_eps(env, Q, eps=0.1)
        results[m] = (Q, path)
        print(f"\n  {NAMES[m]}")
        print(f"    training: avg reward of last 100 games {returns[-100:].mean():7.1f}, "
              f"cliff falls in 500 games: {falls.sum()}")
        print(f"    greedy test (no exploration): reward {total}, {len(path) - 1} steps")
        print(f"    test still exploring (ε=0.1, 1000 games): avg reward {avg_eps:.1f}, "
              f"cliff falls per game {falls_eps:.3f}")
        show_path(path)
    return results


def part2(env, runs):
    print("\n" + "=" * 66)
    print(f"PART 2 · {runs} runs per method (different seeds), averaged")
    print("=" * 66)
    curves, fall_curves = {}, {}
    for m in METHODS:
        all_returns, all_falls, steps = [], [], []
        for seed in range(runs):
            Q, returns, lengths, falls = train(env, m, seed=seed, **SETTINGS)
            all_returns.append(returns)
            all_falls.append(falls)
            steps.append(len(greedy_path(env, Q)[1]) - 1)
        curves[m] = np.mean(all_returns, axis=0)
        fall_curves[m] = np.mean(all_falls, axis=0)
        counts = {n: steps.count(n) for n in sorted(set(steps))}
        print(f"\n  {NAMES[m]}")
        print(f"    avg training reward per game, games 1-500: {curves[m].mean():7.1f}"
              f"   games 401-500: {curves[m][-100:].mean():7.1f}")
        print(f"    avg cliff falls per game, games 401-500: {fall_curves[m][-100:].mean():.3f}")
        print("    greedy path length over the runs: " +
              ", ".join(f"{n} steps × {k}" for n, k in counts.items()))
        if 100 in counts:
            print("    (100 steps = the greedy arrows walk in a circle and never reach G)")
    return curves, fall_curves


def part3(Q, owner):
    print("\n" + "=" * 66)
    print(f"PART 3 · One transition, three targets (using {owner}'s learned table)")
    print("=" * 66)
    s, a, r, s2 = 24, 1, -1, 25                       # row 2 col 0 --right--> row 2 col 1 (cliff edge)
    eps, alpha = 0.1, 0.5
    probs = policy_probs(Q, s2, eps)
    print(f"  s = {s} (row 2, col 0), action = right, reward = {r}, s2 = {s2} (row 2, col 1, cliff edge)")
    print(f"  old Q[s, right] = {Q[s, a]:.2f}")
    print("  Q[s2] = " + "  ".join(f"{ARROWS[i]} {Q[s2, i]:.2f}" for i in range(4)))
    print("  my ε-greedy probabilities in s2 = " + "  ".join(f"{ARROWS[i]} {probs[i]:.3f}" for i in range(4)))
    greedy = int(np.argmax(Q[s2]))
    explore = 2 if greedy != 2 else 0                  # an exploring move: "down"
    rows = [("Q-learning (max)", np.max(Q[s2])),
            (f"SARSA, a2 = {ARROWS[greedy]} (greedy, {100 * probs[greedy]:.1f}%)", Q[s2, greedy]),
            (f"SARSA, a2 = {ARROWS[explore]} (explore, {100 * probs[explore]:.1f}%)", Q[s2, explore]),
            ("Expected SARSA (average)", float(np.dot(probs, Q[s2])))]
    for name, future in rows:
        target = r + future
        new = Q[s, a] + alpha * (target - Q[s, a])
        print(f"  {name:<34} future {future:7.2f}  target {target:7.2f}  "
              f"δ {target - Q[s, a]:+6.2f}  new Q {new:7.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")                     # just save the PNGs, no window

    env = gym.make("CliffWalking-v1")
    results = part1(env)
    curves, fall_curves = part2(env, args.runs)
    part3(results["sarsa"][0], "SARSA")
    part3(results["qlearning"][0], "Q-learning")

    # Chart 1: learning curves (reward per training game) and cliff falls
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for m in METHODS:
        x = np.arange(9, SETTINGS["episodes"])
        axes[0].plot(x, moving_average(curves[m]), color=COLORS[m], lw=2, label=NAMES[m])
        axes[1].plot(x, moving_average(fall_curves[m]), color=COLORS[m], lw=2, label=NAMES[m])
    axes[0].set_ylim(-100, -10)
    axes[0].set_ylabel("Reward in the training game")
    axes[0].set_title(f"CliffWalking: training reward (ε = 0.1 all the time), average of {args.runs} runs")
    axes[1].set_ylabel("Cliff falls per training game")
    axes[1].set_xlabel("Training game")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig("cliff_curves.png", dpi=110, bbox_inches="tight")

    # Chart 2: the learned greedy paths
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5))
    for ax, m in zip(axes, METHODS):
        Q, path = results[m]
        draw_paths(ax, Q, path, f"{NAMES[m]}: greedy path, {len(path) - 1} steps", COLORS[m])
    fig.tight_layout()
    fig.savefig("cliff_paths.png", dpi=110, bbox_inches="tight")
    print("\nCharts saved to cliff_curves.png and cliff_paths.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
