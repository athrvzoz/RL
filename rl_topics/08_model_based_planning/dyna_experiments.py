"""
Dyna-Q experiments: turn the knobs and see what happens.

Experiment 1: planning steps n = 0, 1, 2, 5, 10, 20, 50, 100 on the maze.
              Fewer REAL steps, but more COMPUTER work. Where is the sweet spot?
Experiment 2: "planning in a wrong map". The world changes, the notebook doesn't.
   (a) Blocking maze: after 1000 steps the short way gets blocked.
   (b) Shortcut maze: after 3000 steps a new short way opens.
   Dyna-Q   trusts its notebook.
   Dyna-Q+  adds a small curiosity bonus  kappa * sqrt(steps since I last tried it)
            to imagined rewards, so it goes back to check old, untested moves.

Run:  python dyna_experiments.py            (about 2-3 minutes)
      python dyna_experiments.py --no-show
"""
import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from dyna_q import DynaQ, Maze, run_episodes

sys.stdout.reconfigure(encoding="utf-8")

#   S = start (row 5, col 3), G = goal. Row 3 is the wall that moves.
BLOCKING_BEFORE = ["........G", ".........", ".........", "########.", ".........", "...S....."]
BLOCKING_AFTER = ["........G", ".........", ".........", ".########", ".........", "...S....."]
SHORTCUT_BEFORE = BLOCKING_AFTER
SHORTCUT_AFTER = ["........G", ".........", ".........", ".#######.", ".........", "...S....."]


def experiment1(ns=(0, 1, 2, 5, 10, 20, 50, 100), runs=20, episodes=50):
    print("=" * 74)
    print(f"EXPERIMENT 1 · How many planning steps? ({runs} runs x {episodes} games each)")
    print("=" * 74)
    maze = Maze()
    print("      n   real steps (50 games)   first game < 20 steps   Q-updates   time per run")
    rows = []
    for n in ns:
        t0 = time.time()
        steps, updates = [], []
        for run in range(runs):
            agent = DynaQ(maze.n_states, n_planning=n, seed=100 + run)
            steps.append(run_episodes(maze, agent, episodes))
            updates.append(agent.updates)
        steps = np.array(steps)
        curve = steps.mean(axis=0)
        first = next((i + 1 for i, v in enumerate(curve) if v < 20), None)
        secs = (time.time() - t0) / runs
        rows.append((n, steps.sum(axis=1).mean(), first, np.mean(updates), secs))
        print(f"   {n:>4}   {rows[-1][1]:>12,.0f}            game {first!s:<6}        "
              f"{rows[-1][3]:>10,.0f}   {secs:6.2f} s")
    return rows


def changing_world(before, after, change_at, total_steps, kappa, n=50, alpha=0.5, seed=0):
    """Run one long life (no episodes counted): reach G -> back to S. Returns cumulative reward."""
    maze = Maze(before)
    agent = DynaQ(maze.n_states, n_planning=n, alpha=alpha, kappa=kappa, seed=seed)
    cumulative = np.zeros(total_steps)
    total, s = 0.0, maze.start
    for t in range(total_steps):
        if t == change_at:
            maze.set_layout(after)                    # the WORLD changes; the notebook does not know
        a = agent.act(s)
        s2, r, done = maze.step(s, a)
        agent.learn(s, a, r, s2, done)
        total += r
        cumulative[t] = total
        s = maze.start if done else s2
    return cumulative


def experiment2(runs=5):
    print("\n" + "=" * 74)
    print(f"EXPERIMENT 2 · The world changes (n = 50, alpha 0.5, {runs} runs, kappa = 0.001 for Dyna-Q+)")
    print("=" * 74)
    results = {}
    for name, before, after, change_at, total in (("blocking", BLOCKING_BEFORE, BLOCKING_AFTER, 1000, 3000),
                                                 ("shortcut", SHORTCUT_BEFORE, SHORTCUT_AFTER, 3000, 6000)):
        print(f"\n  {name} maze: the wall changes after step {change_at}")
        print("     before:                after:")
        for b, a in zip(before, after):
            print(f"     {' '.join(b)}      {' '.join(a)}")
        print(f"     shortest path before: {Maze(before).shortest_path()} steps, "
              f"after: {Maze(after).shortest_path()} steps")
        for label, kappa in (("Dyna-Q ", 0.0), ("Dyna-Q+", 0.001)):
            t0 = time.time()
            curves = np.array([changing_world(before, after, change_at, total, kappa, seed=k)
                               for k in range(runs)])
            mean = curves.mean(axis=0)
            results[(name, label)] = mean
            gained_before = mean[change_at - 1]
            gained_after = mean[-1] - mean[change_at - 1]
            last_1000 = mean[-1] - mean[-1001]
            print(f"     {label}: goals reached before the change {gained_before:6.1f}, after {gained_after:6.1f}"
                  f"   (in the last 1000 steps: {last_1000:5.1f})   {(time.time() - t0) / runs:.1f} s/run")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")

    rows = experiment1()
    results = experiment2()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ns = [r[0] for r in rows]
    x = np.arange(len(ns))
    ax = axes[0]
    ax.bar(x, [r[1] for r in rows], color="#2c4bd1")
    ax.set_xticks(x, [str(n) for n in ns])
    ax.set_xlabel("Planning steps n per real step")
    ax.set_ylabel("Real steps needed for 50 games", color="#2c4bd1")
    ax2 = ax.twinx()
    ax2.plot(x, [r[3] for r in rows], color="#a86a08", marker="o", lw=2)
    ax2.set_yscale("log")
    ax2.set_ylabel("Total Q-updates (log scale)", color="#a86a08")
    ax.set_title("More planning: fewer real steps, more computing", fontsize=10)
    for ax, name, change in ((axes[1], "blocking", 1000), (axes[2], "shortcut", 3000)):
        for label, color in (("Dyna-Q ", "#c2413a"), ("Dyna-Q+", "#1e8a5a")):
            ax.plot(results[(name, label)], color=color, lw=2, label=label.strip())
        ax.axvline(change, color="gray", ls="--", lw=1)
        ax.text(change, ax.get_ylim()[1] * 0.92, "  wall moves", color="gray", fontsize=9)
        ax.set_xlabel("Real time steps")
        ax.set_ylabel("Goals reached so far (cumulative reward)")
        ax.set_title(f"{name.capitalize()} maze (average of 5 runs)", fontsize=10)
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("dyna_experiments.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to dyna_experiments.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
