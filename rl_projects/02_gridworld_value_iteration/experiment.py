"""
Project 2: Rebuild the playground.
Solve the gridworld with value iteration, then make the floor slippery and
see how the best plan changes.

Run:
    python experiment.py                      # gamma 0.9, slip 0.4 (60% intended, 20% each side)
    python experiment.py --slip 0.2           # less icy: is the shortcut worth the risk now?
    python experiment.py --gamma 0.5          # short-sighted agent
    python experiment.py --no-show            # save the chart without opening a window
"""
import argparse
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

from gridworld import ACTIONS, GridWorld
from value_iteration import q_value, value_iteration

sys.stdout.reconfigure(encoding="utf-8")   # so arrows print on Windows


def show_grid(env, V, policy=None):
    """Print values (and arrows if a policy is given) laid out like the board."""
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            s = (r, c)
            k = env.kind(s)
            if k == "#":
                row.append("  ####  ")
            elif k == "G":
                row.append("  GOAL  ")
            elif k == "P":
                row.append("  PIT   ")
            else:
                arrow = ACTIONS[policy[s]][3] if policy else " "
                row.append(f"{arrow}{V[s]:+6.2f} ")
        print("   " + "".join(row))


def explain_square(env, V, s, gamma):
    """Print the Bellman optimality equation for one square with real numbers."""
    print(f"   Bellman at square {s}:  V(s) = max over actions of  Σ p·[ r + γ·V(s') ]")
    best = max(ACTIONS, key=lambda a: q_value(env, V, s, a, gamma))
    for a, (name, *_ ) in ACTIONS.items():
        parts = [f"{p:.1f}·[{r:+.2f} + {gamma}·{V[s2]:+.2f}]" for p, s2, r in env.transitions(s, a)]
        mark = "  ← best" if a == best else ""
        print(f"     {name:<6} = {' + '.join(parts)} = {q_value(env, V, s, a, gamma):+.3f}{mark}")
    print()


def play(env, policy, games=10_000, max_steps=100, seed=0):
    """Actually play games with the slip dice. Returns (% goal, % pit, avg steps)."""
    rng = random.Random(seed)
    goal = pit = total_steps = 0
    for _ in range(games):
        s = env.start
        for step in range(1, max_steps + 1):
            outcomes = env.transitions(s, policy[s])
            _, s, _ = rng.choices(outcomes, weights=[p for p, _, _ in outcomes])[0]
            if env.is_terminal(s):
                break
        total_steps += step
        goal += env.kind(s) == "G"
        pit += env.kind(s) == "P"
    return 100 * goal / games, 100 * pit / games, total_steps / games


def draw(ax, env, V, policy, title):
    grid = np.full((env.rows, env.cols), np.nan)
    for s in env.states:
        grid[s] = {"G": 1.0, "P": -1.0}.get(env.kind(s), V[s])
    ax.imshow(np.ma.masked_invalid(grid), cmap="RdYlGn", vmin=-1, vmax=1)
    for r in range(env.rows):
        for c in range(env.cols):
            s = (r, c)
            k = env.kind(s)
            if k == "#":
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#555"))
            elif k in "GP":
                ax.text(c, r, "GOAL\n+1" if k == "G" else "PIT\n−1", ha="center", va="center", fontsize=10, weight="bold")
            else:
                ax.text(c, r - 0.15, ACTIONS[policy[s]][3], ha="center", va="center", fontsize=20)
                ax.text(c, r + 0.25, f"{V[s]:+.2f}", ha="center", va="center", fontsize=9)
                if k == "S":
                    ax.text(c - 0.42, r - 0.38, "start", fontsize=7, va="top")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--slip", type=float, default=0.4)
    parser.add_argument("--step-reward", type=float, default=-0.04)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    g = args.gamma

    # ── PART 1: normal floor ─────────────────────────────────────────────
    print("=" * 64)
    print(f"PART 1 · Normal floor (you always go where you aim), γ = {g}")
    print("=" * 64)
    solid = GridWorld(slip=0.0, step_reward=args.step_reward)

    def print_early_sweeps(sweep, V, change):
        if sweep <= 4:
            print(f"\n After sweep {sweep}  (biggest change {change:.3f}):")
            show_grid(solid, V)

    V_solid, pi_solid, sweeps = value_iteration(solid, gamma=g, on_sweep=print_early_sweeps)
    print(f"\n Converged after {sweeps} sweeps. Final values and best moves:")
    show_grid(solid, V_solid, pi_solid)
    print()
    explain_square(solid, V_solid, (1, 3), g)

    # ── PART 2: slippery floor ───────────────────────────────────────────
    print("=" * 64)
    print(f"PART 2 · Slippery floor: {100 * (1 - args.slip):.0f}% where you aim, "
          f"{100 * args.slip / 2:.0f}% slide to each side")
    print("=" * 64)
    icy = GridWorld(slip=args.slip, step_reward=args.step_reward)
    V_icy, pi_icy, sweeps = value_iteration(icy, gamma=g)
    print(f"\n Converged after {sweeps} sweeps. Final values and best moves:")
    show_grid(icy, V_icy, pi_icy)
    changed = [s for s in pi_icy if pi_icy[s] != pi_solid[s]]
    print(f"\n Squares where the plan changed: "
          + (", ".join(f"{s} was {ACTIONS[pi_solid[s]][0]}, now {ACTIONS[pi_icy[s]][0]}" for s in changed) or "none"))
    print()
    explain_square(icy, V_icy, (1, 3), g)

    # ── PART 3: test both plans on the slippery floor ────────────────────
    print("=" * 64)
    print("PART 3 · Play 10,000 games on the SLIPPERY floor with each plan")
    print("=" * 64)
    for name, pi in [("plan made for normal floor", pi_solid), ("plan made for slippery floor", pi_icy)]:
        goal, pit, steps = play(icy, pi)
        print(f"   {name:<30} goal {goal:5.1f}%   pit {pit:5.1f}%   avg steps {steps:5.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    draw(ax1, solid, V_solid, pi_solid, f"Normal floor (γ = {g})")
    draw(ax2, icy, V_icy, pi_icy, f"Slippery floor, slip = {args.slip} (γ = {g})")
    fig.tight_layout()
    fig.savefig("results.png", dpi=120, bbox_inches="tight")
    print("\nChart saved to results.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
