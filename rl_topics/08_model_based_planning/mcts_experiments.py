"""
MCTS experiments on tic-tac-toe, marked against the perfect answer key (minimax).

Test set: 150 random tic-tac-toe positions where the choice MATTERS
(at least one legal move is worse than the best). For each position we ask
MCTS for a move and check: is it one of the perfect moves?

Experiment 1: simulation budget 10, 30, 100, 300, 1000  (c = 1.4)
Experiment 2: exploration constant c = 0, 0.25, 0.7, 1.4, 3, 10  (100 simulations)
Experiment 3: the AlphaZero idea. Replace "try everything + random rollouts" with a
              "network" that gives priors P and a value. We have no trained network,
              so we use a PRETEND one built from the answer key, with mistakes added:
                 prior:  P(move) proportional to e^(2 x (true score + noise))   (an unreliable hint)
                 value:  true value + random noise (std 0.5)                       (a noisy guess)
              A real AlphaZero learns these two outputs by self-play.

Run:  python mcts_experiments.py            (about 1-2 minutes)
      python mcts_experiments.py --no-show
"""
import argparse
import math
import random
import sys
import time

import matplotlib.pyplot as plt

from games import TicTacToe, minimax_value, perfect_moves
from mcts import best_move, puct_search, search

sys.stdout.reconfigure(encoding="utf-8")
T = TicTacToe


def all_positions():
    seen, todo = set(), [T.start]
    while todo:
        s = todo.pop()
        if s in seen or T.winner(s) is not None:
            continue
        seen.add(s)
        todo.extend(T.play(s, m) for m in T.legal_moves(s))
    return sorted(seen)


def test_set(size=150, seed=0):
    tricky = [s for s in all_positions() if len(perfect_moves(s)) < len(T.legal_moves(s))]
    return tricky, random.Random(seed).sample(tricky, size)


def accuracy(positions, chooser):
    good = sum(chooser(s, k) in perfect_moves(s) for k, s in enumerate(positions))
    return 100 * good / len(positions)


def pretend_prior(s):
    # True score of each move (+1 win, 0 draw, -1 loss) PLUS a fixed random mistake per
    # position, so the "network" is often wrong about which move is best.
    rng = random.Random(int(s.replace(".", "0").replace("X", "1").replace("O", "2"), 3))
    scores = {m: -minimax_value(T.play(s, m)) + rng.gauss(0, 1.0) for m in T.legal_moves(s)}
    weights = {m: math.exp(2 * v) for m, v in scores.items()}
    total = sum(weights.values())
    return {m: w / total for m, w in weights.items()}


def uniform_prior(s):
    moves = T.legal_moves(s)
    return {m: 1 / len(moves) for m in moves}


def pretend_value(s, rng):
    return max(-1.0, min(1.0, minimax_value(s) + rng.gauss(0, 0.5)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")

    tricky, positions = test_set()
    print(f"Positions where the game is not over: {len(all_positions()):,}; "
          f"where the choice matters: {len(tricky):,}; test set: {len(positions)}")
    blind = sum(len(perfect_moves(s)) / len(T.legal_moves(s)) for s in positions) / len(positions)
    print(f"A random move would be perfect {100 * blind:.1f}% of the time")

    print("\n" + "=" * 66)
    print("EXPERIMENT 1 · Simulation budget (c = 1.4): % perfect moves")
    print("=" * 66)
    budgets = [10, 30, 100, 300, 1000]
    acc_budget = []
    for n in budgets:
        t0 = time.time()
        acc_budget.append(accuracy(positions, lambda s, k: best_move(search(T, s, n, 1.4, seed=k))))
        print(f"  {n:>5} simulations: {acc_budget[-1]:5.1f}% perfect   ({time.time() - t0:.1f} s)")

    print("\n" + "=" * 66)
    print("EXPERIMENT 2 · Exploration constant c (100 simulations): % perfect moves")
    print("=" * 66)
    cs = [0, 0.25, 0.7, 1.4, 3, 10]
    acc_c = []
    for c in cs:
        acc_c.append(accuracy(positions, lambda s, k: best_move(search(T, s, 100, c, seed=k))))
        print(f"  c = {c:<5}: {acc_c[-1]:5.1f}% perfect")

    print("\n" + "=" * 66)
    print("EXPERIMENT 3 · The AlphaZero idea: priors and a value instead of blind search")
    print("=" * 66)
    small = [5, 10, 20, 50, 100]
    variants = {
        "plain MCTS (UCT, random rollouts)":
            lambda n: lambda s, k: best_move(search(T, s, n, 1.4, seed=k)),
        "PUCT, equal priors + random rollouts":
            lambda n: lambda s, k: best_move(puct_search(T, s, n, uniform_prior, None, seed=k)),
        "PUCT, pretend-network priors + rollouts":
            lambda n: lambda s, k: best_move(puct_search(T, s, n, pretend_prior, None, seed=k)),
        "PUCT, pretend priors + pretend value":
            lambda n: lambda s, k: best_move(puct_search(T, s, n, pretend_prior, pretend_value, seed=k)),
    }
    acc_az = {}
    print("  simulations:                              " + "".join(f"{n:>7}" for n in small))
    for name, make in variants.items():
        acc_az[name] = [accuracy(positions, make(n)) for n in small]
        print(f"  {name:<42}" + "".join(f"{a:6.1f}%" for a in acc_az[name]))
    print(f"  (the pretend prior alone, no search: plays its top move "
          f"{accuracy(positions, lambda s, k: max(pretend_prior(s).items(), key=lambda kv: kv[1])[0]):.1f}% perfect)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(budgets, acc_budget, marker="o", color="#2c4bd1", lw=2)
    ax.axhline(100 * blind, color="gray", ls="--", lw=1, label="random move")
    ax.set_xscale("log")
    ax.set_xlabel("Simulations per move (log scale)")
    ax.set_ylabel("% of test positions: perfect move")
    ax.set_title("More thinking time = better moves (c = 1.4)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar([str(c) for c in cs], acc_c, color=["#c2413a" if c == 0 else "#2c4bd1" for c in cs])
    ax.set_xlabel("Exploration constant c")
    ax.set_ylabel("% perfect moves (100 simulations)")
    ax.set_title("c = 0 never explores; very big c explores blindly", fontsize=10)
    ax.set_ylim(0, 100)
    for i, a in enumerate(acc_c):
        ax.text(i, a + 1.5, f"{a:.0f}", ha="center", fontsize=9)
    ax = axes[2]
    for (name, accs), color in zip(acc_az.items(), ("#2c4bd1", "#9aa3b9", "#a86a08", "#1e8a5a")):
        ax.plot(small, accs, marker="o", lw=2, color=color, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Simulations per move (log scale)")
    ax.set_ylabel("% perfect moves")
    ax.set_title("AlphaZero idea: priors help most when simulations are few", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("mcts_experiments.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to mcts_experiments.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
