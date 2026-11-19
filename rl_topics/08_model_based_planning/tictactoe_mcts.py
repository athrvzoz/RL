"""
MCTS plays tic-tac-toe (and a little Connect-4).

Part 1: how big is the full game tree? (why we can't just search everything)
Part 2: a tiny search you can check by hand: 12 simulations, printed one by one
Part 3: MCTS vs a random player, at different simulation budgets
Part 4: MCTS vs MCTS
Part 5: which first move does MCTS like? (visit counts on the empty board)
Part 6: the same mcts.py on Connect-4, vs a random player

Run:  python tictactoe_mcts.py            (about 2-3 minutes)
      python tictactoe_mcts.py --no-show
"""
import argparse
import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from games import Connect4, TicTacToe, count_full_tree, minimax_value
from mcts import match, mcts_player, random_player, search, ucb_score

sys.stdout.reconfigure(encoding="utf-8")
T = TicTacToe


def part1_tree_size():
    print("=" * 66)
    print("PART 1 · How big is the game tree?")
    print("=" * 66)
    t0 = time.time()
    nodes, games = count_full_tree()
    print(f"  Tic-tac-toe, full tree: {nodes:,} positions in the tree, {games:,} different games"
          f"  ({time.time() - t0:.1f} s to walk it all)")
    print(f"  Perfect play from the empty board (minimax): value {minimax_value(T.start)}  (0 = draw)")
    print(f"  Different boards (positions met more than once counted once): {minimax_value.cache_info().currsize:,}")
    print("\n  Rough size of the tree = (moves per turn) ^ (turns per game):")
    for name, b, d in (("tic-tac-toe", 4, 9), ("chess", 35, 80), ("Go (19x19)", 250, 150)):
        print(f"     {name:<12} about {b:>3} ^ {d:<3} = 10^{d * math.log10(b):.0f}")


def part2_tiny_search(n_sims=12):
    print("\n" + "=" * 66)
    print(f"PART 2 · A tiny search you can check by hand ({n_sims} simulations, c = 1.4)")
    print("=" * 66)
    s = "XOXOOX..."
    print(T.show(s) + "     X to move. Empty squares: 6, 7, 8")
    log = []
    root = search(T, s, n_sims, c=1.4, seed=1, log=log)
    for k, (path, result) in enumerate(log, 1):
        print(f"  sim {k:>2}: path {'-'.join(map(str, path)):<7} rollout ends: {result}")
    print("\n  Root children after the search (W = wins for X, a draw counts 0.5):")
    for ch in sorted(root.children, key=lambda ch: ch.move):
        print(f"     move {ch.move}:  N = {ch.N:>2}   W = {ch.W:4.1f}   W/N = {ch.W / ch.N:.3f}   "
              f"UCB for the next pick = {ucb_score(ch, root.N, 1.4):.3f}")
        for g in sorted(ch.children, key=lambda g: g.move):
            print(f"         then O plays {g.move}:  N = {g.N}   W(for O) = {g.W:.1f}")
    print(f"  Root: N = {root.N}.  MCTS plays the most visited move: "
          f"{max(root.children, key=lambda ch: ch.N).move}")


def part3_vs_random(budgets, games=100):
    print("\n" + "=" * 66)
    print(f"PART 3 · MCTS vs a random player ({games} games each)")
    print("=" * 66)
    rx, rd, ro = match(T, random_player, random_player, 1000, seed=0)
    print(f"  random vs random (1000 games): X wins {rx / 10:.1f}%, draws {rd / 10:.1f}%, O wins {ro / 10:.1f}%")
    rows = []
    for n in budgets:
        t0 = time.time()
        x_win, x_draw, x_loss = match(T, mcts_player(n), random_player, games, seed=n)
        o_loss, o_draw, o_win = match(T, random_player, mcts_player(n), games, seed=n + 1)
        rows.append((n, x_win, x_draw, x_loss, o_win, o_draw, o_loss))
        print(f"  {n:>5} sims:  as X  win {x_win:>3}  draw {x_draw:>3}  lose {x_loss:>3}   |   "
              f"as O  win {o_win:>3}  draw {o_draw:>3}  lose {o_loss:>3}   ({time.time() - t0:.1f} s)")
    return rows


def part4_self_play(games=50):
    print("\n" + "=" * 66)
    print(f"PART 4 · MCTS vs MCTS ({games} games each)")
    print("=" * 66)
    for nx, no in ((1000, 1000), (50, 1000), (1000, 50)):
        t0 = time.time()
        x, d, o = match(T, mcts_player(nx), mcts_player(no), games, seed=7)
        print(f"  X = MCTS {nx:>4} sims vs O = MCTS {no:>4} sims:  X wins {x:>2}, draws {d:>2}, O wins {o:>2}"
              f"   ({time.time() - t0:.1f} s)")


def part5_first_move(n_sims=2000):
    print("\n" + "=" * 66)
    print(f"PART 5 · First move on the empty board ({n_sims} simulations)")
    print("=" * 66)
    root = search(T, T.start, n_sims, c=1.4, seed=0)
    visits, rates = np.zeros(9), np.zeros(9)
    for ch in root.children:
        visits[ch.move], rates[ch.move] = ch.N, ch.W / ch.N
    print("  Visit counts:                 Average result for X (win 1, draw 0.5):")
    for r in range(3):
        print("     " + " ".join(f"{int(v):>5}" for v in visits[r * 3:r * 3 + 3]) + "          "
              + " ".join(f"{q:.3f}" for q in rates[r * 3:r * 3 + 3]))
    print(f"  MCTS plays square {int(np.argmax(visits))} ({100 * visits.max() / n_sims:.0f}% of all visits)")
    return visits, rates


def part6_connect4(budgets, games=20):
    print("\n" + "=" * 66)
    print(f"PART 6 · Connect-4 (7 columns, 6 rows): MCTS as X vs random ({games} games each)")
    print("=" * 66)
    for n in budgets:
        t0 = time.time()
        x, d, o = match(Connect4, mcts_player(n), random_player, games, seed=n)
        print(f"  {n:>4} sims:  win {x:>2}  draw {d:>2}  lose {o:>2}   ({time.time() - t0:.1f} s)")
    t0 = time.time()
    x, d, o = match(Connect4, mcts_player(400), mcts_player(25), games, seed=5)
    print(f"  MCTS 400 (X) vs MCTS 25 (O): X wins {x}, draws {d}, O wins {o}   ({time.time() - t0:.1f} s)")
    x, d, o = match(Connect4, mcts_player(25), mcts_player(400), games, seed=6)
    print(f"  MCTS 25 (X) vs MCTS 400 (O): X wins {x}, draws {d}, O wins {o}   ({time.time() - t0:.1f} s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")

    part1_tree_size()
    part2_tiny_search()
    budgets = [1, 10, 50, 200, 1000]
    rows = part3_vs_random(budgets)
    part4_self_play()
    visits, rates = part5_first_move()
    part6_connect4([25, 100, 400])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    labels = [str(b) for b in budgets]
    for ax, title, cols in ((axes[0], "MCTS plays X (moves first)", (1, 2, 3)),
                            (axes[1], "MCTS plays O (moves second)", (4, 5, 6))):
        win = np.array([r[cols[0]] for r in rows])
        draw = np.array([r[cols[1]] for r in rows])
        loss = np.array([r[cols[2]] for r in rows])
        ax.bar(labels, win, color="#1e8a5a", label="MCTS wins")
        ax.bar(labels, draw, bottom=win, color="#b9c2d6", label="draw")
        ax.bar(labels, loss, bottom=win + draw, color="#c2413a", label="MCTS loses")
        ax.set_xlabel("Simulations per move")
        ax.set_ylabel("Games out of 100")
        ax.set_title(f"vs random player: {title}", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
    ax = axes[2]
    im = ax.imshow(visits.reshape(3, 3), cmap="Blues")
    for i in range(9):
        ax.text(i % 3, i // 3, f"{int(visits[i])}\n({rates[i]:.2f})", ha="center", va="center",
                color="white" if visits[i] > visits.max() * 0.6 else "black", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("First move: visits (avg result for X)\n2000 simulations", fontsize=10)
    fig.tight_layout()
    fig.savefig("tictactoe_mcts_results.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to tictactoe_mcts_results.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
