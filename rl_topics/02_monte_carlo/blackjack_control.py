"""
Topic 2b: Monte Carlo CONTROL on Blackjack (learn a better way to play).

Prediction (blackjack_prediction.py) only GRADED a fixed policy.
Control also IMPROVES it, with the same loop you know from Q-learning:

    play a whole game with ε-greedy(Q)            <- explore 10% of the time
    game over -> walk backwards -> Q[s,a] = average of the real results after (s,a)
    the next game automatically uses the better Q

Then we compare four ways of playing, 100,000 test games each, no learning:
    random         : coin flip every decision
    stick on 20+   : the simple policy we evaluated before
    stick on 17+   : copy the dealer's rule
    learned by MC  : greedy on the learned Q

Run:  python blackjack_control.py                   (about 3 minutes)
      python blackjack_control.py --episodes 200000 --no-show
"""
import argparse
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from mc import (greedy, mc_control, moving_average, random_policy, stick_on_17, stick_on_20,
                win_draw_loss)

sys.stdout.reconfigure(encoding="utf-8")

DEALER = ["A"] + [str(c) for c in range(2, 11)]


def policy_grid(Q, ace):
    # 1 = hit, 0 = stick, for sums 12..21 (rows) x dealer A..10 (columns)
    return np.array([[int(np.argmax(Q[(t, d, ace)])) for d in range(1, 11)] for t in range(12, 22)])


def print_chart(Q, ace):
    grid = policy_grid(Q, ace)
    print(f"\n  {'USABLE ACE' if ace else 'NO USABLE ACE'}   (H = hit, S = stick)")
    print("    sum | dealer: " + " ".join(f"{d:>2}" for d in DEALER))
    for i in range(9, -1, -1):
        print(f"     {12 + i:>2} |         " + " ".join(f"{'H' if g else 'S':>2}" for g in grid[i]))


def draw_policy(ax, Q, ace, title):
    grid = policy_grid(Q, ace)
    ax.imshow(grid, origin="lower", cmap=plt.matplotlib.colors.ListedColormap(["#2a78d6", "#eda100"]),
              vmin=0, vmax=1, aspect="auto")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, "HIT" if grid[i, j] else "STICK", ha="center", va="center", fontsize=6.5,
                    color="black" if grid[i, j] else "white")
    ax.set_xticks(range(10), DEALER)
    ax.set_yticks(range(10), range(12, 22))
    ax.set_xlabel("Dealer's showing card")
    ax.set_ylabel("Your sum")
    ax.set_title(title, fontsize=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500_000)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    env = gym.make("Blackjack-v1")

    print(f"Training: MC control, {args.episodes:,} games, ε = {args.eps}, γ = 1\n")
    Q, N, rewards = mc_control(env, args.episodes, eps=args.eps, seed=0)
    step = args.episodes // 5
    for ep in range(step, args.episodes + 1, step):
        print(f"  after {ep:>9,} games: average result of the last 10,000 training games "
              f"= {rewards[ep - 10_000:ep].mean():+.3f}")
    print(f"  (state, action) pairs tried: {len(N) * 2 - sum((n == 0).sum() for n in N.values())}")

    print("\nSome learned Q-values  (Q = average final result after doing that action):")
    for s in [(16, 10, 0), (12, 2, 0), (12, 4, 0), (13, 2, 0), (18, 9, 1), (18, 9, 0), (20, 10, 0)]:
        print(f"  sum {s[0]:>2}, dealer {DEALER[s[1] - 1]:>2}, {'usable ace' if s[2] else 'no ace    '}:  "
              f"Q(stick) = {Q[s][0]:+.3f} ({int(N[s][0]):>6,} tries)   "
              f"Q(hit) = {Q[s][1]:+.3f} ({int(N[s][1]):>6,} tries)   -> {'HIT' if np.argmax(Q[s]) else 'STICK'}")

    print("\nThe learned policy:")
    print_chart(Q, 1)
    print_chart(Q, 0)

    print("\nTest: 100,000 games each, no exploration, no learning")
    print(f"  {'policy':<16}{'win %':>8}{'draw %':>8}{'loss %':>8}{'avg result':>12}")
    results = {}
    for name, pol in [("random", random_policy), ("stick on 20+", stick_on_20),
                      ("stick on 17+", stick_on_17), ("learned by MC", greedy(Q))]:
        results[name] = win_draw_loss(env, pol)
        w, d, l, avg = results[name]
        print(f"  {name:<16}{w:8.1f}{d:8.1f}{l:8.1f}{avg:+12.3f}")

    # ---- charts ----------------------------------------------------------------
    fig = plt.figure(figsize=(12, 9))
    draw_policy(fig.add_subplot(2, 2, 1), Q, 1, "Learned policy · usable ace")
    draw_policy(fig.add_subplot(2, 2, 2), Q, 0, "Learned policy · no usable ace")

    ax = fig.add_subplot(2, 2, 3)
    names = list(results)
    wins = np.array([results[n][0] for n in names])
    draws = np.array([results[n][1] for n in names])
    losses = np.array([results[n][2] for n in names])
    ax.barh(names, wins, color="#1baf7a", label="win", edgecolor="white", lw=2)
    ax.barh(names, draws, left=wins, color="#b8b6ae", label="draw", edgecolor="white", lw=2)
    ax.barh(names, losses, left=wins + draws, color="#e34948", label="loss", edgecolor="white", lw=2)
    for i in range(len(names)):
        ax.text(wins[i] / 2, i, f"{wins[i]:.1f}%", ha="center", va="center", fontsize=8)
        ax.text(wins[i] + draws[i] + losses[i] / 2, i, f"{losses[i]:.1f}%", ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("% of 100,000 test games")
    ax.set_title("Win / draw / loss", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)

    ax = fig.add_subplot(2, 2, 4)
    window = 10_000
    ax.plot(np.arange(window - 1, len(rewards)), moving_average(rewards, window), color="#2a78d6", lw=2,
            label=f"training (ε = {args.eps}), last {window:,} games")
    ax.axhline(results["learned by MC"][3], color="#1baf7a", ls="--", lw=1.5, label="learned policy, test")
    ax.axhline(results["stick on 20+"][3], color="#52514e", ls=":", lw=1.5, label="stick on 20+, test")
    ax.set_xlabel("Training game")
    ax.set_ylabel("Average result per game")
    ax.set_title("Learning curve", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("control_results.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to control_results.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
