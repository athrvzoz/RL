"""
Topic 2a: Monte Carlo PREDICTION on Blackjack.

Question: "If I always play the simple rule  'hit until 20, then stick',
           how good is each situation for me?"   ->  V(s) for that FIXED policy.

Blackjack-v1 (Gymnasium, Sutton & Barto rules):
    state  = (your sum, dealer's showing card 1..10 (1 = ace), usable ace? 0/1)
    action = 0 stick (stop taking cards), 1 hit (take another card)
    reward = +1 win, 0 draw, -1 lose, given ONLY when the game ends (0 before that)

Nobody gives us the card probabilities. We just play many games, and when each
game ends we walk backwards through it and average the result into every
state we visited. That average IS the value.

Part 1: watch ONE game and its backwards update, with real numbers.
Part 2: 10,000 vs 500,000 games -> value heatmaps (noisy vs smooth).
Part 3: first-visit vs every-visit (and why they agree in Blackjack).

Run:  python blackjack_prediction.py            (about 40 seconds)
      python blackjack_prediction.py --no-show
"""
import argparse
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from mc import mc_prediction, play_episode, stick_on_20

sys.stdout.reconfigure(encoding="utf-8")

ACTION = ["stick", "hit"]
DEALER = ["A"] + [str(c) for c in range(2, 11)]


def show(s):
    total, dealer, ace = s
    return f"(sum {total:>2}, dealer {DEALER[dealer - 1]:>2}, {'usable ace' if ace else 'no ace    '})"


def watch_one_game(env, V, N, rng, gamma=1.0):
    """Play one more game, print it, then do the backwards MC update out loud."""
    episode = play_episode(env, stick_on_20, rng)
    print("  The game, step by step (nothing is learned while playing):")
    for t, (s, a, r) in enumerate(episode):
        print(f"    step {t}: state {show(s)}  -> {ACTION[a]:<5}  reward {r:+.0f}")
    result = {1: "WIN", 0: "DRAW", -1: "LOSS"}[int(episode[-1][2])]
    print(f"    game over: {result}\n")
    print("  NOW the update, walking BACKWARDS from the last step (γ = 1):")
    G = 0.0
    for t in reversed(range(len(episode))):
        s, a, r = episode[t]
        G = r + gamma * G
        old = V[s]
        N[s] += 1
        V[s] += (G - V[s]) / N[s]
        print(f"    step {t}: G = {r:+.0f} + 1 × (G after) = {G:+.0f}   "
              f"V{show(s)}: {old:+.3f} + ({G:+.0f} − {old:+.3f}) / {N[s]} = {V[s]:+.3f}")


def value_grid(V, ace):
    # Rows = player sum 12..21, columns = dealer card A..10
    return np.array([[V.get((total, d, ace), np.nan) for d in range(1, 11)] for total in range(12, 22)])


def draw_heatmap(ax, grid, title):
    im = ax.imshow(grid, origin="lower", cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 else "black")
    ax.set_xticks(range(10), DEALER)
    ax.set_yticks(range(10), range(12, 22))
    ax.set_xlabel("Dealer's showing card")
    ax.set_ylabel("Your sum")
    ax.set_title(title, fontsize=10)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    env = gym.make("Blackjack-v1")

    print("=" * 72)
    print("PART 1 · One game and its backwards update (after 10,000 games of practice)")
    print("=" * 72)
    V, N = mc_prediction(env, stick_on_20, episodes=10_000, seed=1)
    rng = np.random.default_rng(7)
    watch_one_game(env, V, N, rng)

    print("\n" + "=" * 72)
    print("PART 2 · Value of 'stick on 20+' : 10,000 games vs 500,000 games")
    print("=" * 72)
    runs = {}
    for n in (10_000, 500_000):
        V, N = mc_prediction(env, stick_on_20, episodes=n, seed=0)
        runs[n] = (V, N)
        print(f"\n  {n:,} games: {len(V)} different states seen")
        for s in [(20, 10, 0), (21, 10, 0), (19, 10, 0), (13, 2, 0), (13, 2, 1), (21, 1, 1)]:
            print(f"    V{show(s)} = {V[s]:+.3f}   (from {N[s]:,} visits)")
    V, N = runs[500_000]
    no_ace = value_grid(V, 0)
    print(f"\n  500,000 games, no usable ace: average V for sums 12-19 = {np.nanmean(no_ace[:8]):+.3f}, "
          f"for sum 20 = {np.nanmean(no_ace[8]):+.3f}, for sum 21 = {np.nanmean(no_ace[9]):+.3f}")

    print("\n" + "=" * 72)
    print("PART 3 · First-visit vs every-visit")
    print("=" * 72)
    rng = np.random.default_rng(3)
    env.reset(seed=3)
    repeats = 0
    for _ in range(100_000):
        states = [s for s, _, _ in play_episode(env, stick_on_20, rng)]
        repeats += len(states) != len(set(states))
    print(f"  Games (out of 100,000) in which some state appeared twice: {repeats}")
    V_first, _ = mc_prediction(env, stick_on_20, episodes=50_000, seed=5, first_visit=True)
    V_every, _ = mc_prediction(env, stick_on_20, episodes=50_000, seed=5, first_visit=False)
    diff = max(abs(V_every[s] - V_first[s]) for s in V_first)
    print(f"  Biggest difference between first-visit and every-visit V: {diff:.6f}")
    print("  (Your sum only goes up, and a usable ace can only turn into a non-usable one,")
    print("   so a Blackjack state can never repeat inside one game.)")

    # ---- charts -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), gridspec_kw={"hspace": 0.32})
    for row, n in enumerate((10_000, 500_000)):
        Vn = runs[n][0]
        draw_heatmap(axes[row, 0], value_grid(Vn, 1), f"Usable ace · after {n:,} games")
        im = draw_heatmap(axes[row, 1], value_grid(Vn, 0), f"No usable ace · after {n:,} games")
    fig.colorbar(im, ax=axes, shrink=0.6, label="V(s): average result (−1 lose … +1 win)")
    fig.suptitle("Monte Carlo prediction: value of the policy 'stick on 20 or 21, else hit'")
    fig.savefig("prediction_heatmaps.png", dpi=110, bbox_inches="tight")

    fig = plt.figure(figsize=(12, 5))
    X, Y = np.meshgrid(range(1, 11), range(12, 22))
    for k, (ace, name) in enumerate([(1, "Usable ace"), (0, "No usable ace")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(X, Y, value_grid(V, ace), cmap="RdBu", vmin=-1, vmax=1, edgecolor="#555", lw=0.3)
        ax.set_xlabel("Dealer card (1 = A)")
        ax.set_ylabel("Your sum")
        ax.set_zlim(-1, 1)
        ax.view_init(elev=28, azim=-125)
        ax.set_title(f"{name} · 500,000 games")
    fig.savefig("prediction_surface.png", dpi=110, bbox_inches="tight")
    print("\nCharts saved to prediction_heatmaps.png and prediction_surface.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
