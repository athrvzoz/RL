"""
Topic 2e: turning the knobs of Monte Carlo control on Blackjack.

A. How many training games are enough?        10k / 50k / 200k games, ε = 0.1
B. How much exploration?                         ε = 0.01 / 0.1 / 0.3, 100k games each
C. Exploring starts instead of ε-greedy          random FIRST action, then greedy, 100k games
D. Monte Carlo vs Q-learning (TD) on Blackjack   same 100k games budget

Every learned policy is tested greedily (no exploration) on 50,000 games.

Run:  python experiments.py            (about 4 minutes)
      python experiments.py --no-show
"""
import argparse
import sys
from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from mc import as_state, epsilon_greedy_policy, greedy, mc_control, stick_on_20, win_draw_loss

sys.stdout.reconfigure(encoding="utf-8")
TEST_GAMES = 50_000


def q_learning(env, episodes, alpha=0.05, eps=0.1, gamma=1.0, seed=0):
    """Project 3's Q-learning, but with a dict because Blackjack states are tuples."""
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    Q = defaultdict(lambda: np.zeros(2))
    policy = epsilon_greedy_policy(Q, eps)
    for _ in range(episodes):
        s = as_state(env.reset()[0])
        done = False
        while not done:
            a = policy(s, rng)
            obs, r, terminated, truncated, _ = env.step(a)
            s2 = as_state(obs)
            target = r + gamma * np.max(Q[s2]) * (not terminated)    # bootstrapped guess
            Q[s][a] += alpha * (target - Q[s][a])                    # update NOW, every step
            s = s2
            done = terminated or truncated
    return Q


def test(env, Q):
    return win_draw_loss(env, greedy(Q), games=TEST_GAMES)


def row(label, res):
    w, d, l, avg = res
    print(f"  {label:<34}{w:7.1f}{d:7.1f}{l:7.1f}{avg:+9.3f}")
    return res


def header(title):
    print("\n" + title)
    print(f"  {'':<34}{'win%':>7}{'draw%':>7}{'loss%':>7}{'avg':>9}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    env = gym.make("Blackjack-v1")
    results = {}

    header("Baseline")
    results["stick on 20+"] = row("stick on 20+ (no learning)", win_draw_loss(env, stick_on_20, games=TEST_GAMES))

    header("A · Number of training games (ε = 0.1)")
    for n in (10_000, 50_000, 200_000):
        Q, N, _ = mc_control(env, n, eps=0.1, seed=0)
        results[f"MC {n // 1000}k games"] = row(f"MC, {n:,} games", test(env, Q))

    header("B · Exploration ε (100,000 games)")
    for eps in (0.01, 0.1, 0.3):
        Q, N, rewards = mc_control(env, 100_000, eps=eps, seed=0)
        unseen = sum(int((n == 0).any()) for n in N.values())
        res = row(f"MC, ε = {eps}", test(env, Q))
        print(f"  {'':<34}training avg {rewards.mean():+.3f}, states with an action never tried: {unseen}")
        results[f"MC ε={eps}"] = res

    header("C · Exploring starts (100,000 games)")
    Q, N, _ = mc_control(env, 100_000, seed=0, exploring_starts=True)
    results["MC exploring starts"] = row("MC, random first action, then greedy", test(env, Q))

    header("D · Monte Carlo vs Q-learning (100,000 games, ε = 0.1)")
    results["Q-learning α=0.05"] = row("Q-learning, α = 0.05", test(env, q_learning(env, 100_000, alpha=0.05)))
    results["Q-learning α=0.01"] = row("Q-learning, α = 0.01", test(env, q_learning(env, 100_000, alpha=0.01)))
    print(f"  (MC with the same 100,000 games and ε = 0.1 is the 'MC, ε = 0.1' row in B)")

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results)
    avgs = [results[n][3] for n in names]
    colors = ["#52514e"] + ["#eb6834"] * 3 + ["#2a78d6"] * 3 + ["#1baf7a"] + ["#6250d6"] * 2
    ax.barh(names, avgs, color=colors)
    for i, v in enumerate(avgs):
        ax.text(v - 0.003, i, f"{v:+.3f}", ha="right", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Average result per game ({TEST_GAMES:,} greedy test games; 0 = break even)")
    ax.set_title("Blackjack: what each knob does to the learned policy")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig("experiments.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to experiments.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
