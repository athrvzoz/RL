"""
Topic 2d: OFF-policy Monte Carlo with importance sampling (just the intuition).

Question: "How good is the state  (sum 13, dealer shows 2, usable ace)
           if I play 'stick on 20+'  (the TARGET policy) ...
           ... but I only have games played by someone pressing RANDOM buttons
           (the BEHAVIOUR policy)?"

Trick: re-weight every random game by how likely OUR policy was to play it.
    ratio ρ = product over the game of   π(a|s) / b(a|s)
            = 1 / 0.5 = 2 for every action our policy would also have chosen
            = 0       as soon as ONE action is one our policy would never choose
    ordinary IS :  V = Σ ρ·G / (number of games)
    weighted IS :  V = Σ ρ·G / Σ ρ

Run:  python off_policy.py            (about 1 minute)
      python off_policy.py --no-show
"""
import argparse
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text.blackjack import draw_card

from mc import as_state, random_policy, stick_on_20

sys.stdout.reconfigure(encoding="utf-8")

ACTION = ["stick", "hit"]


def play_from_soft13_vs_2(env, policy, rng):
    """One game that STARTS in (13, dealer 2, usable ace): ace + 2 in hand, dealer shows a 2."""
    env.reset()
    game = env.unwrapped
    game.player = [1, 2]
    game.dealer = [2, draw_card(game.np_random)]
    s = as_state(game._get_obs())
    episode, done = [], False
    while not done:
        a = policy(s, rng)
        obs, r, terminated, truncated, _ = env.step(a)
        episode.append((s, a, r))
        s = as_state(obs)
        done = terminated or truncated
    return episode


def ratio(episode):
    rho = 1.0
    for s, a, _ in episode:
        target_action = stick_on_20(s, None)
        rho *= (1.0 if a == target_action else 0.0) / 0.5      # π(a|s) / b(a|s)
    return rho


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    env = gym.make("Blackjack-v1")
    rng = np.random.default_rng(0)
    env.reset(seed=0)

    true_games = 100_000
    true_v = np.mean([play_from_soft13_vs_2(env, stick_on_20, rng)[-1][2] for _ in range(true_games)])
    print(f"'True' value by playing stick-on-20 itself {true_games:,} times: V = {true_v:+.3f}")
    print("(Sutton & Barto report -0.27726 for this state.)\n")

    print("Five games played by the RANDOM player, and how much each one counts:")
    for i in range(5):
        ep = play_from_soft13_vs_2(env, random_policy, rng)
        moves = ", ".join(f"{s[0]}:{ACTION[a]}" for s, a, _ in ep)
        print(f"  game {i + 1}: {moves:<34} result G = {ep[-1][2]:+.0f}   ratio ρ = {ratio(ep):g}")

    runs, episodes = 100, 1000
    ordinary = np.zeros((runs, episodes))
    weighted = np.zeros((runs, episodes))
    for run in range(runs):
        num, den = 0.0, 0.0
        for n in range(episodes):
            ep = play_from_soft13_vs_2(env, random_policy, rng)
            rho, G = ratio(ep), ep[-1][2]                     # γ = 1: G = final reward
            num += rho * G
            den += rho
            ordinary[run, n] = num / (n + 1)
            weighted[run, n] = num / den if den > 0 else 0.0
    mse_o = ((ordinary - true_v) ** 2).mean(axis=0)
    mse_w = ((weighted - true_v) ** 2).mean(axis=0)
    print(f"\nError (mean squared, {runs} runs) after N random games:")
    print(f"  {'N':>6}{'ordinary IS':>14}{'weighted IS':>14}")
    for n in (10, 100, 1000):
        print(f"  {n:>6}{mse_o[n - 1]:14.3f}{mse_w[n - 1]:14.3f}")
    print(f"  Average estimate after 1000 games: ordinary {ordinary[:, -1].mean():+.3f}, "
          f"weighted {weighted[:, -1].mean():+.3f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(1, episodes + 1), mse_o, color="#eb6834", lw=2, label="ordinary importance sampling")
    ax.plot(np.arange(1, episodes + 1), mse_w, color="#2a78d6", lw=2, label="weighted importance sampling")
    ax.set_xscale("log")
    ax.set_xlabel("Games played by the RANDOM player (log scale)")
    ax.set_ylabel(f"Mean squared error ({runs} runs)")
    ax.set_title("Learning the value of 'stick on 20+' from someone else's games")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("off_policy.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to off_policy.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
