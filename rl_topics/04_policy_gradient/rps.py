"""
Rock-paper-scissors: a game where the BEST policy is random.

Q-learning's policy is "argmax Q": always the same move in the same state.
That is a DETERMINISTIC policy, and in rock-paper-scissors a smart opponent eats it alive.

Part 1: 3 fixed policies vs a "spy" opponent who counters your most common move so far.
Part 2: REINFORCE learning against a biased opponent (likes rock), probabilities printed.
Part 3: REINFORCE against the spy: the probabilities never settle, they swing from move
        to move as the agent dodges the spy (learning in a 2-player game can cycle).

Score per round: win +1, draw 0, lose −1.
Run:  python rps.py            (a few seconds)
      python rps.py --no-show
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from softmax_bandit import grad_log_softmax, softmax

sys.stdout.reconfigure(encoding="utf-8")

MOVES = ["rock", "paper", "scissors"]
BEATS = {0: 1, 1: 2, 2: 0}                 # the move that beats rock is paper (1), etc.


def score(me, them):
    if me == them:
        return 0.0
    return 1.0 if BEATS[them] == me else -1.0


def spy(my_counts, rng):
    """Opponent: counter my most common move so far (random on ties)."""
    best = np.flatnonzero(my_counts == my_counts.max())
    return BEATS[int(rng.choice(best))]


def part1(rng, rounds=3000):
    print("=" * 64)
    print("PART 1 · Fixed policies vs the spy (3,000 rounds each)")
    print("=" * 64)
    policies = {"always rock (like argmax Q)": [1, 0, 0], "70% rock, 30% paper": [0.7, 0.3, 0],
                "1/3 each (random)": [1 / 3, 1 / 3, 1 / 3]}
    for name, p in policies.items():
        counts, total = np.zeros(3), 0.0
        for _ in range(rounds):
            them = spy(counts, rng)
            me = rng.choice(3, p=p)
            total += score(me, them)
            counts[me] += 1
        print(f"  {name:<28} average score per round {total / rounds:+.3f}")


def learn(rng, opponent, rounds, alpha):
    theta = np.zeros(3)
    counts, avg_r, hist, total = np.zeros(3), 0.0, [softmax(theta)], 0.0
    for t in range(1, rounds + 1):
        p = softmax(theta)
        me = rng.choice(3, p=p)                # agent's dice
        them = opponent(counts, rng)           # opponent's choice (part of the "world")
        r = score(me, them)
        theta += alpha * (r - avg_r) * grad_log_softmax(theta, me)   # REINFORCE with baseline
        avg_r += (r - avg_r) / t
        counts[me] += 1
        total += r
        hist.append(softmax(theta))
    return np.array(hist), total / rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    part1(rng)

    print("\n" + "=" * 64)
    print("PART 2 · REINFORCE vs a biased opponent: rock 50%, paper 25%, scissors 25%")
    print("=" * 64)
    biased = lambda counts, rng: int(rng.choice(3, p=[0.5, 0.25, 0.25]))
    h_b, avg_b = learn(np.random.default_rng(1), biased, 2000, 0.05)
    print("     round    rock   paper  scissors")
    for t in [0, 50, 200, 500, 1000, 2000]:
        print(f"     {t:>5}   {h_b[t, 0]:.3f}   {h_b[t, 1]:.3f}   {h_b[t, 2]:.3f}")
    print(f"  average score over all 2,000 rounds: {avg_b:+.3f}   (always-paper would get +0.250)")

    print("\n" + "=" * 64)
    print("PART 3 · REINFORCE vs the spy")
    print("=" * 64)
    h_s, avg_s = learn(np.random.default_rng(2), spy, 3000, 0.05)
    print("     round    rock   paper  scissors")
    for t in [0, 500, 1000, 2000, 3000]:
        print(f"     {t:>5}   {h_s[t, 0]:.3f}   {h_s[t, 1]:.3f}   {h_s[t, 2]:.3f}")
    print(f"  largest probability ever reached after round 500: {h_s[500:].max():.3f}")
    print(f"  average score over all 3,000 rounds: {avg_s:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, h, title in [(axes[0], h_b, "vs biased opponent (likes rock)"), (axes[1], h_s, "vs the spy")]:
        for k in range(3):
            ax.plot(h[:, k], label=MOVES[k], lw=1.8)
        ax.set_title(f"REINFORCE {title}")
        ax.set_xlabel("Round")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Agent's probability of the move")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig("rps_probs.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to rps_probs.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
