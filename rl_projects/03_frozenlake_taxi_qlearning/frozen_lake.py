"""
Project 3a: FrozenLake with Q-learning.

    S F F F      S = start          F = frozen (safe)
    F H F H      H = hole (fall in, game over, reward 0)
    F F F H      G = goal (reward +1, game over)
    H F F G

Reward is 1 ONLY for reaching the goal; everything else gives 0.
So the average reward over many games = the success rate.

Part 1: normal ice (you go where you aim).
Part 2: slippery ice (only 1/3 chance of going where you aim!).
Part 3: compare with value iteration, which IS allowed the rule book
        (env.unwrapped.P), to see if Q-learning found the right answer
        on its own.

Run:  python frozen_lake.py            (about 30 seconds)
      python frozen_lake.py --no-show
"""
import argparse
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from q_learning import evaluate, moving_average, train

sys.stdout.reconfigure(encoding="utf-8")

ARROWS = "←↓→↑"            # FrozenLake actions: 0=left 1=down 2=right 3=up
MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]


def show_policy(Q):
    for r in range(4):
        row = []
        for c in range(4):
            s, tile = r * 4 + c, MAP[r][c]
            row.append(tile if tile in "HG" else ARROWS[int(np.argmax(Q[s]))])
        print("     " + "  ".join(row))


def show_q_table(Q):
    print("     state   " + "".join(f"{a + ' ' + n:>9}" for a, n in zip(ARROWS, ["left", "down", "right", "up"])))
    for s in range(16):
        tile = MAP[s // 4][s % 4]
        cells = "".join(f"{q:9.3f}" for q in Q[s])
        print(f"     {s:>2} ({tile})  {cells}")


def value_iteration_policy(env, gamma=0.99, theta=1e-10):
    """Project 2's algorithm, using FrozenLake's own rule book P[s][a] = [(prob, s2, r, done)]."""
    P = env.unwrapped.P
    V = np.zeros(16)
    while True:
        Q = np.array([[sum(p * (r + gamma * V[s2] * (not done)) for p, s2, r, done in P[s][a])
                       for a in range(4)] for s in range(16)])
        new_V = Q.max(axis=1)
        if np.max(np.abs(new_V - V)) < theta:
            return Q
        V = new_V


def run(slippery, episodes, eps_decay, alpha):
    env = gym.make("FrozenLake-v1", is_slippery=slippery)
    Q, rewards = train(env, episodes, alpha=alpha, gamma=0.99, eps_decay=eps_decay)
    success, steps = evaluate(env, Q)
    return env, Q, rewards, success, steps


def draw_policy(ax, Q, title):
    colors = {"S": "#cfe8ff", "F": "#cfe8ff", "H": "#1f3b57", "G": "#4caf50"}
    for r in range(4):
        for c in range(4):
            tile = MAP[r][c]
            ax.add_patch(plt.Rectangle((c, 3 - r), 1, 1, color=colors[tile], ec="white"))
            label = {"H": "HOLE", "G": "GOAL"}.get(tile) or ARROWS[int(np.argmax(Q[r * 4 + c]))]
            ax.text(c + 0.5, 3.5 - r, label, ha="center", va="center",
                    fontsize=9 if tile in "HG" else 22, color="white" if tile == "H" else "black")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("PART 1 · Normal ice: 2,000 training games")
    print("=" * 60)
    env1, Q1, rewards1, success1, steps1 = run(slippery=False, episodes=2000, eps_decay=0.995, alpha=0.1)
    print("\n  Learned Q-table (one row per square, one column per action):")
    show_q_table(Q1)
    print("\n  Learned policy (arrow = action with the biggest Q):")
    show_policy(Q1)
    print(f"\n  Test, 1000 games with no exploration: success {100 * success1:.1f}%, avg steps {steps1:.1f}")

    print("\n" + "=" * 60)
    print("PART 2 · Slippery ice: 20,000 training games")
    print("=" * 60)
    env2, Q2, rewards2, success2, steps2 = run(slippery=True, episodes=20000, eps_decay=0.9995, alpha=0.05)
    print("\n  Learned policy:")
    show_policy(Q2)
    print(f"\n  Test, 1000 games with no exploration: success {100 * success2:.1f}%, avg steps {steps2:.1f}")

    print("\n" + "=" * 60)
    print("PART 3 · Answer key: value iteration WITH the rule book (slippery)")
    print("=" * 60)
    Q_vi = value_iteration_policy(env2)
    success_vi, steps_vi = evaluate(env2, Q_vi)
    print("\n  Best possible policy:")
    show_policy(Q_vi)
    print(f"\n  Test: success {100 * success_vi:.1f}%, avg steps {steps_vi:.1f}")
    safe = [s for s in range(16) if MAP[s // 4][s % 4] not in "HG"]
    same = sum(np.argmax(Q2[s]) == np.argmax(Q_vi[s]) for s in safe)
    print(f"  Q-learning picked the same action on {same} of {len(safe)} squares, "
          f"and scores {100 * success2:.1f}% vs the best possible {100 * success_vi:.1f}%")

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(100 * moving_average(rewards1), label="normal ice")
    ax.plot(100 * moving_average(rewards2), label="slippery ice")
    ax.axhline(100 * success_vi, color="gray", ls="--", lw=1, label="best possible on slippery ice")
    ax.set_xlabel("Training game")
    ax.set_ylabel("Success rate, last 100 games (%)")
    ax.set_title("FrozenLake: Q-learning learning curves (includes exploration moves)")
    ax.legend()
    ax.grid(alpha=0.3)
    draw_policy(fig.add_subplot(2, 3, 4), Q1, f"Learned, normal ice ({100 * success1:.0f}%)")
    draw_policy(fig.add_subplot(2, 3, 5), Q2, f"Learned, slippery ice ({100 * success2:.0f}%)")
    draw_policy(fig.add_subplot(2, 3, 6), Q_vi, f"Answer key, slippery ({100 * success_vi:.0f}%)")
    fig.tight_layout()
    fig.savefig("frozen_lake_results.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to frozen_lake_results.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
