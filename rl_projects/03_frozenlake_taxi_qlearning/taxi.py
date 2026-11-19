"""
Project 3b: Taxi with Q-learning.

    +---------+
    |R: | : :G|     R, G, Y, B = the 4 pickup / drop-off spots
    | : | : : |     |          = walls the taxi can't drive through
    | : : : : |     yellow box = the taxi
    | | : | : |
    |Y| : |B: |
    +---------+

A passenger waits at one spot and wants to go to another. The taxi must
drive there, pick them up, drive to the destination, and drop them off.

State = taxi row (5) x taxi column (5) x passenger location (4 spots + "in taxi" = 5)
        x destination (4)  =  5 x 5 x 5 x 4  =  500 states
Actions = 0 south, 1 north, 2 east, 3 west, 4 pickup, 5 dropoff
Rewards = -1 every step (hurry up!), +20 for a correct drop-off,
          -10 for an illegal pickup or drop-off.

Run:  python taxi.py            (about 20 seconds)
      python taxi.py --no-show
"""
import argparse
import re
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from q_learning import evaluate, moving_average, train

sys.stdout.reconfigure(encoding="utf-8")

ACTION_NAMES = ["south", "north", "east", "west", "pickup", "dropoff"]
SPOTS = ["R", "G", "Y", "B", "in taxi"]


def describe(env, s):
    row, col, passenger, dest = env.unwrapped.decode(s)
    return f"taxi at (row {row}, col {col}), passenger at {SPOTS[passenger]}, wants to go to {SPOTS[dest]}"


def strip_colors(text):
    # The board uses terminal colour codes; keep them only if the terminal shows colours.
    return text if sys.stdout.isatty() else re.sub(r"\x1b\[[0-9;]*m", "", text)


def watch_one_game(Q, seed):
    env = gym.make("Taxi-v4", render_mode="ansi")
    s, _ = env.reset(seed=seed)
    print(f"  Start: {describe(env, s)}")
    print("  Q-values in this state:  " + "  ".join(f"{n} {q:+.1f}" for n, q in zip(ACTION_NAMES, Q[s])))
    print(strip_colors(env.render()))
    total, done, step = 0, False, 0
    while not done:
        a = int(np.argmax(Q[s]))
        s, r, terminated, truncated, _ = env.step(a)
        total += r
        step += 1
        done = terminated or truncated
        print(f"  step {step:>2}: {ACTION_NAMES[a]:<8} reward {r:+}")
        if a >= 4 or done:                              # show the board at pickup and drop-off
            print(strip_colors(env.render()))
    print(f"  Finished in {step} steps, total reward {total:+}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    env = gym.make("Taxi-v4")
    print(f"Taxi: {env.observation_space.n} states x {env.action_space.n} actions "
          f"= {env.observation_space.n * env.action_space.n} numbers in the Q-table\n")

    # Before training: a random driver (Q all zeros -> random tie-breaking).
    random_score, random_steps = evaluate(env, np.zeros((500, 6)), games=200)
    print(f"Before training (random driver): avg reward {random_score:.1f}, avg steps {random_steps:.0f} "
          f"(the game stops itself at 200 steps)\n")

    Q, rewards = train(env, args.episodes, alpha=0.1, gamma=0.99, eps_decay=0.999)
    for ep in range(500, args.episodes + 1, 500):
        print(f"  after {ep:>5} training games: avg reward of last 100 = {rewards[ep - 100:ep].mean():7.1f}")

    score, steps = evaluate(env, Q)
    print(f"\nAfter training (no exploration, 1000 games): avg reward {score:.1f}, avg steps {steps:.1f}")
    filled = np.count_nonzero(Q.any(axis=1))
    print(f"States the agent visited and learned about: {filled} of 500\n")

    print("=" * 60)
    print("Watch the trained taxi drive one game")
    print("=" * 60)
    watch_one_game(Q, seed=7)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(rewards, alpha=0.25, lw=0.6, label="each training game")
    ax.plot(np.arange(99, len(rewards)), moving_average(rewards), lw=2, label="average of last 100")
    ax.set_ylim(-800, 20)
    ax.set_xlabel("Training game")
    ax.set_ylabel("Total reward in the game")
    ax.set_title("Taxi: Q-learning learning curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("taxi_results.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to taxi_results.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
