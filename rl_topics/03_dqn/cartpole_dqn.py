"""
Train a DQN on CartPole and look inside it.

    CartPole: keep a pole balanced on a moving cart.
    state  = 4 real numbers: cart position, cart speed, pole angle, pole spin
    action = 0 push left, 1 push right
    reward = +1 for every step the pole stays up; the game stops at 500 steps
             (truncated) or when the pole falls / the cart leaves the track (terminated)

Prints training progress, tests the best network, shows its Q-values for a few
hand-picked states, and saves cartpole_dqn.png.

Run:  python cartpole_dqn.py            (about 2-3 minutes)
      python cartpole_dqn.py --no-show
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn import evaluate, moving_average, train

sys.stdout.reconfigure(encoding="utf-8")

# Hand-picked states: [cart position, cart speed, pole angle (radians), pole spin]
SHOW_STATES = {
    "balanced, still":            [0.0, 0.0, 0.00, 0.0],
    "pole leaning RIGHT":         [0.0, 0.0, 0.10, 0.0],
    "pole leaning LEFT":          [0.0, 0.0, -0.10, 0.0],
    "pole falling right fast":    [0.0, 0.0, 0.05, 1.0],
    "almost fallen right":        [0.0, 0.0, 0.19, 0.5],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    print(f"Training DQN on CartPole for {args.steps:,} steps\n")
    net, hist = train(total_steps=args.steps, seed=0)

    score = evaluate(net, games=100, seed=50_000)
    print(f"\nBest network, 100 test games with no exploration: average score {score:.1f} (max 500)")
    print(f"Training games played: {len(hist['returns'])}")

    print("\nWhat the network thinks (Q-values) in some states:")
    with torch.no_grad():
        for name, st in SHOW_STATES.items():
            q = net(torch.tensor(st, dtype=torch.float32)).numpy()
            best = "RIGHT" if q[1] > q[0] else "LEFT"
            print(f"  {name:<26} Q(left) {q[0]:6.1f}   Q(right) {q[1]:6.1f}   -> push {best}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ends = np.array(hist["ends"])
    ax1.plot(ends, hist["returns"], alpha=0.3, lw=0.8, label="each training game")
    ma = moving_average(hist["returns"])
    ax1.plot(ends[len(ends) - len(ma):], ma, lw=2, label="average of last 20")
    st, sc = zip(*hist["evals"])
    ax1.plot(st, sc, "o-", ms=3, lw=1.2, label="test, no exploration")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Score (steps balanced)")
    ax1.set_title("DQN on CartPole")
    ax1.legend()
    ax1.grid(alpha=0.3)
    if hist["losses"]:
        ls, lv = zip(*hist["losses"])
        ax2.plot(ls, lv, lw=0.8)
    ax2.set_yscale("log")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Loss (Huber), log scale")
    ax2.set_title("The loss does NOT simply go down")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("cartpole_dqn.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to cartpole_dqn.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
