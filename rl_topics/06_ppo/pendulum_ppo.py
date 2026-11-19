"""
PPO on Pendulum: a CONTINUOUS action.

    Pendulum: swing a pendulum up and hold it upright.
    state  = [cos(angle), sin(angle), spin speed]
    action = ONE real number: the motor torque, anywhere in [−2, 2]
    reward = −(angle² + 0.1·speed² + 0.001·torque²) every step (0 is perfect), 200 steps per game

The actor now outputs the CENTRE (mean) of a bell curve; the width is a learned number.
It rolls the bell-curve dice to choose a torque.

Run:  python pendulum_ppo.py            (about 5-7 minutes)
      python pendulum_ppo.py --no-show
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from ppo import evaluate, smooth, train

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    print("PPO on Pendulum-v1 (continuous torque), γ = 0.9, rewards scaled ×0.1\n")
    agent, hist, eps = train("Pendulum-v1", args.steps, seed=0, gamma=0.9, lr=1e-3,
                             num_steps=256, reward_scale=0.1, ent_coef=0.0)
    s, r = zip(*eps)
    print(f"\nFirst 50 games: avg {np.mean(r[:50]):.0f}    last 50 games: avg {np.mean(r[-50:]):.0f}")
    scores = evaluate("Pendulum-v1", agent, games=20)
    print(f"Exam, 20 new games, centre of the bell curve: avg {np.mean(scores):.0f}, "
          f"best {max(scores):.0f}, worst {min(scores):.0f}   (random play ≈ −1200, 0 = perfect)")
    print(f"Learned bell-curve width (std of the torque): {agent.log_std.exp().item():.3f}  (started at 0.607)")

    print("\nThe policy's chosen torque (bell-curve centre), pendulum still, at different angles:")
    with torch.no_grad():
        for deg in [-90, -30, -10, 0, 10, 30, 90, 180]:
            th = np.radians(deg)
            mean = agent.actor(torch.tensor([np.cos(th), np.sin(th), 0.0], dtype=torch.float32)).item()
            print(f"  angle {deg:>+4}° (0 = upright)   torque {mean:+.2f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(s, r, ".", ms=2, alpha=0.3, label="one game")
    sm = smooth(r, 20)
    ax.plot(s[len(s) - len(sm):], sm, lw=2, label="average of 20 games")
    ax.set_title("PPO on Pendulum (continuous action)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Score (0 = perfect)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("pendulum_ppo.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to pendulum_ppo.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
