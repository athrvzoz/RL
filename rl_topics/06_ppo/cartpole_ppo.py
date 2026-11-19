"""
PPO on CartPole (2 discrete actions).

Prints progress (score, approx-KL, clip fraction, entropy), what the very first
update did to 6 moves, the exam score, and saves cartpole_ppo.png.

Run:  python cartpole_ppo.py            (about 2-3 minutes)
      python cartpole_ppo.py --no-show
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from ppo import evaluate, smooth, train

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    print(f"PPO on CartPole-v1: 8 games at once, 128 steps each → batch of 1,024 moves,\n"
          f"10 epochs × 8 minibatches per batch, clip ε = 0.2\n")
    agent, hist, eps = train("CartPole-v1", args.steps, seed=0, snapshot=True)

    print("\nThe FIRST update, 6 moves from the batch:")
    print(f"  {'action':>6} {'advantage':>10} {'π_old':>7} {'π_new':>7} {'ratio':>7}")
    for m in hist[0]["snapshot"]:
        print(f"  {'right' if m['action'] == 1 else 'left':>6} {m['adv']:>+10.2f} {m['p_old']:>7.3f} "
              f"{m['p_new']:>7.3f} {m['ratio']:>7.3f}")

    first500 = next((s for s, r in eps if r >= 500), None)
    print(f"\nGames played: {len(eps)}; first perfect (500) game at step {first500:,}" if first500
          else f"\nGames played: {len(eps)}; no perfect game")
    scores = evaluate("CartPole-v1", agent, games=20)
    print(f"Exam, 20 new games, most likely action: avg {np.mean(scores):.1f}, min {min(scores):.0f}")

    steps = [h["step"] for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    s, r = zip(*eps)
    axes[0].plot(s, r, ".", ms=2, alpha=0.3, label="one game")
    sm = smooth(r, 20)
    axes[0].plot(s[len(s) - len(sm):], sm, lw=2, label="average of 20 games")
    axes[0].set_title("PPO on CartPole")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[1].plot(steps, [h["approx_kl"] for h in hist], lw=1.5)
    axes[1].set_title("approx-KL: how far each update moved the policy")
    axes[1].set_xlabel("Environment steps")
    axes[2].plot(steps, [h["clip_frac"] for h in hist], lw=1.5, color="#C47A00")
    axes[2].set_title("clip fraction: share of moves hitting the ±20% fence")
    axes[2].set_xlabel("Environment steps")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("cartpole_ppo.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to cartpole_ppo.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
