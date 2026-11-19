"""
Second game: A2C on Acrobot-v1, with exactly the same a2c.py.

    Acrobot: two poles joined like an arm, hanging down. You can only push the
    middle joint (the "elbow"). Swing the tip above the line to win.
    State  = 6 numbers (cos/sin of the two angles + their two spinning speeds)
    Action = 0 (push elbow left), 1 (no push), 2 (push elbow right)   -> 3 actions now
    Reward = -1 every step until you win, so a SHORTER game = a BETTER score.
             Max 500 steps (a random player usually scores about -500).

Run:  python acrobot_a2c.py            (about 2 minutes)
      python acrobot_a2c.py --no-show
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from a2c import evaluate, moving_average, train_a2c

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--steps", type=int, default=200_000)
    args = parser.parse_args()

    print("=" * 66)
    print("Training A2C on Acrobot-v1: 8 games at once, n_steps = 5")
    print("=" * 66)
    model, hist = train_a2c("Acrobot-v1", total_steps=args.steps, n_envs=8, n_steps=5,
                            gamma=0.99, lr=7e-4, ent_coef=0.01, seed=0, log_every=1000)
    ret, ends = np.array(hist["ep_return"]), np.array(hist["ep_end_step"])
    print(f"\n  Games finished during training: {len(ret)}")
    print(f"  First 50 games: avg {ret[:50].mean():.1f}     last 50 games: avg {ret[-50:].mean():.1f}")
    greedy = evaluate(model, "Acrobot-v1", games=20, greedy=True)
    sampled = evaluate(model, "Acrobot-v1", games=20, greedy=False)
    print(f"  Exam, greedy : avg {np.mean(greedy):.1f}  (best {max(greedy):.0f}, worst {min(greedy):.0f})")
    print(f"  Exam, sampled: avg {np.mean(sampled):.1f}")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axs[0]
    ax.plot(ends, ret, ".", ms=2, alpha=0.3, color="tab:blue", label="one game")
    ax.plot(ends[49:], moving_average(ret, 50), color="tab:blue", lw=2, label="average of last 50 games")
    ax.set(xlabel="Environment steps", ylabel="Score (−1 per step; higher = faster)",
           title="A2C on Acrobot-v1")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax = axs[1]
    ax.plot(hist["update_step"][99:], moving_average(hist["entropy"], 100), color="tab:purple", lw=2)
    ax.axhline(np.log(3), color="gray", ls="--", lw=1)
    ax.text(hist["update_step"][-1], np.log(3) - 0.02, "ln 3 = 1.099 (all 3 actions equal)",
            ha="right", va="top", fontsize=9, color="gray")
    ax.set(xlabel="Environment steps", ylabel="Entropy", title="Actor entropy (3 actions)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("acrobot_a2c.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to acrobot_a2c.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
