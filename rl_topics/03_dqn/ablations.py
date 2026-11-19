"""
Ablations: switch DQN's tricks off one at a time and see what breaks.

    full DQN        replay buffer + target network
    no replay       learn only from the move just made (like tabular Q-learning)
    no target net   the targets come from the network being trained
    Double DQN      full DQN + the Double trick

Each variant: 2 seeds × 40,000 steps. Also tracks the network's average max-Q
on a fixed set of states, to spot over-optimistic values.

Run:  python ablations.py            (about 10-15 minutes)
      python ablations.py --no-show
"""
import argparse
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from dqn import evaluate, train

sys.stdout.reconfigure(encoding="utf-8")

VARIANTS = {
    "full DQN": dict(),
    "no replay": dict(use_replay=False),
    "no target net": dict(use_target=False),
    "Double DQN": dict(double=True),
}
COLORS = {"full DQN": "#2C4BD1", "no replay": "#C2413A", "no target net": "#C47A00", "Double DQN": "#1E8A5A"}


def watch_states(n=200, seed=123):
    """States from random play: a fixed 'exam' to read the network's Q-values on."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    out, (s, _) = [], env.reset(seed=seed)
    while len(out) < n:
        out.append(s)
        s, _, term, trunc, _ = env.step(int(rng.integers(2)))
        if term or trunc:
            s, _ = env.reset()
    return np.array(out, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    ws = watch_states()

    results = {}
    for name, kw in VARIANTS.items():
        runs = []
        for seed in range(args.seeds):
            net, hist = train(total_steps=args.steps, seed=seed, watch_states=ws, verbose=False, **kw)
            final = evaluate(hist["final_net"], games=20, seed=70_000)
            best = evaluate(net, games=20, seed=70_000)
            runs.append((hist, final, best))
            evals = [s for _, s in hist["evals"]]
            print(f"  {name:<14} seed {seed}: test score during training: "
                  f"mean {np.mean(evals):5.1f}, best {max(evals):5.1f} | final net {final:5.1f} | "
                  f"best net {best:5.1f} | max-Q on exam states at end {hist['watch_q'][-1][1]:6.1f}")
        results[name] = runs
        print()

    print("Summary (average over seeds)")
    print(f"  {'variant':<14}{'avg test during training':>26}{'final net':>12}{'best net':>11}{'avg max-Q at end':>19}")
    for name, runs in results.items():
        avg_eval = np.mean([np.mean([s for _, s in h["evals"]]) for h, _, _ in runs])
        print(f"  {name:<14}{avg_eval:>26.1f}{np.mean([f for _, f, _ in runs]):>12.1f}"
              f"{np.mean([b for _, _, b in runs]):>11.1f}{np.mean([h['watch_q'][-1][1] for h, _, _ in runs]):>19.1f}")
    print("  (for reference: a perfect CartPole policy scores 500; with γ = 0.99 the largest\n"
          "   possible Q is about 1/(1−0.99) = 100)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, runs in results.items():
        steps = [s for s, _ in runs[0][0]["evals"]]
        mean_eval = np.mean([[s for _, s in h["evals"]] for h, _, _ in runs], axis=0)
        mean_q = np.mean([[q for _, q in h["watch_q"]] for h, _, _ in runs], axis=0)
        ax1.plot(steps, mean_eval, "o-", ms=3, color=COLORS[name], label=name)
        ax2.plot(steps, mean_q, "o-", ms=3, color=COLORS[name], label=name)
    ax1.set_title(f"Test score (no exploration), mean of {args.seeds} seeds")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.axhline(100, color="gray", ls="--", lw=1, label="largest possible Q (≈100)")
    ax2.set_title("Average max-Q on 200 fixed states")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Q")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("ablations.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to ablations.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
