"""
Policy gradient on a 3-armed bandit: watch the PROBABILITIES move.

No states, no network: the policy is just 3 numbers (preferences, "logits")
turned into probabilities by softmax. The true average payouts are
    arm 0: 1.0     arm 1: 2.0     arm 2: 1.5     (each pull: that average + noise)

Part 1: one update by hand, with and without a baseline.
Part 2: 1,000 pulls, REINFORCE with and without a baseline, 200 runs averaged.

Run:  python bandit_pg.py            (a few seconds)
      python bandit_pg.py --no-show
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

TRUE_MEANS = np.array([1.0, 2.0, 1.5])


def softmax(h):
    e = np.exp(h - h.max())               # subtract the max: same answer, no overflow
    return e / e.sum()


def part1():
    print("=" * 66)
    print("PART 1 · One update by hand (α = 0.1)")
    print("=" * 66)
    h = np.array([0.0, 0.0, 0.0])
    p = softmax(h)
    print(f"  preferences h = {h}  →  probabilities π = {np.round(p, 3)}")
    a, r = 1, 3.0
    print(f"  rolled the dice: picked arm {a}, got reward r = {r}")
    print("  gradient of log π(a): (1 − π_a) for the chosen arm, −π_b for the others")
    grad = -p.copy()
    grad[a] += 1
    print(f"  ∇log π(arm {a}) = {np.round(grad, 3)}")
    for name, b in [("no baseline", 0.0), ("baseline b = 2.5", 2.5), ("baseline b = 3.5", 3.5)]:
        h2 = h + 0.1 * (r - b) * grad
        print(f"  {name:<18} step = 0.1 × ({r} − {b}) × ∇  → h = {np.round(h2, 3)}  "
              f"→ π = {np.round(softmax(h2), 3)}")


def run(baseline, steps, rng, alpha=0.1):
    h = np.zeros(3)
    avg_r = 0.0
    p_best = np.zeros(steps)
    rewards = np.zeros(steps)
    for t in range(steps):
        p = softmax(h)
        a = rng.choice(3, p=p)                       # roll the AGENT's dice
        r = TRUE_MEANS[a] + rng.normal()
        b = avg_r if baseline else 0.0
        grad = -p
        grad[a] += 1
        h += alpha * (r - b) * grad                  # REINFORCE (one-step game)
        avg_r += (r - avg_r) / (t + 1)               # running average reward = the baseline
        p_best[t] = p[1]
        rewards[t] = r
    return p_best, rewards, softmax(h)


def part2(no_show):
    print("\n" + "=" * 66)
    print("PART 2 · 1,000 pulls, 200 runs averaged (α = 0.1)")
    print("=" * 66)
    steps, runs = 1000, 200
    curves = {}
    for name, base in [("no baseline", False), ("with baseline", True)]:
        rng = np.random.default_rng(0)
        P, R, finals = [], [], []
        for _ in range(runs):
            pb, rw, fp = run(base, steps, rng)
            P.append(pb)
            R.append(rw)
            finals.append(fp)
        P, R, finals = np.array(P), np.array(R), np.array(finals)
        curves[name] = P.mean(axis=0)
        print(f"  {name:<14} P(best arm) after 100 pulls {P[:, 99].mean():.3f}, after 1000 {P[:, -1].mean():.3f}"
              f"   avg reward {R.mean():.3f}   runs that ended preferring a wrong arm: "
              f"{int((finals.argmax(axis=1) != 1).sum())} of {runs}")
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, c in curves.items():
        ax.plot(c, lw=2, label=name)
    ax.set_xlabel("Pull")
    ax.set_ylabel("Probability of choosing the best arm")
    ax.set_title("3-armed bandit: policy gradient, average of 200 runs")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("bandit_pg.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to bandit_pg.png")
    if not no_show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    part1()
    part2(args.no_show)
