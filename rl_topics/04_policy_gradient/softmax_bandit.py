"""
Policy gradients by hand, with numpy only (no neural network).

The "policy" here is just 2 or 3 numbers called LOGITS (θ). Softmax turns them into
action probabilities. REINFORCE nudges the logits so good actions become more likely.

Part 1: softmax, logits → probabilities (the [2, 1, 0] example)
Part 2: ONE REINFORCE update on a 2-action softmax, every number printed
Part 3: the log-derivative trick, checked: exact gradient vs average of sampled G·∇log π
Part 4: why a baseline helps: same average gradient, much smaller spread (variance)
Part 5: a 3-armed bandit. The WORLD's win chances (0.2, 0.5, 0.8) never change;
        the AGENT's action probabilities move. With and without a baseline.

Run:  python softmax_bandit.py            (a few seconds)
      python softmax_bandit.py --no-show
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def softmax(logits):
    z = np.exp(logits - np.max(logits))      # subtracting the max changes nothing, avoids overflow
    return z / z.sum()


def grad_log_softmax(logits, a):
    """∇_θ log π(a) for a softmax policy = onehot(a) − π. (Derived in the booklet, chapter 05.)"""
    g = -softmax(logits)
    g[a] += 1.0
    return g


def part1():
    print("=" * 64)
    print("PART 1 · Softmax: logits → probabilities")
    print("=" * 64)
    logits = np.array([2.0, 1.0, 0.0])
    e = np.exp(logits)
    print(f"  logits θ        = {logits}")
    print(f"  e^θ             = {np.round(e, 3)}   sum = {e.sum():.3f}")
    print(f"  probabilities π = {np.round(e / e.sum(), 3)}   (add up to {np.sum(e / e.sum()):.0f})")
    print(f"  add 10 to every logit → {np.round(softmax(logits + 10), 3)}   (no change!)")


def part2():
    print("\n" + "=" * 64)
    print("PART 2 · One REINFORCE update, 2 actions (Left, Right), by hand")
    print("=" * 64)
    theta = np.array([1.0, 0.0])
    alpha, a, G = 0.1, 1, 2.0
    p = softmax(theta)
    g = grad_log_softmax(theta, a)
    new_theta = theta + alpha * G * g
    p_new = softmax(new_theta)
    print(f"  logits θ = {theta}  →  π = [L {p[0]:.3f}, R {p[1]:.3f}]")
    print(f"  the agent rolled its dice and got a = Right (only {p[1]:.1%} likely); return G = {G}")
    print(f"  ∇log π(Right) = onehot(Right) − π = [0 − {p[0]:.3f}, 1 − {p[1]:.3f}] = {np.round(g, 3)}")
    print(f"  step = α·G·∇log π = {alpha}·{G}·{np.round(g, 3)} = {np.round(alpha * G * g, 3)}")
    print(f"  new θ = {np.round(new_theta, 3)}  →  π = [L {p_new[0]:.3f}, R {p_new[1]:.3f}]")
    # same update, but a bad outcome
    G_bad = -2.0
    t_bad = theta + alpha * G_bad * g
    pb = softmax(t_bad)
    print(f"  if instead G = {G_bad}: new θ = {np.round(t_bad, 3)} → π = [L {pb[0]:.3f}, R {pb[1]:.3f}]")
    # and the more likely action, same G
    g0 = grad_log_softmax(theta, 0)
    t0 = theta + alpha * G * g0
    p0 = softmax(t0)
    print(f"  if a = Left (likely) with G = {G}: ∇log π(Left) = {np.round(g0, 3)}, "
          f"new π = [L {p0[0]:.3f}, R {p0[1]:.3f}]")


def part3(rng):
    print("\n" + "=" * 64)
    print("PART 3 · Log-derivative trick: exact gradient vs sampled estimate")
    print("=" * 64)
    theta = np.array([1.0, 0.0])
    R = np.array([1.0, 3.0])                 # Left always pays 1, Right always pays 3
    p = softmax(theta)
    J = p @ R
    # exact: J = Σ π(a)·R(a). ∂π(a)/∂θ for softmax = π(a)·(onehot(a) − π)
    exact = sum(p[a] * grad_log_softmax(theta, a) * R[a] for a in range(2))
    print(f"  π = {np.round(p, 3)},  rewards R = {R},  J(θ) = Σ π·R = {J:.3f}")
    print(f"  exact gradient  ∇J = Σ π(a)·∇log π(a)·R(a) = {np.round(exact, 4)}")
    for n in [10, 100, 10000]:
        acts = rng.choice(2, size=n, p=p)
        est = np.mean([R[a] * grad_log_softmax(theta, a) for a in acts], axis=0)
        print(f"  sampled average of G·∇log π over {n:>5} plays = {np.round(est, 4)}")
    h = 1e-5                                  # check with "nudge θ and see how J moves" (school definition)
    num = [((softmax(theta + h * np.eye(2)[i]) @ R) - (softmax(theta - h * np.eye(2)[i]) @ R)) / (2 * h)
           for i in range(2)]
    print(f"  finite-difference check (J(θ+h) − J(θ−h)) / 2h       = {np.round(num, 4)}")


def part4(rng):
    print("\n" + "=" * 64)
    print("PART 4 · Baseline: same direction on average, far less noise")
    print("=" * 64)
    theta = np.array([0.0, 0.0])             # 50 / 50
    R = np.array([10.0, 12.0])               # BOTH actions pay well; Right is a bit better
    p = softmax(theta)
    b = p @ R                                # average return = 11
    print(f"  π = {p},  Left pays {R[0]:.0f}, Right pays {R[1]:.0f},  baseline b = average = {b:.0f}")
    for a in range(2):
        g = grad_log_softmax(theta, a)
        print(f"  play {'Left ' if a == 0 else 'Right'}: no baseline step ∝ {R[a]:>4.0f}·{g} = {R[a] * g}"
              f"   with baseline ∝ {R[a] - b:+.0f}·{g} = {(R[a] - b) * g}")
    acts = rng.choice(2, size=10000, p=p)
    for name, base in [("no baseline", 0.0), ("baseline 11", b)]:
        samples = np.array([(R[a] - base) * grad_log_softmax(theta, a)[1] for a in acts])
        print(f"  {name}: gradient for the Right logit, 10,000 samples: "
              f"mean {samples.mean():+.3f}, std {samples.std():.3f}")


def bandit(rng, steps, alpha, use_baseline, win_chance):
    """REINFORCE on a 3-armed bandit. Returns π after every play."""
    theta = np.zeros(3)                      # π starts at 1/3, 1/3, 1/3
    avg_reward = 0.0                         # baseline: running average of rewards so far
    history = [softmax(theta)]
    for t in range(1, steps + 1):
        p = softmax(theta)
        a = rng.choice(3, p=p)               # AGENT'S dice: which arm to pull
        r = float(rng.random() < win_chance[a])   # WORLD'S dice: does that arm pay 1 or 0?
        b = avg_reward if use_baseline else 0.0
        theta += alpha * (r - b) * grad_log_softmax(theta, a)
        avg_reward += (r - avg_reward) / t   # update the average AFTER using it
        history.append(softmax(theta))
    return np.array(history)


def part5(rng_seed):
    print("\n" + "=" * 64)
    print("PART 5 · 3-armed bandit: world's win chances vs agent's action probabilities")
    print("=" * 64)
    win_chance = np.array([0.2, 0.5, 0.8])   # THE WORLD'S RULES: fixed, never learned, never changed
    print(f"  World's win chances (given, fixed, hidden from the agent): {win_chance}")
    runs = {}
    for use_b in [False, True]:
        h = bandit(np.random.default_rng(rng_seed), 1000, 0.1, use_b, win_chance)
        runs[use_b] = h
        print(f"\n  {'WITH' if use_b else 'NO'} baseline, α = 0.1.  Agent's action probabilities π:")
        print("     play     arm A    arm B    arm C")
        for t in [0, 10, 25, 50, 100, 200, 500, 1000]:
            print(f"     {t:>4}   {h[t, 0]:7.3f}  {h[t, 1]:7.3f}  {h[t, 2]:7.3f}")
    print("\n  Over 200 different seeds, 300 plays each: final P(best arm C)")
    for use_b in [False, True]:
        finals = np.array([bandit(np.random.default_rng(s), 300, 0.1, use_b, win_chance)[-1, 2]
                           for s in range(200)])
        print(f"     {'with' if use_b else 'no  '} baseline: mean {finals.mean():.3f}, "
              f"worst seed {finals.min():.3f}, seeds where C < 0.5: {np.sum(finals < 0.5)}")
    return runs, win_chance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    part1()
    part2()
    part3(rng)
    part4(rng)
    runs, win_chance = part5(rng_seed=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    names = [f"arm A (wins {win_chance[0]:.0%})", f"arm B (wins {win_chance[1]:.0%})",
             f"arm C (wins {win_chance[2]:.0%})"]
    for ax, use_b in zip(axes, [False, True]):
        h = runs[use_b]
        for k in range(3):
            ax.plot(h[:, k], label=names[k], lw=2)
        ax.set_title(f"REINFORCE, {'WITH' if use_b else 'no'} baseline: agent's π(arm)")
        ax.set_xlabel("Play number")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Probability the agent picks this arm")
    axes[0].legend(title="world's win chance never changes")
    fig.tight_layout()
    fig.savefig("bandit_probs.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to bandit_probs.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
