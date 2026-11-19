"""
TD(0) prediction: learning V(s) from experience, on the "random walk".

          0      +----+----+----+----+----+      +1
       [LEFT] <- | A  | B  | C  | D  | E  | -> [RIGHT]
                 +----+----+----+----+----+
                             start

Every game starts in C. Each step the walker goes left or right, 50/50
(a coin flip; WE do not choose anything here). Falling off the right end
gives +1, the left end gives 0. All other steps give 0. gamma = 1.

PREDICTION means: the policy is fixed (coin flips), we only want to learn
"how good is each square?" = V(s) = the chance of finishing on the right.
The true answers are known exactly: A 1/6, B 2/6, C 3/6, D 4/6, E 5/6.
That lets us measure how wrong each method still is.

Part 1: watch TD(0) and Monte Carlo update on the SAME first game, by hand.
Part 2: TD(0) vs Monte Carlo over 100 games (average of 100 runs).
Part 3: n-step TD on a longer, 19-square walk: the dial between TD and MC.

Run:  python td_prediction.py            (about 1 minute)
      python td_prediction.py --no-show
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

NAMES = "LABCDER"                                     # 0 = left end, 1..5 = A..E, 6 = right end


def play_walk(rng, n_squares=5, left_reward=0.0):
    """One game. Returns the states visited [start, ..., end] and the reward of each step."""
    s = (n_squares + 1) // 2                          # start in the middle square
    states, rewards = [s], []
    while 0 < s < n_squares + 1:
        s += rng.choice([-1, 1])                      # coin flip: left or right
        states.append(s)
        rewards.append(1.0 if s == n_squares + 1 else (left_reward if s == 0 else 0.0))
    return states, rewards


def td0_update(V, states, rewards, alpha, gamma=1.0, log=False):
    """TD(0): after EVERY step, nudge V of the state we left toward r + gamma * V(s2)."""
    for t in range(len(rewards)):
        s, r, s2 = states[t], rewards[t], states[t + 1]
        target = r + gamma * V[s2]                    # V of an end square is always 0
        delta = target - V[s]                         # TD error: the surprise
        old = V[s]
        V[s] += alpha * delta
        if log:
            print(f"    step {t + 1}: {NAMES[s]} → {NAMES[s2]}, r = {r:.0f}   target = {r:.0f} + {V[s2]:.3f} "
                  f"= {target:.3f}   δ = {delta:+.3f}   V({NAMES[s]}) {old:.3f} → {V[s]:.3f}")


def mc_update(V, states, rewards, alpha, gamma=1.0, log=False):
    """Monte Carlo: wait for the END, then nudge every visited state toward the real total G."""
    G = 0.0
    for t in reversed(range(len(rewards))):           # walk backwards to add up the future
        G = rewards[t] + gamma * G
        s = states[t]
        old = V[s]
        V[s] += alpha * (G - V[s])
        if log:
            print(f"    {NAMES[s]} (step {t + 1}): G = {G:.0f}   V({NAMES[s]}) {old:.3f} → {V[s]:.3f}")


def nstep_update(V, states, rewards, alpha, n, gamma=1.0):
    """n-step TD: use n real rewards, THEN bootstrap from V. n=1 is TD(0); huge n is Monte Carlo."""
    T = len(rewards)
    for tau in range(T):
        end = min(tau + n, T)
        G = sum(gamma ** (i - tau) * rewards[i] for i in range(tau, end))   # n real rewards
        if end < T:
            G += gamma ** n * V[states[end]]          # ...plus a guess for the rest
        V[states[tau]] += alpha * (G - V[states[tau]])


def rms_error(V, true_v):
    return float(np.sqrt(np.mean((V[1:-1] - true_v) ** 2)))


def part1():
    print("=" * 70)
    print("PART 1 · The first game, by hand (α = 0.1, every V starts at 0.5)")
    print("=" * 70)
    rng = np.random.default_rng(3)
    states, rewards = play_walk(rng)
    print("  The game: " + " → ".join(NAMES[s] for s in states) + f"   (reward at the end: {rewards[-1]:.0f})")
    V = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
    print("\n  TD(0): update after every step")
    td0_update(V, states, rewards, alpha=0.1, log=True)
    print("  V after the game: " + "  ".join(f"{NAMES[s]} {V[s]:.3f}" for s in range(1, 6)))
    V = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
    print("\n  Monte Carlo: wait for the end, then update every step with the real total G")
    mc_update(V, states, rewards, alpha=0.1, log=True)
    print("  V after the game: " + "  ".join(f"{NAMES[s]} {V[s]:.3f}" for s in range(1, 6)))


def part2(runs=100, games=100):
    print("\n" + "=" * 70)
    print(f"PART 2 · TD(0) vs Monte Carlo, {games} games, average of {runs} runs")
    print("=" * 70)
    true_v = np.arange(1, 6) / 6
    curves = {}
    for method, alphas in [("TD(0)", [0.05, 0.1, 0.15]), ("MC", [0.01, 0.02, 0.04])]:
        for alpha in alphas:
            errors = np.zeros(games + 1)
            for run in range(runs):
                rng = np.random.default_rng(run)
                V = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
                errors[0] += rms_error(V, true_v)
                for g in range(games):
                    states, rewards = play_walk(rng)
                    (td0_update if method == "TD(0)" else mc_update)(V, states, rewards, alpha)
                    errors[g + 1] += rms_error(V, true_v)
            curves[(method, alpha)] = errors / runs
            e = curves[(method, alpha)]
            print(f"  {method:<6} α = {alpha:<5} RMS error after 10 games {e[10]:.3f}, "
                  f"after 50 {e[50]:.3f}, after 100 {e[100]:.3f}")
    # one long TD run: what V looks like
    rng = np.random.default_rng(0)
    V = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
    snapshots = {0: V[1:6].copy()}
    for g in range(1, 101):
        td0_update(V, *play_walk(rng), alpha=0.1)
        if g in (1, 10, 100):
            snapshots[g] = V[1:6].copy()
    print("\n  One TD(0) run (α = 0.1), the values after N games:")
    for g, v in snapshots.items():
        print(f"    after {g:>3} games: " + "  ".join(f"{NAMES[i + 1]} {x:.2f}" for i, x in enumerate(v)))
    print("    true values:     " + "  ".join(f"{NAMES[i + 1]} {x:.2f}" for i, x in enumerate(true_v)))
    return curves, snapshots, true_v


def part3(runs=50, games=10):
    print("\n" + "=" * 70)
    print(f"PART 3 · n-step TD on a 19-square walk (left end −1, right end +1), "
          f"first {games} games, {runs} runs")
    print("=" * 70)
    true_v = np.arange(-18, 20, 2) / 20               # -0.9, -0.8, ..., 0.9
    alphas = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    ns = [1, 2, 4, 8, 16, 1000]                       # 1000 = longer than any game = Monte Carlo
    table = {}
    for n in ns:
        row = []
        for alpha in alphas:
            total = 0.0
            for run in range(runs):
                rng = np.random.default_rng(run)
                V = np.zeros(21)
                for g in range(games):
                    states, rewards = play_walk(rng, n_squares=19, left_reward=-1.0)
                    nstep_update(V, states, rewards, alpha, n)
                    total += rms_error(V, true_v)
            row.append(total / (runs * games))
        table[n] = np.array(row)
        best = int(np.argmin(row))
        label = "MC" if n == 1000 else f"n={n}"
        print(f"  {label:<5} best α = {alphas[best]:.2f}, average RMS error = {row[best]:.3f}")
    return table, alphas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")                     # just save the PNGs, no window

    part1()
    curves, snapshots, true_v = part2()
    table, alphas = part3()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    x = np.arange(5)
    for g, v in snapshots.items():
        ax.plot(x, v, marker="o", lw=2, label=f"after {g} games")
    ax.plot(x, true_v, "k--", lw=1.5, label="true value")
    ax.set_xticks(x, list("ABCDE"))
    ax.set_ylabel("V(s)")
    ax.set_title("TD(0) learning V on the 5-square walk (α = 0.1)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    for (method, alpha), e in curves.items():
        ax.plot(e, lw=2, ls="-" if method == "TD(0)" else "--",
                color={0.05: "#9AB0F0", 0.1: "#2C4BD1", 0.15: "#15257A",
                       0.01: "#F0C27A", 0.02: "#C47A00", 0.04: "#6E4400"}[alpha],
                label=f"{method}, α={alpha}")
    ax.set_xlabel("Games played")
    ax.set_ylabel("RMS error (average of 100 runs)")
    ax.set_title("TD(0) (solid) vs Monte Carlo (dashed)")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("td_vs_mc.png", dpi=110, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for n, row in table.items():
        ax.plot(alphas, row, marker="o", lw=2, label="Monte Carlo" if n == 1000 else f"n = {n}")
    ax.set_ylim(0.2, 0.6)
    ax.set_xlabel("α (learning rate)")
    ax.set_ylabel("Average RMS error, first 10 games")
    ax.set_title("n-step TD on the 19-square walk: a middle n wins")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("nstep.png", dpi=110, bbox_inches="tight")
    print("\nCharts saved to td_vs_mc.png and nstep.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
