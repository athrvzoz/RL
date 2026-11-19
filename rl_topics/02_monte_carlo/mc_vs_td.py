"""
Topic 2c: Monte Carlo vs TD(0), on examples small enough to check by hand.

TD(0) is the "learn after every step from a guess" idea that Q-learning uses:
    TD(0):  V[s] += α · ( r + γ·V[s2]  −  V[s] )          target uses a GUESS (V[s2])
    MC   :  V[s] += α · ( G            −  V[s] )          target is the REAL return G

Part 1: the SAME game, updated by MC and by TD(0), with every number printed.
Part 2: first-visit vs every-visit on a game where a state repeats.
Part 3: the classic 5-state random walk. Who learns faster, and who is noisier?

Run:  python mc_vs_td.py            (about 20 seconds)
      python mc_vs_td.py --no-show
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def part1_same_game(gamma=0.9, alpha=0.5):
    # One game: A -> B -> C -> end.  Rewards on leaving each state: 0, -1, +10.
    game = [("A", 0.0), ("B", -1.0), ("C", 10.0)]
    start = {"A": 2.0, "B": 4.0, "C": 6.0}                     # current guesses before the game
    print(f"  Game:  A --(r=0)--> B --(r=-1)--> C --(r=+10)--> END      γ = {gamma}, α = {alpha}")
    print(f"  Guesses before the game: V(A) = {start['A']}, V(B) = {start['B']}, V(C) = {start['C']}\n")

    print("  MONTE CARLO: play to the END first, then walk BACKWARDS")
    V = dict(start)
    G = 0.0
    for s, r in reversed(game):
        G = r + gamma * G
        old = V[s]
        V[s] = old + alpha * (G - old)
        print(f"    {s}:  G = {r:+.0f} + {gamma} × (G after) = {G:5.2f}    "
              f"V({s}) = {old:.2f} + {alpha} × ({G:.2f} − {old:.2f}) = {V[s]:.2f}")
    mc = V

    print("\n  TD(0): update DURING the game, after every step, using the next state's GUESS")
    V = dict(start)
    for t, (s, r) in enumerate(game):
        last = t == len(game) - 1
        nxt = None if last else game[t + 1][0]
        target = r + (0.0 if last else gamma * V[nxt])
        old = V[s]
        V[s] = old + alpha * (target - old)
        how = "END: no future" if last else f"{r:+.0f} + {gamma} × V({nxt})={V[nxt]:.2f}"
        print(f"    step {t}, {s}:  target = {how:<24} = {target:5.2f}    "
              f"V({s}) = {old:.2f} + {alpha} × ({target:.2f} − {old:.2f}) = {V[s]:.2f}")
    print(f"\n  After ONE game:   MC  V(A)={mc['A']:.2f}  V(B)={mc['B']:.2f}  V(C)={mc['C']:.2f}")
    print(f"                    TD  V(A)={V['A']:.2f}  V(B)={V['B']:.2f}  V(C)={V['C']:.2f}")


def part2_first_vs_every():
    game = [("A", 1.0), ("B", 0.0), ("A", 2.0)]                 # A is visited twice, γ = 1
    G, returns = 0.0, []
    for s, r in reversed(game):
        G = r + G
        returns.append((s, G))
    returns.reverse()
    print("  Game: A --(r=1)--> B --(r=0)--> A --(r=2)--> END     (γ = 1)")
    for t, (s, G) in enumerate(returns):
        print(f"    step {t}: state {s}, return from here G = {G:.0f}")
    a_returns = [G for s, G in returns if s == "A"]
    print(f"  First-visit uses only the first A:  sample for V(A) = {a_returns[0]:.0f}")
    print(f"  Every-visit uses both A's:          samples {a_returns[0]:.0f} and {a_returns[1]:.0f}, "
          f"average = {np.mean(a_returns):.1f}")


TRUE_V = np.arange(1, 6) / 6                                     # 1/6, 2/6, ..., 5/6


def random_walk_episode(rng):
    """States 0..4 (A..E), start at C (=2). Step left/right 50/50.
    Fall off the left end: reward 0. Fall off the right end: reward +1."""
    s, steps = 2, []
    while True:
        s2 = s + (1 if rng.random() < 0.5 else -1)
        r = 1.0 if s2 == 5 else 0.0
        steps.append((s, r, s2))
        if s2 in (-1, 5):
            return steps
        s = s2


def run_random_walk(method, alpha, episodes=100, runs=100, seed=0):
    rng = np.random.default_rng(seed)
    errors = np.zeros((runs, episodes + 1))
    for run in range(runs):
        V = np.full(5, 0.5)                                      # start every guess at 0.5
        errors[run, 0] = np.sqrt(np.mean((V - TRUE_V) ** 2))
        for ep in range(episodes):
            game = random_walk_episode(rng)
            if method == "TD":
                for s, r, s2 in game:                            # update every step, from a guess
                    future = V[s2] if 0 <= s2 <= 4 else 0.0
                    V[s] += alpha * (r + future - V[s])
            else:
                G = 0.0
                for s, r, s2 in reversed(game):                  # wait for the end, real return
                    G = r + G
                    V[s] += alpha * (G - V[s])
            errors[run, ep + 1] = np.sqrt(np.mean((V - TRUE_V) ** 2))
    return errors.mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("PART 1 · The same game, updated by Monte Carlo and by TD(0)")
    print("=" * 72)
    part1_same_game()

    print("\n" + "=" * 72)
    print("PART 2 · First-visit vs every-visit")
    print("=" * 72)
    part2_first_vs_every()

    print("\n" + "=" * 72)
    print("PART 3 · Random walk  A B [C] D E : true values 1/6 .. 5/6")
    print("=" * 72)
    rng = np.random.default_rng(1)
    targets = {"MC  (real final reward)": [], "TD  (r + V(next)), V = true values": [],
               "TD  (r + V(next)), V = all 0 (bad guesses)": []}
    for _ in range(10_000):                     # how noisy / how wrong is each TARGET for state C?
        game = random_walk_episode(rng)
        s, r, s2 = game[0]                      # the first step always leaves C
        targets["MC  (real final reward)"].append(game[-1][1])          # γ = 1: G = final reward
        targets["TD  (r + V(next)), V = true values"].append(r + TRUE_V[s2])
        targets["TD  (r + V(next)), V = all 0 (bad guesses)"].append(r + 0.0)
    print("  The TARGET each method uses for state C, over 10,000 games (true value 0.500):")
    for name, t in targets.items():
        seen = ", ".join(f"{v:.3f}" for v in sorted(set(t)))
        print(f"    {name:<44} mean {np.mean(t):.3f}   spread (std) {np.std(t):.3f}   values: {seen}")

    curves = {}
    for method, alphas in [("MC", (0.01, 0.02, 0.04)), ("TD", (0.05, 0.1, 0.15))]:
        for a in alphas:
            curves[(method, a)] = run_random_walk(method, a)
    print("\n  Average error (RMS, 100 runs) after N games:")
    print(f"    {'method':<12}" + "".join(f"{n:>9}" for n in (0, 10, 25, 50, 100)))
    for (method, a), c in curves.items():
        print(f"    {method} α={a:<6}" + "".join(f"{c[n]:9.3f}" for n in (0, 10, 25, 50, 100)))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    styles = {0: "-", 1: "--", 2: ":"}
    for (method, a), c in curves.items():
        alphas = (0.01, 0.02, 0.04) if method == "MC" else (0.05, 0.1, 0.15)
        ax.plot(c, color="#eb6834" if method == "MC" else "#2a78d6", ls=styles[alphas.index(a)], lw=2,
                label=f"{method}, α = {a}")
    ax.set_xlabel("Games played")
    ax.set_ylabel("Error of V (RMS, average of 100 runs)")
    ax.set_title("Random walk: Monte Carlo (orange) vs TD(0) (blue)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig("mc_vs_td_randomwalk.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to mc_vs_td_randomwalk.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
