"""
Headline experiment: A2C learns CartPole-v1.

    CartPole: keep a pole balanced on a moving cart.
    State  = 4 numbers: cart position, cart speed, pole angle, pole spinning speed
    Action = 0 (push cart left) or 1 (push cart right)
    Reward = +1 for every step the pole stays up.  Max 500 steps (then "truncated").

What this script shows:
    Part 1  training progress (8 games side by side, learn every 5 steps)
    Part 2  the exam: 20 games with the trained actor, no learning
    Part 3  inside the networks: the actor's probabilities and the critic's V(s)
            along one game, for an EARLY (20k steps) and the FINAL actor-critic
    Chart   cartpole_a2c.png: learning curve, entropy, V(s) along a game

Run:  python cartpole_a2c.py            (about 1 minute)
      python cartpole_a2c.py --no-show
"""
import argparse
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from a2c import ActorCritic, evaluate, moving_average, to_t, train_a2c

sys.stdout.reconfigure(encoding="utf-8")
GAMMA = 0.99


@torch.no_grad()
def watch_one_game(model, seed, greedy):
    """Play one game; record pole angle, pi(right), V(s) at every step."""
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    rows, done = [], False
    while not done:
        logits, v = model(to_t(obs))
        pi = torch.softmax(logits, -1)
        a = int(torch.argmax(pi)) if greedy else int(torch.distributions.Categorical(pi).sample())
        rows.append((obs[2], pi[1].item(), v.item()))
        obs, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
    env.close()
    angle, p_right, V = map(np.array, zip(*rows))
    # What REALLY happened from each step: discounted sum of the remaining +1 rewards.
    n = len(V)
    real = np.array([(1 - GAMMA ** (n - t)) / (1 - GAMMA) for t in range(n)])
    return angle, p_right, V, real, terminated


@torch.no_grad()
def critic_vs_spin(model, spins):
    """Ask the critic: cart in the middle, pole upright but SPINNING at speed `spin`. How good?"""
    states = np.zeros((len(spins), 4))
    states[:, 3] = spins
    return model.value(to_t(states)).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--steps", type=int, default=200_000)
    args = parser.parse_args()

    print("=" * 66)
    print("PART 1 · Training A2C on CartPole-v1: 8 games at once, n_steps = 5")
    print("=" * 66)
    model, hist = train_a2c("CartPole-v1", total_steps=args.steps, n_envs=8, n_steps=5,
                            gamma=GAMMA, lr=7e-4, ent_coef=0.01, seed=0,
                            checkpoints=[20_000], log_every=500)
    ret, ends = np.array(hist["ep_return"]), np.array(hist["ep_end_step"])
    first_500 = ends[np.argmax(ret >= 500)] if (ret >= 500).any() else None
    print(f"\n  Games finished during training: {len(ret)}")
    print(f"  First game that reached the 500-step limit: at step {first_500:,}")
    for lo, hi in [(0, 20_000), (20_000, 50_000), (50_000, 100_000), (100_000, 200_000)]:
        m = (ends > lo) & (ends <= hi)
        print(f"  steps {lo:>7,}–{hi:>7,}: avg return {ret[m].mean():6.1f} over {m.sum():>4} games")

    print("\n" + "=" * 66)
    print("PART 2 · The exam: 20 new games, no learning")
    print("=" * 66)
    greedy = evaluate(model, games=20, greedy=True)
    sampled = evaluate(model, games=20, greedy=False)
    print(f"  Greedy  (always the most likely action): avg {np.mean(greedy):.1f}, min {min(greedy):.0f}")
    print(f"  Sampled (roll the actor's dice)        : avg {np.mean(sampled):.1f}, min {min(sampled):.0f}")

    print("\n" + "=" * 66)
    print("PART 3 · Inside the networks along one game")
    print("=" * 66)
    early = ActorCritic(4, 2)
    early.load_state_dict(hist["snapshots"][20_000])
    games = {}
    for name, m, greedy_flag in [("early (20k steps)", early, False), ("final", model, True)]:
        games[name] = watch_one_game(m, seed=7, greedy=greedy_flag)
        angle, p_right, V, real, term = games[name]
        print(f"\n  {name}: game lasted {len(V)} steps ({'pole FELL' if term else 'time limit'})")
        print("    step   pole angle   pi(right)   critic V(s)   real discounted return")
        idx = sorted(set([0, 1, 2] + list(range(max(0, len(V) - 4), len(V)))))
        for t in idx:
            print(f"    {t:>4}   {angle[t]:+9.3f}   {p_right[t]:9.3f}   {V[t]:11.1f}   {real[t]:12.1f}")
    print("\n  Max possible discounted return with gamma 0.99: 1/(1-0.99) = 100")
    print("  (the time limit is NOT a real end, so a never-falling game is worth ~100)")

    spins = np.linspace(-3.5, 3.5, 71)
    V_early, V_final = critic_vs_spin(early, spins), critic_vs_spin(model, spins)
    print("\n  Critic's V(s): cart in the middle, pole upright but spinning (rad/s):")
    print("    spin speed   early V   final V")
    for t in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        i = int(np.argmin(np.abs(spins - t)))
        print(f"    {spins[i]:+9.1f}   {V_early[i]:7.1f}   {V_final[i]:7.1f}")
    torch.save(model.state_dict(), "cartpole_a2c.pt")

    # ------------------------------ chart ------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axs[0, 0]
    ax.plot(ends, ret, ".", ms=2, alpha=0.25, color="tab:blue", label="one game")
    ax.plot(ends[49:], moving_average(ret, 50), color="tab:blue", lw=2, label="average of last 50 games")
    ax.axhline(500, color="gray", ls="--", lw=1)
    ax.set(xlabel="Environment steps (all 8 games together)", ylabel="Game score (steps balanced)",
           title="A2C learning curve, CartPole-v1")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axs[0, 1]
    ax.plot(hist["update_step"], hist["entropy"], color="tab:purple", alpha=0.3, lw=0.8)
    ax.plot(hist["update_step"][99:], moving_average(hist["entropy"], 100), color="tab:purple", lw=2)
    ax.axhline(np.log(2), color="gray", ls="--", lw=1)
    ax.text(hist["update_step"][-1], np.log(2) - 0.02, "ln 2 = 0.693 (50/50, fully unsure)",
            ha="right", va="top", fontsize=9, color="gray")
    ax.set(xlabel="Environment steps", ylabel="Entropy of the actor",
           title="Entropy: how unsure the actor is")
    ax.grid(alpha=0.3)

    ax = axs[1, 0]
    for name, color in [("early (20k steps)", "tab:orange"), ("final", "tab:green")]:
        angle, p_right, V, real, term = games[name]
        ax.plot(V, color=color, lw=2, label=f"critic V(s), {name}: {len(V)} steps")
    angle, p_right, V, real, term = games["early (20k steps)"]
    ax.plot(real, color="black", ls="--", lw=1, label="real discounted return, early game")
    ax.set(xlabel="Step in the game", ylabel="Value",
           title="Critic's V(s) along one game (seed 7)")
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=9)

    ax = axs[1, 1]
    ax.plot(spins, V_early, color="tab:orange", lw=2, label="early critic (20k steps)")
    ax.plot(spins, V_final, color="tab:green", lw=2, label="final critic (200k steps)")
    ax.set(xlabel="Pole spinning speed (rad/s); cart in the middle, pole upright",
           ylabel="Critic's V(s)", title="What the critic learned: a fast-falling pole is bad news")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", fontsize=9)
    fig.tight_layout()
    fig.savefig("cartpole_a2c.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to cartpole_a2c.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
