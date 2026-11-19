"""
The headline experiment: DDPG vs TD3 vs SAC on Pendulum-v1.

Pendulum: a pole on a motor. It starts hanging at a random angle. The motor is weak
(torque between -2 and +2), too weak to lift the pole straight up in one push, so the agent
must learn to SWING it back and forth to build up speed, then balance it at the top.
The action is ONE real number. A Q-table or DQN's argmax cannot handle that directly.

What this script does:
  1. trains DDPG, TD3 and SAC for 15,000 steps each (75 games of 200 steps), same seed
  2. prints a test score every 1,000 steps, plus "what the critic thinks the start is worth"
     next to "what it was really worth" (to see over-estimation)
  3. plays one test game from the hardest start (hanging straight down, not moving)
  4. draws the learned torque for every (angle, speed): a map of the policy

Run:  python pendulum_experiment.py              (about 5 minutes on CPU)
      python pendulum_experiment.py --steps 10000 --no-show
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small memory footprint: other
os.environ.setdefault("OMP_NUM_THREADS", "2")        # programs share this computer
import argparse
import json
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from common import policy_grid, train
from ddpg_td3 import DDPG
from sac import SAC

sys.stdout.reconfigure(encoding="utf-8")

COLORS = {"DDPG": "#2a78d6", "TD3": "#eb6834", "SAC": "#1baf7a"}
TORQUE_CMAP = LinearSegmentedColormap.from_list("torque", ["#2a78d6", "#f0efec", "#e34948"])


def angle(obs):
    # Turn [cos θ, sin θ] back into θ in (-π, π]. θ = 0 is straight UP, ±π is straight DOWN.
    return float(np.arctan2(obs[1], obs[0]))


def swing_up_test(agent, seed=7):
    """One test game from the hardest start: hanging straight down, not moving."""
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    env.unwrapped.state = np.array([np.pi, 0.0])          # θ = π (down), θ_dot = 0
    s = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    rows, done = [], False
    while not done:
        a = agent.act(s, explore=False)
        s2, r, terminated, truncated, _ = env.step(a)
        rows.append((angle(s), float(s[2]), float(a[0]), float(r)))
        s, done = s2, terminated or truncated
    return np.array(rows)        # columns: angle, speed, torque, reward


def random_baseline(games=20, seed=10_000):
    env = gym.make("Pendulum-v1")
    env.action_space.seed(0)
    totals = []
    for g in range(games):
        env.reset(seed=seed + g)
        total, done = 0.0, False
        while not done:
            _, r, terminated, truncated, _ = env.step(env.action_space.sample())
            total, done = total + r, terminated or truncated
        totals.append(total)
    return float(np.mean(totals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    rand = random_baseline()
    print(f"Random torques score {rand:.1f} per game (200 steps). Closer to 0 is better.\n")

    makers = {"DDPG": lambda: DDPG(seed=args.seed),
              "TD3": lambda: DDPG(td3=True, seed=args.seed),
              "SAC": lambda: SAC(seed=args.seed)}
    logs, maps, swings, results = {}, {}, {}, {"random": rand}
    os.makedirs("models", exist_ok=True)
    for name, make in makers.items():
        print("=" * 72)
        print(f"{name}: {args.steps:,} steps on Pendulum-v1 (seed {args.seed})")
        print("=" * 72)
        agent = make()
        log = train(agent, total_steps=args.steps, seed=args.seed)
        logs[name] = log
        maps[name] = policy_grid(agent)
        swings[name] = swing_up_test(agent)
        torch.save(agent.actor.state_dict(), f"models/{name.lower()}_actor.pt")
        torch.save(agent.critic1.state_dict(), f"models/{name.lower()}_critic.pt")
        sw = swings[name]
        up = np.flatnonzero(np.abs(sw[:, 0]) < 0.1)       # within ~6° of straight up
        first_up = int(up[0]) + 1 if len(up) else None
        results[name] = {"steps": log["step"], "eval": log["eval"], "disc": log["disc"], "q": log["q"],
                         "ep_end": log["ep_end"], "ep_return": log["ep_return"],
                         "seconds": log["seconds"], "swing_return": float(sw[:, 3].sum()),
                         "swing_first_up_step": first_up,
                         "swing_torques_first10": [round(float(x), 2) for x in sw[:10, 2]]}
        if name == "SAC":
            results[name]["alpha_history"] = agent.alpha_history[::50]
            results[name]["final_alpha"] = agent.alpha
        print(f"\n  final test score {log['eval'][-1]:.1f}   best {max(log['eval']):.1f}   "
              f"time {log['seconds']:.0f}s")
        print(f"  swing-up from hanging down: total reward {sw[:, 3].sum():.1f}, "
              f"first within 6° of upright at step {first_up}")
        print(f"  first 10 torques: {results[name]['swing_torques_first10']}\n")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=1)

    print("=" * 72)
    print("SUMMARY (test score = average total reward over 5 games, higher is better)")
    print("=" * 72)
    print(f"  {'':6}" + "".join(f"{s:>9,}" for s in logs["DDPG"]["step"][::2]))
    for name, log in logs.items():
        print(f"  {name:6}" + "".join(f"{v:9.0f}" for v in log["eval"][::2]))

    # ---------- chart 1: learning curves ----------
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for name, log in logs.items():
        ax.plot(log["step"], log["eval"], lw=2, marker="o", ms=4, color=COLORS[name], label=name)
    ax.axhline(rand, color="#888", ls="--", lw=1)
    ax.text(args.steps, rand + 40, f"random torques ({rand:.0f})", ha="right", color="#555")
    ax.axvline(1000, color="#bbb", lw=1)
    ax.text(1100, -1500, "learning\nstarts", color="#555", fontsize=9)
    ax.set_xlabel("Environment steps (200 steps = 1 game)")
    ax.set_ylabel("Test score: avg total reward, 5 games")
    ax.set_title("Pendulum-v1: learning curves (higher is better; 0 would be perfect)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig("pendulum_curves.png", dpi=120, bbox_inches="tight")

    # ---------- chart 2: policy maps ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, name in zip(axes, logs):
        th, sp, tq = maps[name]
        im = ax.pcolormesh(np.degrees(th), sp, tq, cmap=TORQUE_CMAP, vmin=-2, vmax=2, shading="auto")
        sw = swings[name]
        ax.plot(np.degrees(sw[:, 0]), sw[:, 1], ".", color="#151B2F", ms=3)
        ax.plot(np.degrees(sw[0, 0]), sw[0, 1], "o", color="#151B2F", ms=7)
        ax.set_title(f"{name}: torque chosen")
        ax.set_xlabel("angle θ in degrees (0 = upright, ±180 = hanging down)")
        ax.set_xticks([-180, -90, 0, 90, 180])
    axes[0].set_ylabel("angular speed θ_dot (rad/s)")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("torque (red = +2, blue = -2)")
    fig.savefig("pendulum_policy_maps.png", dpi=120, bbox_inches="tight")

    # ---------- chart 3: the swing-up game ----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    for name, sw in swings.items():
        ax1.plot(np.degrees(np.abs(sw[:, 0])), color=COLORS[name], lw=2, label=name)
        ax2.plot(sw[:, 2], color=COLORS[name], lw=1.6, label=name)
    ax1.set_ylabel("|angle| from upright (°)")
    ax1.set_yticks([0, 45, 90, 135, 180])
    ax1.set_title("One test game from hanging straight down (no noise)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("torque")
    ax2.set_xlabel("step")
    ax2.set_xlim(0, 120)
    ax2.grid(alpha=0.3)
    fig.savefig("pendulum_swing.png", dpi=120, bbox_inches="tight")

    # ---------- chart 4: what the critic believes vs reality ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, name in zip(axes, ["DDPG", "TD3"]):
        log = logs[name]
        ax.plot(log["step"], log["q"], lw=2, color=COLORS[name], label="critic's guess Q(s0, μ(s0))")
        ax.plot(log["step"], log["disc"], lw=2, ls="--", color="#151B2F", label="real discounted return")
        ax.set_title(name)
        ax.set_xlabel("Environment steps")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("value of the start state")
    fig.savefig("pendulum_critic_vs_real.png", dpi=120, bbox_inches="tight")

    print("\nSaved: pendulum_curves.png, pendulum_policy_maps.png, pendulum_swing.png,")
    print("       pendulum_critic_vs_real.png, results.json, models/")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
