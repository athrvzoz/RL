"""
Look inside the TRAINED networks saved by pendulum_experiment.py (no training here).

  1. The critic's hill: for a few pendulum states, ask the DDPG critic Q(s, a) for every
     torque a from -2 to +2. Where does the actor's torque μ(s) sit on that hill?
  2. The chain rule with real numbers: ∂Q/∂a (slope of the hill) × ∂a/∂b (how the actor's
     last bias moves its torque) = ∂Q/∂b, checked against PyTorch autograd.
  3. SAC's bell curves: mean torque and spread σ in the same states.
  4. Draws sac_alpha.png and redraws pendulum_curves.png from results.json.

Run:  python peek_inside.py            (a few seconds)
      python peek_inside.py --no-show
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import policy_grid
from ddpg_td3 import DDPG
from pendulum_experiment import TORQUE_CMAP, swing_up_test
from sac import SAC

sys.stdout.reconfigure(encoding="utf-8")

# (name, angle in degrees with 0 = upright, angular speed)
STATES = [("hanging down, still", 180, 0.0), ("45° right of top, still", 45, 0.0),
          ("10° left of top, still", -10, 0.0), ("at top, spinning +3", 0, 3.0)]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]


def obs(deg, speed):
    th = math.radians(deg)
    return torch.tensor([math.cos(th), math.sin(th), speed], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    ddpg = DDPG()
    ddpg.actor.load_state_dict(torch.load("models/ddpg_actor.pt"))
    ddpg.critic1.load_state_dict(torch.load("models/ddpg_critic.pt"))
    sac = SAC()
    sac.actor.load_state_dict(torch.load("models/sac_actor.pt"))

    print("=" * 72)
    print("1. DDPG: the critic's hill and where the actor stands on it")
    print("=" * 72)
    torques = torch.linspace(-2, 2, 81).unsqueeze(1)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for (name, deg, speed), c in zip(STATES, COLORS):
        s = obs(deg, speed)
        with torch.no_grad():
            q_all = ddpg.critic1(torch.cat([s.repeat(81, 1), torques], 1)).squeeze(1)
        a = ddpg.mu(ddpg.actor, s).detach().requires_grad_(True)
        q = ddpg.critic1(torch.cat([s, a]))
        q.backward()
        best = torques[int(q_all.argmax())].item()
        print(f"  {name:26} μ(s) = {a.item():+.3f}   Q(s, μ(s)) = {q.item():7.2f}   "
              f"∂Q/∂a = {a.grad.item():+7.3f}   top of hill at a = {best:+.2f}")
        ax.plot(torques.squeeze(1), q_all, color=c, lw=2, label=name)
        ax.plot(a.item(), q.item(), "o", color=c, ms=9, mec="white", mew=2)
    ax.set_xlabel("torque a")
    ax.set_ylabel("critic's Q(s, a)")
    ax.set_title("Trained DDPG critic: Q for every torque (dot = the actor's choice μ(s))")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.savefig("critic_hill.png", dpi=120, bbox_inches="tight")

    print("\n" + "=" * 72)
    print("2. Chain rule with real numbers (state: 45° right of top, still)")
    print("=" * 72)
    s = obs(45, 0.0)
    last = ddpg.actor[-1]                                # the actor's last layer: z = w·h + b
    ddpg.actor.zero_grad()
    z = ddpg.actor(s)                                    # one number
    a = 2 * torch.tanh(z)
    a.retain_grad()
    q = ddpg.critic1(torch.cat([s, a]))
    q.backward()
    dq_da = a.grad.item()
    da_db = 2 * (1 - math.tanh(z.item()) ** 2)          # ∂a/∂z × ∂z/∂b, and ∂z/∂b = 1
    print(f"  z = {z.item():+.4f}   a = 2·tanh(z) = {a.item():+.4f}   Q = {q.item():.3f}")
    print(f"  ∂Q/∂a = {dq_da:+.4f}           (slope of the critic's hill at that torque)")
    print(f"  ∂a/∂b = 2·(1 - tanh²(z)) = {da_db:.4f}   (how the torque moves if the bias moves)")
    print(f"  ∂Q/∂b = {dq_da:+.4f} × {da_db:.4f} = {dq_da * da_db:+.4f}")
    print(f"  autograd says ∂Q/∂b = {last.bias.grad.item():+.4f}")
    print(f"  so gradient ASCENT nudges b by +lr × {dq_da * da_db:+.4f} "
          f"({'raising' if dq_da > 0 else 'lowering'} the torque)")

    print("\n" + "=" * 72)
    print("3. SAC: the bell curve in each state")
    print("=" * 72)
    for name, deg, speed in STATES:
        with torch.no_grad():
            mean, log_std = sac.actor(obs(deg, speed)).chunk(2)
        sd = math.exp(max(-5, min(2, log_std.item())))
        print(f"  {name:26} mean {mean.item():+7.3f} -> torque 2·tanh = {2 * math.tanh(mean.item()):+.3f}"
              f"    σ = {sd:.3f}")

    # ---------- redraw charts from results.json ----------
    with open("results.json") as f:
        res = json.load(f)
    alpha = res["SAC"]["alpha_history"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(np.arange(len(alpha)) * 50, alpha, color="#1baf7a", lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("SAC update number")
    ax.set_ylabel("temperature α (log scale)")
    ax.set_title("SAC tunes its own curiosity: α over training")
    ax.grid(alpha=0.3)
    fig.savefig("sac_alpha.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for name, c in zip(["DDPG", "TD3", "SAC"], ["#2a78d6", "#eb6834", "#1baf7a"]):
        ax.plot(res[name]["steps"], res[name]["eval"], lw=2, marker="o", ms=4, color=c, label=name)
    ax.axhline(res["random"], color="#888", ls="--", lw=1)
    ax.text(15_000, res["random"] + 40, f"random torques ({res['random']:.0f})", ha="right", color="#555")
    ax.axvline(1000, color="#bbb", lw=1)
    ax.text(1100, -1500, "learning\nstarts", color="#555", fontsize=9)
    ax.set_xlabel("Environment steps (200 steps = 1 game)")
    ax.set_ylabel("Test score: avg total reward, 5 games")
    ax.set_title("Pendulum-v1: learning curves (higher is better; 0 would be perfect)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig("pendulum_curves.png", dpi=120, bbox_inches="tight")
    print(f"\n  SAC α: start {alpha[0]:.3f}, after 2,000 updates {alpha[40]:.3f}, "
          f"after 7,000 {alpha[140]:.3f}, final {res['SAC']['final_alpha']:.3f}")
    # policy maps again, from the saved actors, with shorter labels
    agents = {"DDPG": ddpg, "TD3": DDPG(td3=True), "SAC": sac}
    agents["TD3"].actor.load_state_dict(torch.load("models/td3_actor.pt"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (name, agent) in zip(axes, agents.items()):
        th, sp, tq = policy_grid(agent)
        im = ax.pcolormesh(np.degrees(th), sp, tq, cmap=TORQUE_CMAP, vmin=-2, vmax=2, shading="auto")
        sw = swing_up_test(agent)
        ax.plot(np.degrees(sw[:, 0]), sw[:, 1], ".", color="#151B2F", ms=3)
        ax.plot(np.degrees(sw[0, 0]), sw[0, 1], "o", color="#151B2F", ms=7)
        ax.set_title(f"{name}: torque chosen")
        ax.set_xlabel("angle θ (°)   0 = upright, ±180 = down")
        ax.set_xticks([-180, -90, 0, 90, 180])
    axes[0].set_ylabel("angular speed θ_dot (rad/s)")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("torque (red = +2, blue = -2)")
    fig.savefig("pendulum_policy_maps.png", dpi=120, bbox_inches="tight")

    print("\nSaved critic_hill.png, sac_alpha.png, pendulum_curves.png, pendulum_policy_maps.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
