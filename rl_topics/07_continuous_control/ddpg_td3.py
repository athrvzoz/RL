"""
DDPG and TD3 for continuous actions (RL From Zero, continuous control).

DDPG = "DQN for real-number actions":
  * a CRITIC  Q(s, a) -> number   learns exactly like DQN: target = r + γ·Q_target(s2, a2)
  * an ACTOR  μ(s)    -> torque   replaces DQN's argmax. It is trained to push its torque
                                  "uphill" on the critic: gradient ascent on Q(s, μ(s)).
  * exploration = add a little random noise to the actor's torque.

TD3 = DDPG + three fixes. In this file they are the three `if self.td3:` blocks:
  1. twin critics, use the SMALLER of the two in the target  (fights over-estimation)
  2. delayed actor: update the actor only every 2nd critic update
  3. target policy smoothing: add clipped noise to the target action a2

Run:  python ddpg_td3.py               (DDPG then TD3, about 2-3 minutes)
      python ddpg_td3.py --algo td3 --steps 20000 --no-show
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small memory footprint: other
os.environ.setdefault("OMP_NUM_THREADS", "2")        # programs share this computer
import argparse
import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import mlp, soft_update, train

sys.stdout.reconfigure(encoding="utf-8")


class DDPG:
    def __init__(self, obs_dim=3, act_dim=1, max_action=2.0, gamma=0.99, tau=0.005,
                 lr=1e-3, noise=0.1, td3=False, policy_delay=2, target_noise=0.2,
                 noise_clip=0.5, seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.max_action, self.gamma, self.tau, self.noise = max_action, gamma, tau, noise
        self.td3, self.policy_delay = td3, policy_delay
        self.target_noise, self.noise_clip = target_noise, noise_clip

        self.actor = mlp(obs_dim, act_dim)                    # μ(s): state -> torque
        self.critic1 = mlp(obs_dim + act_dim, 1)              # Q(s, a): (state, torque) -> number
        self.critic2 = mlp(obs_dim + act_dim, 1)              # only used by TD3
        # Target networks: slow-moving copies, used ONLY to compute the Bellman target.
        self.actor_t = copy.deepcopy(self.actor)
        self.critic1_t = copy.deepcopy(self.critic1)
        self.critic2_t = copy.deepcopy(self.critic2)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        critic_params = list(self.critic1.parameters()) + (list(self.critic2.parameters()) if td3 else [])
        self.critic_opt = torch.optim.Adam(critic_params, lr=lr)
        self.updates = 0

    def mu(self, net, s):
        # tanh squeezes any number into (-1, 1); x2 makes it a legal torque in (-2, 2).
        return self.max_action * torch.tanh(net(s))

    def act(self, s, explore=True):
        with torch.no_grad():
            a = self.mu(self.actor, torch.as_tensor(s, dtype=torch.float32)).numpy()
        if explore:   # exploration: wobble the torque a little (Gaussian noise, σ = 0.1 × 2)
            a = a + self.rng.normal(0, self.noise * self.max_action, size=a.shape)
        return np.clip(a, -self.max_action, self.max_action).astype(np.float32)

    def q_estimate(self, s):
        # What the critic believes the state is worth if we follow the actor: Q(s, μ(s)).
        with torch.no_grad():
            s = torch.as_tensor(s, dtype=torch.float32)
            return float(self.critic1(torch.cat([s, self.mu(self.actor, s)])))

    def update(self, s, a, r, s2, done):
        self.updates += 1
        # ---------- 1. CRITIC: same as DQN / Q-learning ----------
        with torch.no_grad():
            a2 = self.mu(self.actor_t, s2)                         # target actor's torque in s2
            if self.td3:                                           # TD3 fix 3: smoothing
                eps = (torch.randn_like(a2) * self.target_noise * self.max_action)
                eps = eps.clamp(-self.noise_clip * self.max_action, self.noise_clip * self.max_action)
                a2 = (a2 + eps).clamp(-self.max_action, self.max_action)
            q2 = self.critic1_t(torch.cat([s2, a2], 1))
            if self.td3:                                           # TD3 fix 1: min of twins
                q2 = torch.min(q2, self.critic2_t(torch.cat([s2, a2], 1)))
            target = r + self.gamma * (1 - done) * q2              # Bellman: r + γ·Q(s2, a2)
        critic_loss = F.mse_loss(self.critic1(torch.cat([s, a], 1)), target)
        if self.td3:
            critic_loss = critic_loss + F.mse_loss(self.critic2(torch.cat([s, a], 1)), target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.td3 and self.updates % self.policy_delay != 0:     # TD3 fix 2: delayed actor
            return

        # ---------- 2. ACTOR: climb the critic's hill ----------
        # Maximise Q(s, μ(s))  <=>  minimise -Q(s, μ(s)). backward() applies the chain rule
        # ∂Q/∂θ = ∂Q/∂a · ∂a/∂θ for us. Only the actor's optimiser steps, so the critic is frozen.
        actor_loss = -self.critic1(torch.cat([s, self.mu(self.actor, s)], 1)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---------- 3. TARGETS: creep 0.5% toward the online networks ----------
        soft_update(self.actor_t, self.actor, self.tau)
        soft_update(self.critic1_t, self.critic1, self.tau)
        if self.td3:
            soft_update(self.critic2_t, self.critic2, self.tau)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ddpg", "td3", "both"], default="both")
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    algos = ["ddpg", "td3"] if args.algo == "both" else [args.algo]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name in algos:
        print(f"\n{name.upper()} on Pendulum-v1, {args.steps:,} steps, seed {args.seed}")
        agent = DDPG(td3=(name == "td3"), seed=args.seed)
        log = train(agent, total_steps=args.steps, seed=args.seed)
        print(f"  final test score {log['eval'][-1]:.1f}   ({log['seconds']:.0f} s)")
        ax.plot(log["step"], log["eval"], marker="o", label=name.upper())
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Test score (avg of 5 games)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig("ddpg_td3_curve.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to ddpg_td3_curve.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
