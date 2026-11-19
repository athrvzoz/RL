"""
SAC, Soft Actor-Critic, for continuous actions (RL From Zero, continuous control).

Differences from DDPG / TD3:
  * The actor is STOCHASTIC: it outputs a bell curve (Gaussian) over torques, a mean and a
    spread σ. It samples a torque from it, then squashes it with tanh into (-2, 2).
  * "Reward for staying curious": every step's reward gets a bonus  -α·log π(a|s).
    Unlikely (surprising) actions have a very negative log π, so -α·log π is a big bonus.
    The policy is paid to keep its bell curve wide unless narrowing it clearly pays off.
  * α (the "temperature") is tuned automatically so the policy keeps about a target amount
    of randomness (target entropy = -1 for one action).
  * Twin critics with the min, like TD3. No target actor is needed.
  * Exploration comes from the sampling itself: no extra noise is added.

Run:  python sac.py                (about 2-3 minutes)
      python sac.py --alpha 0.2    (fixed temperature instead of automatic)
      python sac.py --no-show
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small memory footprint: other
os.environ.setdefault("OMP_NUM_THREADS", "2")        # programs share this computer
import argparse
import copy
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import mlp, soft_update, train

sys.stdout.reconfigure(encoding="utf-8")


class SAC:
    def __init__(self, obs_dim=3, act_dim=1, max_action=2.0, gamma=0.99, tau=0.005, lr=1e-3,
                 alpha=None, seed=0):
        """alpha=None -> learn α automatically; a number -> keep α fixed at that number."""
        torch.manual_seed(seed)
        self.max_action, self.gamma, self.tau = max_action, gamma, tau
        self.actor = mlp(obs_dim, 2 * act_dim)                 # outputs [mean, log σ]
        self.critic1 = mlp(obs_dim + act_dim, 1)
        self.critic2 = mlp(obs_dim + act_dim, 1)
        self.critic1_t = copy.deepcopy(self.critic1)
        self.critic2_t = copy.deepcopy(self.critic2)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters()) +
                                           list(self.critic2.parameters()), lr=lr)
        self.auto_alpha = alpha is None
        start = 1.0 if alpha is None else alpha                # automatic α starts at 1.0
        self.log_alpha = torch.tensor(math.log(start) if start > 0 else -math.inf, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(act_dim)                  # "keep about this much randomness"
        self.alpha_history = []

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def sample(self, s):
        """Reparameterisation trick: torque = squash(mean + σ·ε), with ε ~ N(0, 1) drawn OUTSIDE
        the network. Returns (torque, log π(torque | s), deterministic torque)."""
        mean, log_std = self.actor(s).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)                         # keep σ between 0.007 and 7.4
        std = log_std.exp()
        eps = torch.randn_like(mean)                           # the dice roll: no gradient needed
        u = mean + std * eps                                   # differentiable in mean and std
        a = torch.tanh(u)                                      # squash into (-1, 1)
        # log-probability of u under the bell curve, then the tanh "squash correction"
        log_prob = (-0.5 * eps ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        log_prob = log_prob - torch.log(1 - a ** 2 + 1e-6).sum(-1, keepdim=True)
        return self.max_action * a, log_prob, self.max_action * torch.tanh(mean)

    def act(self, s, explore=True):
        with torch.no_grad():
            a, _, a_mean = self.sample(torch.as_tensor(s, dtype=torch.float32))
        return (a if explore else a_mean).numpy().astype(np.float32)   # test = the bell's centre

    def q_estimate(self, s):
        with torch.no_grad():
            s = torch.as_tensor(s, dtype=torch.float32)
            a = self.sample(s)[2]
            return float(self.critic1(torch.cat([s, a])))

    def update(self, s, a, r, s2, done):
        alpha = self.alpha
        # ---------- 1. CRITICS: Bellman target with the curiosity bonus ----------
        with torch.no_grad():
            a2, logp2, _ = self.sample(s2)                     # a FRESH sample from the current actor
            q2 = torch.min(self.critic1_t(torch.cat([s2, a2], 1)),
                           self.critic2_t(torch.cat([s2, a2], 1)))
            target = r + self.gamma * (1 - done) * (q2 - alpha * logp2)
        sa = torch.cat([s, a], 1)
        critic_loss = F.mse_loss(self.critic1(sa), target) + F.mse_loss(self.critic2(sa), target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---------- 2. ACTOR: high Q AND high randomness ----------
        a_new, logp, _ = self.sample(s)
        q_new = torch.min(self.critic1(torch.cat([s, a_new], 1)), self.critic2(torch.cat([s, a_new], 1)))
        actor_loss = (alpha * logp - q_new).mean()             # maximise  Q - α·log π
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---------- 3. TEMPERATURE α: raise it if the policy got too sure of itself ----------
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        self.alpha_history.append(self.alpha)

        soft_update(self.critic1_t, self.critic1, self.tau)
        soft_update(self.critic2_t, self.critic2, self.tau)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--alpha", type=float, default=None, help="fixed α (default: automatic)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    print(f"SAC on Pendulum-v1, {args.steps:,} steps, α = {args.alpha or 'auto'}, seed {args.seed}")
    agent = SAC(alpha=args.alpha, seed=args.seed)
    log = train(agent, total_steps=args.steps, seed=args.seed)
    print(f"  final test score {log['eval'][-1]:.1f}   final α {agent.alpha:.3f}   ({log['seconds']:.0f} s)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(log["step"], log["eval"], marker="o")
    ax1.set_xlabel("Environment steps")
    ax1.set_ylabel("Test score (avg of 5 games)")
    ax1.grid(alpha=0.3)
    ax2.plot(agent.alpha_history)
    ax2.set_xlabel("Update")
    ax2.set_ylabel("temperature α")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)
    fig.savefig("sac_curve.png", dpi=110, bbox_inches="tight")
    print("Chart saved to sac_curve.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
