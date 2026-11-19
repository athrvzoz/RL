"""
Plain REINFORCE (Monte Carlo policy gradient), used here only as the "before" picture.

Only ONE network: the actor pi(a|s). No critic.
    1. Play a WHOLE game with the current actor.
    2. For every step t, compute G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
       (the real total reward from that step to the end).
    3. Push up log pi(a_t|s_t) in proportion to G_t:   theta <- theta + alpha * G_t * grad log pi

Its two problems, which actor-critic fixes:
    - it must WAIT for the game to end before it can learn anything;
    - G_t is very noisy (one lucky or unlucky game changes it a lot) -> high variance.

This file only defines train_reinforce(); compare_reinforce.py runs it.
"""
import time

import gymnasium as gym
import numpy as np
import torch

from a2c import mlp, to_t

torch.set_num_threads(2)


def train_reinforce(env_id="CartPole-v1", total_steps=200_000, gamma=0.99, lr=1e-3,
                    seed=0, log_every=0):
    torch.manual_seed(seed)
    env = gym.make(env_id)
    actor = mlp(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    hist = {"ep_end_step": [], "ep_return": [], "weight_std": []}
    step, ep, start = 0, 0, time.time()
    obs, _ = env.reset(seed=seed)

    while step < total_steps:
        # ---------- 1. play ONE WHOLE game ----------
        states, actions, rewards, done = [], [], [], False
        while not done:
            with torch.no_grad():                              # just playing, no learning yet
                a = int(torch.distributions.Categorical(logits=actor(to_t(obs))).sample())
            states.append(obs)
            actions.append(a)
            obs, r, terminated, truncated, _ = env.step(a)
            rewards.append(r)
            done = terminated or truncated
            step += 1
        obs, _ = env.reset()

        # ---------- 2. returns G_t, computed backwards from the end ----------
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns = to_t(returns[::-1])

        # ---------- 3. policy gradient step: weight each log-prob by G_t ----------
        dist = torch.distributions.Categorical(logits=actor(to_t(states)))   # whole game at once
        logps = dist.log_prob(torch.as_tensor(actions))
        loss = -(logps * returns).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        ep += 1
        hist["ep_end_step"].append(step)
        hist["ep_return"].append(float(sum(rewards)))
        hist["weight_std"].append(float(returns.std()) if len(returns) > 1 else 0.0)
        if log_every and ep % log_every == 0:
            print(f"   step {step:>7,}  game {ep:>5}  avg return (last 20) "
                  f"{np.mean(hist['ep_return'][-20:]):7.1f}  ({time.time() - start:.0f}s)")
    env.close()
    return actor, hist
