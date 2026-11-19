"""
Shared pieces for DDPG, TD3 and SAC (RL From Zero, continuous control).

Everything here is the same for all three algorithms:
  * mlp()           a small neural network = a "Q-table that can fill in gaps"
  * ReplayBuffer    a memory of past moves, sampled at random (same idea as DQN)
  * soft_update()   Polyak averaging: the target network creeps toward the real one
  * train()         the play-and-learn loop (one env.step, one learning step)
  * evaluate()      the "exam": play with NO exploration noise

Pendulum-v1 in one paragraph:
  state  = [cos θ, sin θ, θ_dot]   (θ = 0 means "pole straight up")
  action = ONE real number: the motor torque, anywhere in [-2, +2]
  reward = -(θ² + 0.1·θ_dot² + 0.001·torque²)   every step, so always <= 0
  a game is 200 steps; it never "terminates", it is only "truncated" by time.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small memory footprint: other
os.environ.setdefault("OMP_NUM_THREADS", "2")        # programs share this computer
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)          # other programs share this CPU


def mlp(n_in, n_out, hidden=128):
    # in -> 128 -> 128 -> out, with ReLU "bends" in between.
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_out))


def soft_update(target, online, tau):
    # target_weight <- (1 - tau) * target_weight + tau * online_weight
    # With tau = 0.005 the target moves only 0.5% of the way each step.
    with torch.no_grad():
        for t, o in zip(target.parameters(), online.parameters()):
            t.mul_(1 - tau).add_(tau * o)


class ReplayBuffer:
    """Stores (s, a, r, s2, done) rows. sample() returns a random mini-batch as tensors."""

    def __init__(self, obs_dim, act_dim, size=100_000, seed=0):
        self.s = np.zeros((size, obs_dim), dtype=np.float32)
        self.a = np.zeros((size, act_dim), dtype=np.float32)
        self.r = np.zeros((size, 1), dtype=np.float32)
        self.s2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.n, self.i, self.size = 0, 0, size
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done):
        self.s[self.i], self.a[self.i], self.r[self.i] = s, a, r
        self.s2[self.i], self.done[self.i] = s2, done
        self.i = (self.i + 1) % self.size          # when full, overwrite the oldest
        self.n = min(self.n + 1, self.size)

    def sample(self, batch=128):
        idx = self.rng.integers(0, self.n, size=batch)
        return tuple(torch.as_tensor(x[idx]) for x in (self.s, self.a, self.r, self.s2, self.done))


def evaluate(agent, games=5, seed=10_000, gamma=0.99):
    """Play `games` games with NO exploration. Returns (avg total reward, avg discounted return,
    avg critic estimate of the start state). The last two let us measure over-estimation."""
    env = gym.make("Pendulum-v1")
    totals, discounted, q_guess = [], [], []
    for g in range(games):
        s, _ = env.reset(seed=seed + g)
        q_guess.append(agent.q_estimate(s))
        total, disc, k, done = 0.0, 0.0, 0, False
        while not done:
            s, r, terminated, truncated, _ = env.step(agent.act(s, explore=False))
            total += r
            disc += gamma ** k * r
            k += 1
            done = terminated or truncated
        totals.append(total)
        discounted.append(disc)
    return float(np.mean(totals)), float(np.mean(discounted)), float(np.mean(q_guess))


def train(agent, total_steps=15_000, start_steps=1_000, batch=128, eval_every=1_000, seed=0,
          verbose=True):
    """The play-and-learn loop shared by DDPG, TD3 and SAC.

    Every step:  choose a torque -> env.step -> store the row -> learn from a random mini-batch.
    The first `start_steps` torques are purely random, to fill the memory with variety.
    """
    env = gym.make("Pendulum-v1")
    env.action_space.seed(seed)
    buf = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], seed=seed)
    log = {"step": [], "eval": [], "disc": [], "q": [], "ep_end": [], "ep_return": []}
    s, _ = env.reset(seed=seed)
    ep_return, t0 = 0.0, time.time()

    for t in range(1, total_steps + 1):
        if t <= start_steps:
            a = env.action_space.sample()               # warm-up: random torque in [-2, 2]
        else:
            a = agent.act(s, explore=True)              # actor's torque (+ noise for DDPG/TD3)
        s2, r, terminated, truncated, _ = env.step(a)
        # Store "terminated", NOT "truncated": the 200-step time limit is not a real end,
        # the pendulum would keep swinging, so the future still counts (see booklet ch. 5).
        buf.add(s, a, r, s2, float(terminated))
        s, ep_return = s2, ep_return + r
        if terminated or truncated:
            log["ep_end"].append(t)
            log["ep_return"].append(ep_return)
            s, _ = env.reset()
            ep_return = 0.0

        if t > start_steps:
            agent.update(*buf.sample(batch))

        if t % eval_every == 0:
            score, disc, q = evaluate(agent)
            for k, v in zip(("step", "eval", "disc", "q"), (t, score, disc, q)):
                log[k].append(v)
            if verbose:
                print(f"    step {t:>6,}   test score {score:8.1f}   critic thinks start is worth "
                      f"{q:7.1f}, really {disc:7.1f}   ({time.time() - t0:4.0f}s)")
    log["seconds"] = time.time() - t0
    return log


def policy_grid(agent, n=61):
    """The learned torque for every (angle, angular speed) pair: a 'map' of the policy."""
    thetas = np.linspace(-np.pi, np.pi, n)
    speeds = np.linspace(-8, 8, n)
    T, W = np.meshgrid(thetas, speeds)
    obs = np.stack([np.cos(T), np.sin(T), W], axis=-1).reshape(-1, 3).astype(np.float32)
    torque = agent.act(obs, explore=False)[:, 0].reshape(n, n)   # all 3,721 states at once
    return thetas, speeds, torque


def moving_average(x, window=5):
    return np.convolve(x, np.ones(window) / window, mode="valid")
