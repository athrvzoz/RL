"""
A fairer exam: test the three SAVED agents on 100 fresh games each (training only
used 5 test games, which can be easy or hard by luck of the starting position).

Run:  python eval100.py            (about 1-2 minutes, no training)
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys

import gymnasium as gym
import numpy as np
import torch

from common import evaluate
from ddpg_td3 import DDPG
from sac import SAC

sys.stdout.reconfigure(encoding="utf-8")

agents = {"DDPG": DDPG(), "TD3": DDPG(td3=True), "SAC": SAC()}
for name, agent in agents.items():
    agent.actor.load_state_dict(torch.load(f"models/{name.lower()}_actor.pt"))
    agent.critic1.load_state_dict(torch.load(f"models/{name.lower()}_critic.pt"))

env = gym.make("Pendulum-v1")
rng = np.random.default_rng(0)
rand = []
for g in range(100):
    env.reset(seed=50_000 + g)
    total, done = 0.0, False
    while not done:
        _, r, term, trunc, _ = env.step(rng.uniform(-2, 2, size=1).astype(np.float32))
        total += r
        done = term or trunc
    rand.append(total)
print(f"  random torques   avg over 100 games {np.mean(rand):8.1f}")
for name, agent in agents.items():
    avg, _, _ = evaluate(agent, games=100, seed=50_000)
    print(f"  {name:<6}           avg over 100 games {avg:8.1f}")
print("  (0 would be perfect: upright from the first step with no torque. Starting hanging\n"
      "   down costs about −350 to swing up, so the average depends on where games start.)")
