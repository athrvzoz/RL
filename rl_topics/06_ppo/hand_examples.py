"""
PPO's maths with small numbers you can check by hand.

  1. the probability ratio r = π_new / π_old
  2. the clipped objective min(r·A, clip(r, 1−ε, 1+ε)·A) for every kind of case
  3. GAE on a 3-step rollout, for λ = 0, 0.95 and 1
  4. approx-KL for a small policy change

Run:  python hand_examples.py
"""
import sys

import numpy as np
import torch

from ppo import compute_gae

sys.stdout.reconfigure(encoding="utf-8")
EPS = 0.2

print("1 · THE RATIO")
p_old, p_new = 0.50, 0.60
print(f"  π_old(a|s) = {p_old}, π_new(a|s) = {p_new}  →  r = {p_new}/{p_old} = {p_new / p_old:.2f}")

print("\n2 · THE CLIPPED OBJECTIVE  (ε = 0.2, so r is clipped to [0.8, 1.2])")
print(f"  {'A':>5} {'r':>5} {'r·A':>7} {'clip(r)·A':>10} {'min':>7}   what happens")
cases = [(+2, 1.10, "A>0, small change: normal push up"),
         (+2, 1.50, "A>0, already +50%: clipped → no more push"),
         (+2, 0.70, "A>0, but prob went DOWN: unclipped, fix it"),
         (-2, 0.90, "A<0, small change: normal push down"),
         (-2, 0.50, "A<0, already −50%: clipped → no more push"),
         (-2, 1.40, "A<0, but prob went UP: unclipped, fix it")]
for A, r, note in cases:
    un, cl = r * A, np.clip(r, 1 - EPS, 1 + EPS) * A
    print(f"  {A:>+5} {r:>5.2f} {un:>+7.2f} {cl:>+10.2f} {min(un, cl):>+7.2f}   {note}")

print("\n3 · GAE on a 3-step rollout (γ = 0.9), the game continues afterwards")
rewards = torch.tensor([1.0, 0.0, 2.0])
values = torch.tensor([3.0, 2.0, 4.0])
last_value = torch.tensor(5.0)
dones = torch.zeros(3)
for t in range(3):
    nv = values[t + 1] if t < 2 else last_value
    print(f"  δ{t} = r{t} + 0.9·V(s{t + 1}) − V(s{t}) = {rewards[t]:.0f} + 0.9×{nv:.0f} − {values[t]:.0f} "
          f"= {rewards[t] + 0.9 * nv - values[t]:+.2f}")
for lam in [0.0, 0.95, 1.0]:
    adv, ret = compute_gae(rewards, values, dones, last_value, 0.9, lam)
    print(f"  λ = {lam:<4}: A = {np.round(adv.numpy(), 3)}   critic targets = {np.round(ret.numpy(), 3)}")

print("\n4 · APPROX-KL: how far did the policy move?")
old = torch.tensor([0.5, 0.5])
for new in [torch.tensor([0.55, 0.45]), torch.tensor([0.7, 0.3]), torch.tensor([0.95, 0.05])]:
    kl = float((old * (old / new).log()).sum())
    print(f"  π_old = {old.tolist()} → π_new = {[round(x, 2) for x in new.tolist()]}   KL = {kl:.4f}")
