"""
Every small worked example in the booklet, computed by the computer so you can check
your hand calculations. No training here; it runs in a couple of seconds.

  1. why "try every action" explodes                 (counting)
  2. soft / Polyak target update                     (multiplication)
  3. the DDPG critic's Bellman target                (same as Q-learning)
  4. the actor's chain rule  ∂Q/∂θ = ∂Q/∂a · ∂a/∂θ    (school chain rule, checked by PyTorch)
  5. over-estimation from "max of noisy guesses"     (and how TD3's min fixes it)
  6. TD3's target policy smoothing                   (clip, add, clip)
  7. SAC's reparameterisation trick + tanh squash    (checked by PyTorch)
  8. SAC's soft Bellman target                       (Q-learning + a curiosity bonus)
  9. entropy of a bell curve                         ("how spread out")

Run:  python hand_examples.py
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import math
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(2)


def title(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    title("1. Why you can't try every number")
    bins = len(np.arange(-2, 2.0001, 0.1))
    print(f"  Pendulum torque from -2 to +2 in steps of 0.1: {bins} choices (fine for DQN)")
    for joints in (2, 3, 7):
        print(f"  robot with {joints} joints, {bins} choices each: {bins} ** {joints} = {bins ** joints:,} actions")
    print("  ...and even 0.1 is too coarse for a motor: 0.73 and 0.74 are different torques.")

    title("2. Soft (Polyak) target update, tau = 0.005")
    online, target, tau = 0.80, 0.50, 0.005
    print(f"  online weight {online}, target weight {target}")
    print(f"  one step: (1 - {tau}) × {target} + {tau} × {online} = {(1 - tau) * target + tau * online:.4f}")
    for n in (1, 10, 100, 138, 500, 1000):
        print(f"    after {n:>4} steps: target = {online - (online - target) * (1 - tau) ** n:.4f}")
    print(f"  the gap halves every ln(0.5)/ln(0.995) = {math.log(0.5) / math.log(1 - tau):.0f} steps")
    tau2 = 0.1
    t = target
    print(f"  bigger tau = {tau2} (easier by hand):", end="")
    for _ in range(3):
        t = (1 - tau2) * t + tau2 * online
        print(f"  {t:.4f}", end="")
    print()

    title("3. DDPG critic target (exactly Q-learning's target)")
    r, gamma, q_next, q_now = -0.5, 0.99, -20.0, -18.0
    y = r + gamma * q_next
    print(f"  target = r + γ·Q_target(s2, μ_target(s2)) = {r} + {gamma} × ({q_next}) = {y:.2f}")
    print(f"  critic currently says Q(s, a) = {q_now}  ->  error {y - q_now:.2f}, squared {(y - q_now) ** 2:.2f}")

    title("4. The actor's chain rule (toy actor and toy critic)")
    # toy critic:  Q(s, a) = 5 - (a - 2s)^2    (a hill whose top is at a = 2s)
    # toy actor 1: a = θ·s
    s, theta, lr = 1.0, 0.5, 0.1
    for step in range(4):
        a = theta * s
        q = 5 - (a - 2 * s) ** 2
        dq_da = -2 * (a - 2 * s)
        da_dth = s
        grad = dq_da * da_dth
        print(f"  step {step}: θ={theta:.4f}  a={a:.4f}  Q={q:.4f}  ∂Q/∂a={dq_da:.4f}  ∂a/∂θ={da_dth}"
              f"  ∂Q/∂θ={grad:.4f}  ->  θ_new = θ + {lr}×{grad:.4f} = {theta + lr * grad:.4f}")
        theta += lr * grad
    th = torch.tensor(0.5, requires_grad=True)
    (5 - (th * 1.0 - 2.0) ** 2).backward()
    print(f"  PyTorch autograd agrees: ∂Q/∂θ at θ=0.5 is {th.grad.item():.4f}")
    # toy actor 2: a = 2·tanh(θ·s)   (the real actor's last layer)
    th = torch.tensor(0.5, requires_grad=True)
    a = 2 * torch.tanh(th * s)
    q = 5 - (a - 2 * s) ** 2
    q.backward()
    tnh = math.tanh(0.5)
    print(f"\n  with a tanh actor a = 2·tanh(θ·s), θ=0.5, s=1:")
    print(f"    tanh(0.5) = {tnh:.4f}, a = {2 * tnh:.4f}, Q = {q.item():.4f}")
    print(f"    ∂Q/∂a = -2(a - 2) = {-2 * (2 * tnh - 2):.4f}")
    print(f"    ∂a/∂θ = 2·(1 - tanh²)·s = 2 × {1 - tnh ** 2:.4f} × 1 = {2 * (1 - tnh ** 2):.4f}")
    print(f"    ∂Q/∂θ = {-2 * (2 * tnh - 2):.4f} × {2 * (1 - tnh ** 2):.4f} = "
          f"{-2 * (2 * tnh - 2) * 2 * (1 - tnh ** 2):.4f}   (autograd: {th.grad.item():.4f})")

    title("5. Over-estimation: the max of noisy guesses is too high")
    q1 = np.array([10.8, 9.5, 10.2, 9.1, 11.3])     # critic 1, five actions, TRUE value of each = 10
    q2 = np.array([9.6, 10.4, 10.9, 10.3, 9.8])     # critic 2, same actions, its own random errors
    k = int(np.argmax(q1))
    print(f"  true value of every action: 10.0")
    print(f"  critic 1 guesses {q1.tolist()}  ->  max = {q1.max()} (action {k + 1}), too high by {q1.max() - 10:.1f}")
    print(f"  critic 2 at that action: {q2[k]}   ->  min({q1[k]}, {q2[k]}) = {min(q1[k], q2[k])}")
    rng = np.random.default_rng(0)
    trials, n_actions = 100_000, 10
    e1 = rng.normal(0, 1, (trials, n_actions))
    e2 = rng.normal(0, 1, (trials, n_actions))
    pick = e1.argmax(1)
    single = e1.max(1).mean()
    twin = np.minimum(e1[np.arange(trials), pick], e2[np.arange(trials), pick]).mean()
    print(f"  simulation, {trials:,} trials, {n_actions} actions, true value 0, each guess off by N(0,1):")
    print(f"    one critic, max          : average error {single:+.3f}   (always too optimistic)")
    print(f"    TD3, min of twin critics : average error {twin:+.3f}")
    print(f"  bootstrapping feeds this error into the next target, so it adds up step after step;")
    print(f"  with γ = 0.99 one step of +{single:.2f} can grow toward {single:.2f}/(1-0.99) = {single / 0.01:.0f}")

    title("6. TD3 target policy smoothing")
    mu_t, noise_std, noise_clip, max_a = 1.6, 0.2 * 2, 0.5 * 2, 2.0
    for drawn in (0.3, -1.4):
        clipped = float(np.clip(drawn, -noise_clip, noise_clip))
        a2 = float(np.clip(mu_t + clipped, -max_a, max_a))
        print(f"  μ_target(s2) = {mu_t}, noise drawn {drawn:+.1f} -> clip to ±{noise_clip} -> {clipped:+.1f}"
              f" -> a2 = clip({mu_t} {clipped:+.1f}) = {a2:.1f}")
    print(f"  (noise std = 0.2 × 2 = {noise_std}, clip = 0.5 × 2 = {noise_clip}, torque limit ±{max_a})")

    title("7. SAC: reparameterisation + tanh squash")
    mean = torch.tensor(0.3, requires_grad=True)
    std = torch.tensor(0.5, requires_grad=True)
    eps = 0.8                                          # the dice roll, drawn from N(0, 1)
    u = mean + std * eps
    a = torch.tanh(u)
    torque = 2 * a
    torque.backward()
    log_gauss = -0.5 * eps ** 2 - math.log(0.5) - 0.5 * math.log(2 * math.pi)
    squash = math.log(1 - a.item() ** 2)
    print(f"  mean 0.3, σ 0.5, ε = {eps}:  u = 0.3 + 0.5 × 0.8 = {u.item():.3f}")
    print(f"  tanh({u.item():.3f}) = {a.item():.4f}  ->  torque = 2 × {a.item():.4f} = {torque.item():.4f}")
    print(f"  ∂torque/∂mean = 2·(1 - tanh²) = {mean.grad.item():.4f}    "
          f"∂torque/∂σ = 2·(1 - tanh²)·ε = {std.grad.item():.4f}")
    print(f"  log N(u; 0.3, 0.5) = -½ε² - ln σ - ½ln(2π) = -{0.5 * eps ** 2:.3f} + {-math.log(0.5):.4f}"
          f" - {0.5 * math.log(2 * math.pi):.4f} = {log_gauss:.4f}")
    print(f"  squash correction: - ln(1 - tanh²) = - ln({1 - a.item() ** 2:.4f}) = {-squash:+.4f}")
    print(f"  log π(a|s) = {log_gauss:.4f} {-squash:+.4f} = {log_gauss - squash:.4f}")

    title("8. SAC soft Bellman target")
    r, gamma, q1n, q2n, alpha, logp2 = -1.0, 0.99, -15.0, -15.6, 0.2, -0.09
    bonus = -alpha * logp2
    y = r + gamma * (min(q1n, q2n) + bonus)
    print(f"  r = {r}, Q1' = {q1n}, Q2' = {q2n}, α = {alpha}, log π(a2|s2) = {logp2}")
    print(f"  min = {min(q1n, q2n)}, bonus -α·log π = {bonus:.3f}")
    print(f"  target = {r} + {gamma} × ({min(q1n, q2n)} + {bonus:.3f}) = {y:.4f}")
    for lp in (1.5, -2.0):
        print(f"    if log π had been {lp:+.1f}: bonus {-alpha * lp:+.2f}, "
              f"target {r + gamma * (min(q1n, q2n) - alpha * lp):.4f}")

    title("9. Entropy of a bell curve: H = ½·ln(2πe·σ²)")
    for sd in (2.0, 1.0, 0.5, 0.1, 0.01):
        print(f"  σ = {sd:<5}  H = {0.5 * math.log(2 * math.pi * math.e * sd ** 2):+.3f}")


if __name__ == "__main__":
    main()
