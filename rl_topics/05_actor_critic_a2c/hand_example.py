"""
The booklet's worked examples, done by the computer so you can check your hand working.

A tiny world with ONE state s and TWO actions (left, right).
    Actor  : two "preference" numbers h = [h_left, h_right]  ->  softmax  ->  probabilities
    Critic : one number V(s) (a table entry, exactly like project 2's V)

Each example does ONE actor-critic update by the formulas, then repeats it with
PyTorch's automatic derivatives (autograd) to prove the formulas are right.

Run:  python hand_example.py        (under 5 seconds)
"""
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(2)
NAMES = ["left", "right"]


def softmax(h):
    e = np.exp(h - np.max(h))
    return e / e.sum()


def one_update(h, V_s, V_s2, r, a, gamma=0.9, alpha=0.1, beta=0.1, done=False, title=""):
    print("=" * 70)
    print(title)
    print("=" * 70)
    h = np.array(h, dtype=float)
    pi = softmax(h)
    print(f"  GIVEN    preferences h = [{h[0]:.4f}, {h[1]:.4f}]   ->  pi = [{pi[0]:.4f}, {pi[1]:.4f}]")
    print(f"           critic V(s) = {V_s}, V(s') = {V_s2},  reward r = {r},  gamma = {gamma}")
    print(f"           action taken: {NAMES[a]}  (it had probability {pi[a]:.4f})")

    # 1. TD error = the critic's surprise (Q-learning's 'target - old guess', but with V)
    target = r + gamma * V_s2 * (not done)
    delta = target - V_s
    print(f"\n  STEP 1   target = r + gamma*V(s') = {r} + {gamma}*{V_s2} = {target:.4f}")
    print(f"           delta  = target - V(s)   = {target:.4f} - {V_s} = {delta:+.4f}")

    # 2. Critic: nudge V(s) toward the target (same as Q-learning's alpha nudge)
    V_new = V_s + beta * delta
    print(f"\n  STEP 2   critic: V(s) <- {V_s} + {beta}*({delta:+.4f}) = {V_new:.4f}")

    # 3. Actor: grad of log pi(a) w.r.t. h is (one_hot(a) - pi)
    grad = -pi.copy()
    grad[a] += 1.0
    h_new = h + alpha * delta * grad
    pi_new = softmax(h_new)
    print(f"\n  STEP 3   grad log pi({NAMES[a]}) = one_hot - pi = [{grad[0]:+.4f}, {grad[1]:+.4f}]")
    print(f"           h <- h + alpha*delta*grad = [{h_new[0]:.4f}, {h_new[1]:.4f}]")
    print(f"           new pi = [{pi_new[0]:.4f}, {pi_new[1]:.4f}]   "
          f"pi({NAMES[a]}): {pi[a]:.4f} -> {pi_new[a]:.4f}  ({pi_new[a] - pi[a]:+.4f})")

    # 4. Check with autograd: same update written as two LOSSES and plain gradient descent
    ht = torch.tensor(h, requires_grad=True)
    Vt = torch.tensor(float(V_s), dtype=torch.float64, requires_grad=True)
    logp = torch.log_softmax(ht, dim=0)[a]
    adv = (r + gamma * V_s2 * (not done)) - Vt
    actor_loss = -logp * adv.detach()            # detach: the actor must not move the critic
    critic_loss = 0.5 * adv ** 2                 # half the squared error
    (actor_loss + critic_loss).backward()
    with torch.no_grad():
        h_auto = ht - alpha * ht.grad            # alpha == beta here, so one learning rate works
        V_auto = Vt - beta * Vt.grad
    ok = np.allclose(h_auto.numpy(), h_new) and abs(V_auto.item() - V_new) < 1e-9
    print(f"\n  CHECK    autograd gives h = [{h_auto[0]:.4f}, {h_auto[1]:.4f}], V = {V_auto.item():.4f}"
          f"   -> {'MATCHES' if ok else 'DIFFERENT!'}\n")
    return delta, V_new, pi_new


def entropy_table():
    print("=" * 70)
    print("ENTROPY of a 2-action policy:  H = -sum pi*ln(pi)   (bigger = more unsure)")
    print("=" * 70)
    for p in [0.5, 0.6, 0.8, 0.9, 0.99]:
        H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        print(f"  pi = [{p:.2f}, {1 - p:.2f}]   H = {H:.3f}")
    print()


def n_step_example(gamma=0.9):
    print("=" * 70)
    print("n-STEP RETURNS: 3 rewards of +1, then the critic's guess V(s3) = 5.0")
    print("=" * 70)
    rewards, V_end = [1.0, 1.0, 1.0], 5.0
    R = V_end
    for t in reversed(range(3)):
        R = rewards[t] + gamma * R
        print(f"  R_{t} = r_{t} + {gamma}*R_{t + 1} = {rewards[t]} + {gamma}*{(R - rewards[t]) / gamma:.4f} = {R:.4f}")
    print(f"  (1-step, like Q-learning, from s2 alone would be  1 + 0.9*V(s3) = {1 + gamma * V_end:.4f})\n")


def main():
    one_update(h=[0.0, 0.0], V_s=0.5, V_s2=0.6, r=1.0, a=1,
               title="EXAMPLE 1  Better than expected (delta > 0), actor starts 50/50")
    one_update(h=[np.log(4.0), 0.0], V_s=2.0, V_s2=1.5, r=0.0, a=0,
               title="EXAMPLE 2  Worse than expected (delta < 0), a COMMON action (80%)")
    one_update(h=[np.log(4.0), 0.0], V_s=2.0, V_s2=3.0, r=0.0, a=1,
               title="EXAMPLE 3  Better than expected, a RARE action (20%)")
    one_update(h=[0.0, 0.0], V_s=0.5, V_s2=9.9, r=-1.0, a=0, done=True,
               title="EXAMPLE 4  The game ENDS (terminated): no future, V(s') is ignored")
    entropy_table()
    n_step_example()


if __name__ == "__main__":
    main()
