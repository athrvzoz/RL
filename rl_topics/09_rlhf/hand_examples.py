"""
The small hand calculations from the booklet, checked by the computer.

Do them on paper first (a calculator for e^x is enough), then run this to check.
  1. Bradley-Terry: how likely is "A beats B", the loss, and its slope (gradient)
  2. KL divergence between two 3-outcome dice
  3. The RLHF reward with the KL penalty:  reward = r_RM - beta * KL
  4. One DPO pair: the loss and which way it pushes
  5. Recap: one policy-gradient step on a softmax row, and PPO's clip

Run:  python hand_examples.py
"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bradley_terry():
    print("=" * 60)
    print("1 · Bradley-Terry: reward model says r_A = 2.0, r_B = 1.0")
    print("=" * 60)
    r_a, r_b = 2.0, 1.0
    d = r_a - r_b
    p = sigmoid(d)
    print(f"  d = r_A - r_B = {d}")
    print(f"  P(A preferred) = sigmoid(d) = 1 / (1 + e^-{d}) = 1 / (1 + {np.exp(-d):.4f}) = {p:.4f}")
    print(f"  If the human picked A:  loss = -log(sigmoid(d)) = -log({p:.4f}) = {-np.log(p):.4f}")
    print(f"     slope d(loss)/d(r_A) = -(1 - sigmoid(d)) = {-(1 - p):.4f}   (raise r_A)")
    print(f"     slope d(loss)/d(r_B) = +(1 - sigmoid(d)) = {+(1 - p):.4f}   (lower r_B)")
    print(f"  If the human picked B (the model was WRONG):  loss = -log(sigmoid(-d)) = {-np.log(sigmoid(-d)):.4f}")
    print(f"     slope d(loss)/d(r_B) = -(1 - sigmoid(-d)) = {-(1 - sigmoid(-d)):.4f}  (big push)")
    lr = 0.5
    new_a, new_b = r_a + lr * (1 - p), r_b - lr * (1 - p)
    print(f"  One gradient step (lr {lr}) after 'A won':  r_A {r_a} -> {new_a:.4f},  r_B {r_b} -> {new_b:.4f}")
    print(f"     new P(A preferred) = {sigmoid(new_a - new_b):.4f},  new loss = {-np.log(sigmoid(new_a - new_b)):.4f}")
    print("\n  Table of sigmoid / loss for different gaps d:")
    for d in [-2, -1, 0, 1, 2, 4]:
        print(f"     d = {d:>2}:  sigmoid = {sigmoid(d):.3f}   loss = {-np.log(sigmoid(d)):.3f}   "
              f"slope = {-(1 - sigmoid(d)):.3f}")


def kl_example():
    print("\n" + "=" * 60)
    print("2 · KL divergence: how far has the policy drifted from the SFT model?")
    print("=" * 60)
    words = ["sorry", "refund", "great"]
    ref = np.array([0.5, 0.3, 0.2])
    for name, new in [("small change", [0.4, 0.5, 0.1]), ("big change", [0.05, 0.9, 0.05])]:
        new = np.array(new)
        terms = new * np.log(new / ref)
        print(f"  {name}:  pi_ref = {ref.tolist()}   pi_new = {new.tolist()}")
        for w, q, r, t in zip(words, new, ref, terms):
            print(f"     {w:>7}: {q:.2f} * log({q:.2f}/{r:.2f}) = {q:.2f} * {np.log(q / r):+.4f} = {t:+.4f}")
        print(f"     KL = sum = {terms.sum():.4f}")


def penalised_reward():
    print("\n" + "=" * 60)
    print("3 · Reward used by RL:  r_RM - beta * log(pi_new(y) / pi_ref(y))")
    print("=" * 60)
    # one reply that the new policy now writes MUCH more often than the SFT model did
    r_rm, p_new, p_ref = 2.5, 0.40, 0.02
    log_ratio = np.log(p_new / p_ref)
    print(f"  reward model says {r_rm};  pi_new(y) = {p_new}, pi_ref(y) = {p_ref};  "
          f"log ratio = {log_ratio:.4f}")
    for beta in [0.0, 0.1, 0.5, 1.0]:
        print(f"     beta = {beta:<4}: {r_rm} - {beta} * {log_ratio:.3f} = {r_rm - beta * log_ratio:+.4f}")


def dpo_pair():
    print("\n" + "=" * 60)
    print("4 · One DPO pair (beta = 0.5)")
    print("=" * 60)
    beta = 0.5
    # log-probabilities of the preferred (w) and rejected (l) reply
    lw, lw_ref = np.log(0.10), np.log(0.08)
    ll, ll_ref = np.log(0.05), np.log(0.06)
    z = beta * ((lw - lw_ref) - (ll - ll_ref))
    print(f"  winner: log(0.10/0.08) = {lw - lw_ref:+.4f}    loser: log(0.05/0.06) = {ll - ll_ref:+.4f}")
    print(f"  z = beta * (winner gain - loser gain) = {beta} * ({lw - lw_ref:+.4f} - ({ll - ll_ref:+.4f}))"
          f" = {z:.4f}")
    print(f"  loss = -log(sigmoid(z)) = -log({sigmoid(z):.4f}) = {-np.log(sigmoid(z)):.4f}")
    print(f"  push strength beta*(1 - sigmoid(z)) = {beta * (1 - sigmoid(z)):.4f}"
          "  (raise log pi(winner), lower log pi(loser))")
    print(f"  (at the very start pi = pi_ref, so z = 0 and loss = log 2 = {np.log(2):.4f})")


def policy_gradient_recap():
    print("\n" + "=" * 60)
    print("5 · Recap: one policy-gradient step on a 3-word softmax row")
    print("=" * 60)
    words = ["sorry", "refund", "great"]
    logits = np.zeros(3)
    probs = np.exp(logits) / np.exp(logits).sum()
    a, adv, lr = 1, +1.0, 0.3                      # wrote "refund", it went well
    grad = -probs.copy()
    grad[a] += 1                                   # d log pi(a) / d logits = onehot(a) - pi
    new = logits + lr * adv * grad
    new_probs = np.exp(new) / np.exp(new).sum()
    print(f"  logits {logits.tolist()} -> probs {np.round(probs, 4).tolist()}")
    print(f"  wrote '{words[a]}', advantage {adv:+}:  gradient = onehot - pi = {np.round(grad, 4).tolist()}")
    print(f"  new logits = old + {lr} * {adv} * gradient = {np.round(new, 4).tolist()}")
    print(f"  new probs  = {np.round(new_probs, 4).tolist()}")
    print("\n  PPO clip (epsilon = 0.2): objective = min(ratio*A, clip(ratio, 0.8, 1.2)*A)")
    for ratio, A in [(1.1, 2.0), (1.3, 2.0), (0.7, -2.0), (0.7, 2.0)]:
        plain, clipped = ratio * A, np.clip(ratio, 0.8, 1.2) * A
        used = min(plain, clipped)
        note = "clipped: gradient 0, stop pushing" if used != plain else "normal gradient"
        print(f"     ratio {ratio}, A {A:+}: min({plain:+.2f}, {clipped:+.2f}) = {used:+.2f}   ({note})")


if __name__ == "__main__":
    bradley_terry()
    kl_example()
    penalised_reward()
    dpo_pair()
    policy_gradient_recap()
