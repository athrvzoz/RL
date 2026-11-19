"""
A neural network small enough to do by hand: 2 inputs -> 2 hidden (ReLU) -> 2 outputs.

Shows, with real numbers:
  1. the forward pass = dot products / a matrix multiplication + ReLU
  2. the DQN loss for one move
  3. backpropagation = the chain rule, one weight at a time
  4. one gradient-descent step, and how Q changes after it
  5. why max over noisy estimates is too optimistic (Double DQN's reason)

Run:  python hand_example.py
"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# state s = 2 numbers (think: pole angle, pole spin)
s = np.array([1.0, 2.0])
W1 = np.array([[0.5, -0.2],       # hidden neuron 1 weights
               [0.3, 0.1]])       # hidden neuron 2 weights
b1 = np.array([0.1, -0.4])
W2 = np.array([[1.0, -1.0],       # output 0 = Q(s, left)
               [0.5, 2.0]])       # output 1 = Q(s, right)
b2 = np.array([0.0, 0.2])


def forward(W1, b1, W2, b2, s):
    z = W1 @ s + b1               # matrix x vector = one dot product per hidden neuron
    h = np.maximum(z, 0)          # ReLU: negative -> 0
    q = W2 @ h + b2
    return z, h, q


print("1 · FORWARD PASS")
z, h, q = forward(W1, b1, W2, b2, s)
print(f"  z1 = 0.5×1 + (−0.2)×2 + 0.1 = {z[0]:.2f}     z2 = 0.3×1 + 0.1×2 − 0.4 = {z[1]:.2f}")
print(f"  ReLU:  h1 = {h[0]:.2f}   h2 = {h[1]:.2f}")
print(f"  Q(left)  = 1.0×{h[0]:.2f} + (−1.0)×{h[1]:.2f} + 0.0 = {q[0]:.2f}")
print(f"  Q(right) = 0.5×{h[0]:.2f} + 2.0×{h[1]:.2f} + 0.2 = {q[1]:.2f}")

print("\n2 · THE LOSS FOR ONE MOVE")
a, r, gamma, max_q_next = 1, 1.0, 0.9, 1.0
target = r + gamma * max_q_next
loss = (target - q[a]) ** 2
print(f"  took RIGHT, r = {r}, best Q in s2 (from the target net) = {max_q_next}")
print(f"  target = {r} + {gamma} × {max_q_next} = {target:.2f}")
print(f"  loss = (target − Q(s,right))² = ({target:.2f} − {q[a]:.2f})² = {loss:.4f}")

print("\n3 · BACKPROP (chain rule) for a few weights")
dL_dq = -2 * (target - q[a])
print(f"  dL/dQ = −2 × (target − Q) = {dL_dq:.2f}")
dq_dW2_10, dq_dW2_11 = h[0], h[1]
print(f"  dL/dW2[right,1] = dL/dQ × h1 = {dL_dq:.2f} × {h[0]:.2f} = {dL_dq * h[0]:.3f}")
print(f"  dL/dW2[right,2] = dL/dQ × h2 = {dL_dq:.2f} × {h[1]:.2f} = {dL_dq * h[1]:.3f}")
dL_dh = dL_dq * W2[a]
dL_dz = dL_dh * (z > 0)
print(f"  dL/dh1 = dL/dQ × W2[right,1] = {dL_dq:.2f} × {W2[a, 0]} = {dL_dh[0]:.3f}")
print(f"  ReLU passes the gradient only where z > 0 (z1 = {z[0]:.2f} > 0, so it passes)")
print(f"  dL/dW1[1,1] = dL/dz1 × s1 = {dL_dz[0]:.3f} × {s[0]} = {dL_dz[0] * s[0]:.3f}")

print("\n4 · ONE GRADIENT-DESCENT STEP (learning rate 0.01), all weights")
lr = 0.01
gW2 = np.zeros_like(W2)
gW2[a] = dL_dq * h
gb2 = np.zeros_like(b2)
gb2[a] = dL_dq
gW1 = np.outer(dL_dz, s)
gb1 = dL_dz
W1n, b1n, W2n, b2n = W1 - lr * gW1, b1 - lr * gb1, W2 - lr * gW2, b2 - lr * gb2
_, _, qn = forward(W1n, b1n, W2n, b2n, s)
_, _, qn_check = forward(W1n, b1n, W2n, b2n, s)
print(f"  Q(s, right): {q[a]:.3f} -> {qn[a]:.3f}   (target {target:.2f})")
print(f"  new loss = {(target - qn[a]) ** 2:.4f}   (was {loss:.4f})")
print(f"  Q(s, left) also moved: {q[0]:.3f} -> {qn[0]:.3f}  (shared hidden layer!)")

print("\n5 · WHY max IS TOO OPTIMISTIC")
rng = np.random.default_rng(0)
true_q = np.zeros(4)                      # all 4 actions are truly worth 0
noisy = true_q + rng.normal(0, 1, size=(100_000, 4))   # estimates with noise
print(f"  4 actions, all truly worth 0. Estimates = truth + noise (std 1).")
print(f"  average of ONE estimate: {noisy[:, 0].mean():+.3f}")
print(f"  average of the MAX of the 4 estimates: {noisy.max(axis=1).mean():+.3f}   <- too high!")
pick = noisy.argmax(axis=1)
second = true_q[pick] + rng.normal(0, 1, size=100_000)  # independent second opinion of the picked action
print(f"  Double trick: pick with one estimate, value with an independent one: {second.mean():+.3f}")
