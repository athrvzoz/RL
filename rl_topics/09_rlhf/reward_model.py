"""
Stages 2 and 3 of RLHF on our toy chat assistant.

  Stage 2 · SFT: copy a few human-written demo replies  -> the "reference" policy pi_ref
  Stage 3 · Reward model: ask (simulated) humans which of two replies is better,
            3,000 times, and fit a Bradley-Terry model:
                P(A preferred over B) = sigmoid(r(A) - r(B))
            The learned r(.) is the reward model. It will be the "judge" during RL.

Why not just write the reward by hand? Because in real life nobody can write a
formula for "a good answer". People CAN say which of two answers they like more.
(Here the humans are simulated by the hidden function toy_world.true_score, so
we can check the result. The learner only ever sees the A/B choices.)

The reward model is deliberately simple: it only counts words. It never sees a
word repeated eight times during training, and it adds counts in a straight line,
so it will happily give "refund refund refund ..." a high score. That blind spot
is what reward hacking (rlhf_ppo.py) will exploit.

Run:  python reward_model.py            (a few seconds)
      python reward_model.py --no-show
Saves: stage_data.npz (used by rlhf_ppo.py and dpo.py), reward_model.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from toy_world import (DEMOS, P, PROMPTS, V, WORDS, features, human_prefers_first, sft,
                       sample_replies, sigmoid, to_words, true_score)

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "stage_data.npz")


def collect_comparisons(theta_ref, n_pairs, rng):
    """Write two replies to the same prompt with the SFT model, ask a 'human' to pick one."""
    prompts = rng.integers(P, size=n_pairs)
    replies_a = sample_replies(theta_ref, prompts, rng)
    replies_b = sample_replies(theta_ref, prompts, rng)
    winners, losers = [], []
    for p, a, b in zip(prompts, replies_a, replies_b):
        if human_prefers_first(p, to_words(a), to_words(b), rng):
            winners.append(a); losers.append(b)
        else:
            winners.append(b); losers.append(a)
    return prompts, winners, losers


def train_reward_model(X_diff, epochs=500, lr=1.0, l2=1e-3):
    """
    Bradley-Terry training. Each row of X_diff = features(winner) - features(loser),
    so  r(winner) - r(loser) = X_diff @ w   (the reward model is r(y) = w . features(y)).

        loss  = mean of  -log sigmoid(r_w - r_l)
        slope = -(1 - sigmoid(r_w - r_l)) * (features_w - features_l)
    """
    w = np.zeros(X_diff.shape[1])
    losses = []
    for _ in range(epochs):
        d = X_diff @ w                                          # r_w - r_l for every pair
        losses.append(np.mean(-np.log(sigmoid(d))))
        grad = -((1 - sigmoid(d))[:, None] * X_diff).mean(axis=0) + l2 * w
        w -= lr * grad                                          # gradient DESCENT on the loss
    return w, np.array(losses)


def accuracy(gap):
    """Fraction of pairs where the human's pick gets the higher score (a tie counts as half)."""
    return np.mean(gap > 0) + 0.5 * np.mean(gap == 0)


def build(n_pairs=3000, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    theta_ref = sft()                                           # stage 2
    prompts, winners, losers = collect_comparisons(theta_ref, n_pairs, rng)
    X = np.array([features(p, to_words(a)) - features(p, to_words(b))
                  for p, a, b in zip(prompts, winners, losers)])
    n_train = int(0.8 * n_pairs)                                # 80% to learn, 20% held out
    w, losses = train_reward_model(X[:n_train])
    return dict(theta_ref=theta_ref, rm_w=w, losses=losses, X=X, n_train=n_train,
                prompts=prompts, winners=winners, losers=losers)


def save(d):
    pad = lambda lst: np.array([t + [-1] * (9 - len(t)) for t in lst])    # same length rows
    np.savez(DATA, theta_ref=d["theta_ref"], rm_w=d["rm_w"], n_train=d["n_train"],
             prompts=d["prompts"], winners=pad(d["winners"]), losers=pad(d["losers"]))


def load():
    """Used by rlhf_ppo.py and dpo.py. Builds the data first if it isn't there yet."""
    if not os.path.exists(DATA):
        save(build())
    z = np.load(DATA)
    unpad = lambda arr: [[int(t) for t in row if t >= 0] for row in arr]
    return dict(theta_ref=z["theta_ref"], rm_w=z["rm_w"], n_train=int(z["n_train"]),
                prompts=z["prompts"], winners=unpad(z["winners"]), losers=unpad(z["losers"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    print("=" * 64)
    print("STAGE 2 · Supervised fine-tuning on", sum(len(d) for d in DEMOS), "demo replies")
    print("=" * 64)
    rng = np.random.default_rng(1)
    zero = np.zeros((P, V + 1, V))
    d = build()
    for name, theta in [("BEFORE SFT (every word equally likely)", zero),
                        ("AFTER SFT (this is pi_ref)", d["theta_ref"])]:
        print(f"\n  {name}: sample replies to '{PROMPTS[1]}'")
        for r in sample_replies(theta, np.full(5, 1), rng):
            w = to_words(r)
            print(f"     {' '.join(w) or '(empty)':<45} true score {true_score(1, w):+.1f}")
    big = np.repeat(np.arange(P), 2000)
    for name, theta in [("before SFT", zero), ("after SFT ", d["theta_ref"])]:
        reps = [to_words(r) for r in sample_replies(theta, big, np.random.default_rng(7))]
        print(f"  Average hidden true score {name}: "
              f"{np.mean([true_score(p, w) for p, w in zip(big, reps)]):+.2f}"
              f"   (avg length {np.mean([len(w) for w in reps]):.2f} words)")

    print("\n" + "=" * 64)
    print("STAGE 3 · Reward model from 3,000 human comparisons")
    print("=" * 64)
    X, n_train, w = d["X"], d["n_train"], d["rm_w"]
    for i in range(3):
        p, a, b = d["prompts"][i], to_words(d["winners"][i]), to_words(d["losers"][i])
        print(f"  pair {i}: '{PROMPTS[p]}'\n     human picked: {' '.join(a) or '(empty)':<40}"
              f"over: {' '.join(b) or '(empty)'}")
    gap = X @ w                                                 # r(winner) - r(loser)
    true_gap = np.array([true_score(p, to_words(a)) - true_score(p, to_words(b))
                         for p, a, b in zip(d["prompts"], d["winners"], d["losers"])])
    print(f"\n  Loss: start {d['losses'][0]:.4f} (= log 2, a coin flip)  ->  end {d['losses'][-1]:.4f}")
    print(f"  Accuracy on the 2,400 training pairs : {100 * accuracy(gap[:n_train]):.1f}%")
    print(f"  Accuracy on the 600 held-out pairs   : {100 * accuracy(gap[n_train:]):.1f}%")
    print(f"  Best possible (judge with the hidden true score): "
          f"{100 * accuracy(true_gap[n_train:]):.1f}%   <- labellers are noisy, nobody gets 100%")

    print("\n  Learned reward per word (r(reply) = sum of these over the words in it):")
    print("     " + " " * 22 + "".join(f"{x:>9}" for x in WORDS[1:]))
    for p in range(P):
        print(f"     {PROMPTS[p]:<22}" + "".join(f"{w[p * V + v]:+9.2f}" for v in range(1, V)))

    print("\n  The blind spot: the reward model scores replies by adding up word weights.")
    for text in ["sorry refund", "hi sorry refund please", "sorry refund sorry refund",
                 "refund refund refund refund refund refund refund refund"]:
        ws = text.split()
        print(f"     {text:<58} RM {features(1, ws) @ w:+.2f}   true {true_score(1, ws):+.1f}")

    save(d)
    print(f"\n  Saved {os.path.basename(DATA)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    ax.plot(d["losses"], color="#2C4BD1")
    ax.axhline(np.log(2), color="gray", ls="--", lw=1)
    ax.text(len(d["losses"]) * 0.45, np.log(2) - 0.008, "log 2 = coin flip", color="gray")
    ax.set_xlabel("Training step (full pass over 2,400 pairs)")
    ax.set_ylabel("Bradley–Terry loss  −log σ(r_w − r_l)")
    ax.set_title(f"Reward model training · held-out accuracy {100 * accuracy(gap[n_train:]):.1f}%")
    ax.grid(alpha=0.3)
    ax = axes[1]
    M = w.reshape(P, V)[:, 1:]
    im = ax.imshow(M, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(V - 1), WORDS[1:])
    ax.set_yticks(range(P), PROMPTS)
    for i in range(P):
        for j in range(V - 1):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(M[i, j]) > 0.6 else "black")
    ax.set_title("Learned reward per word (blue = good, red = bad)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reward_model.png"), dpi=110, bbox_inches="tight")
    print("  Chart saved to reward_model.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
