"""
DPO (Direct Preference Optimization): learn from the SAME human comparisons,
but skip the reward model and skip PPO.

For each pair (winner y_w, loser y_l) to prompt x:

    z    = beta * [ log(pi(y_w)/pi_ref(y_w))  -  log(pi(y_l)/pi_ref(y_l)) ]
                    ▲ how much MORE likely the     ▲ how much more likely the
                      winner became vs SFT           loser became vs SFT
    loss = -log sigmoid(z)

It is the Bradley-Terry loss again, but the "reward" of a reply is now
beta * log(pi(y)/pi_ref(y)), something the policy itself defines. So training
the policy on this loss IS training a reward model, and no separate judge is needed.

It's plain supervised learning on a fixed dataset: no replies are written during
training, no reward model is called, no advantages, no clipping.

Run:  python dpo.py              (about 30 seconds; needs rlhf_results.npz from rlhf_ppo.py)
      python dpo.py --no-show
Saves: dpo_vs_rlhf.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from reward_model import accuracy, load
from toy_world import (P, PROMPTS, evaluate, flatten, grad_of_logprob, reply_logprobs,
                       sample_replies, sigmoid, to_words)

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))


def dpo(theta_ref, prompts, winners, losers, beta, epochs=600, lr=1.0, eval_every=20,
        rm_w=None, test=None):
    n = len(prompts)
    flat_w, flat_l = flatten(prompts, winners), flatten(prompts, losers)
    ref_w = reply_logprobs(theta_ref, flat_w, n)          # SFT model's log pi of each winner
    ref_l = reply_logprobs(theta_ref, flat_l, n)          # ... and of each loser (fixed)
    theta = theta_ref.copy()
    eval_rng = np.random.default_rng(2024)
    history = []
    for ep in range(epochs + 1):
        gain_w = reply_logprobs(theta, flat_w, n) - ref_w
        gain_l = reply_logprobs(theta, flat_l, n) - ref_l
        z = beta * (gain_w - gain_l)
        if ep % eval_every == 0:
            row = dict(ep=ep, loss=np.mean(-np.log(sigmoid(z))),
                       **evaluate(theta, theta_ref, rm_w, eval_rng, 300))
            if test is not None:
                row["test_acc"] = implicit_accuracy(theta, theta_ref, *test)
            history.append(row)
        if ep == epochs:
            break
        # slope of -log sigmoid(z):  push = beta * (1 - sigmoid(z)) per pair
        # -> raise log pi(winner) and lower log pi(loser), hardest where the model is most wrong
        push = beta * (1 - sigmoid(z))
        grad = grad_of_logprob(theta, flat_w, push[flat_w[3]]) \
            - grad_of_logprob(theta, flat_l, push[flat_l[3]])
        theta += lr * grad / n                             # gradient ASCENT on log sigmoid(z)
    return theta, history


def implicit_accuracy(theta, theta_ref, prompts, winners, losers):
    """DPO's hidden reward model: r(y) = log pi(y)/pi_ref(y). Does it rank held-out pairs right?"""
    n = len(prompts)
    fw, fl = flatten(prompts, winners), flatten(prompts, losers)
    gap = (reply_logprobs(theta, fw, n) - reply_logprobs(theta_ref, fw, n)) \
        - (reply_logprobs(theta, fl, n) - reply_logprobs(theta_ref, fl, n))
    return accuracy(gap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--betas", default="0.1,0.3,1.0")
    parser.add_argument("--epochs", type=int, default=600)
    args = parser.parse_args()
    betas = [float(b) for b in args.betas.split(",")]

    data = load()
    theta_ref, rm_w, k = data["theta_ref"], data["rm_w"], data["n_train"]
    pr, W, L = data["prompts"], data["winners"], data["losers"]
    train = (pr[:k], W[:k], L[:k])                           # the SAME 2,400 pairs the RM used
    test = (pr[k:], W[k:], L[k:])
    start = evaluate(theta_ref, theta_ref, rm_w, np.random.default_rng(5), 2000)

    runs = {}
    for beta in betas:
        print("=" * 70)
        print(f"DPO, beta = {beta}, {args.epochs} passes over {k} preference pairs")
        print("=" * 70)
        theta, hist = dpo(theta_ref, *train, beta, epochs=args.epochs, rm_w=rm_w, test=test)
        runs[beta] = (theta, hist)
        for h in hist[::5] + ([hist[-1]] if len(hist) % 5 != 1 else []):
            print(f"  pass {h['ep']:>4}: loss {h['loss']:.4f}  held-out acc {100 * h['test_acc']:.1f}%"
                  f"  RM {h['rm']:+.2f}  TRUE {h['true']:+.2f}  KL {h['kl']:.2f}  len {h['length']:.2f}")
        rng = np.random.default_rng(3)
        texts = [" ".join(to_words(sample_replies(theta, np.array([p]), rng)[0])) or "(empty)"
                 for p in range(P)]
        print("  sample replies: " + "  |  ".join(texts))

    # ---- final comparison, measured with many replies --------------------------------
    rl = np.load(os.path.join(HERE, "rlhf_results.npz"))
    rl_betas = list(rl["betas"])
    best_rl = max(rl_betas, key=lambda b: rl[f"true_{b}"][:, -1].mean())
    best_dpo = max(betas, key=lambda b: runs[b][1][-1]["true"])
    rng = np.random.default_rng(77)
    rows = [("SFT (pi_ref)", theta_ref),
            (f"RLHF PPO, β={rl_betas[0]:g}", rl[f"theta_{rl_betas[0]}"]),
            (f"RLHF PPO, β={best_rl:g}", rl[f"theta_{best_rl}"])] + \
           [(f"DPO, β={b:g}", runs[b][0]) for b in betas]
    print("\n" + "=" * 70)
    print("COMPARISON (6,000 fresh replies each, seed-0 models)")
    print("=" * 70)
    print(f"  {'model':<20} {'RM score':>9} {'TRUE':>7} {'KL':>6} {'length':>7} {'repeats':>8}")
    table = []
    for name, th in rows:
        m = evaluate(th, theta_ref, rm_w, rng, 2000)
        table.append((name, m))
        print(f"  {name:<20} {m['rm']:>+9.2f} {m['true']:>+7.2f} {m['kl']:>6.2f} "
              f"{m['length']:>7.2f} {m['repeats']:>8.2f}")
    print(f"\n  Reward model's held-out accuracy (from reward_model.py) vs DPO's built-in judge:")
    for b in betas:
        print(f"     DPO β={b:g}: {100 * runs[b][1][-1]['test_acc']:.1f}%")

    # ---- charts -------------------------------------------------------------------------
    colors = ["#A86A08", "#2C4BD1", "#1E8A5A"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for b, c in zip(betas, colors):
        h = runs[b][1]
        eps = [x["ep"] for x in h]
        axes[0].plot(eps, [x["loss"] for x in h], color=c, label=f"DPO β = {b:g}")
        axes[1].plot(eps, [x["true"] for x in h], color=c, label=f"DPO β = {b:g}")
    axes[1].axhline(start["true"], color="gray", ls="--", lw=1, label="SFT model")
    axes[0].set_title("DPO loss (starts at log 2 = 0.693)")
    axes[1].set_title("TRUE human score during DPO")
    for ax in axes[:2]:
        ax.set_xlabel("pass over the 2,400 pairs")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax = axes[2]
    names = [n for n, _ in table]
    ax.barh(names, [m["true"] for _, m in table],
            color=["#9AA3B9", "#C2413A", "#2C4BD1"] + colors[:len(betas)])
    for i, (_, m) in enumerate(table):
        ax.text(max(m["true"], 0) + 0.1, i, f"{m['true']:+.2f}  (KL {m['kl']:.2f})",
                va="center", ha="left", fontsize=8.5)
    ax.axvline(0, color="#151B2F", lw=0.8)
    ax.set_xlim(-8, 7.5)
    ax.invert_yaxis()
    ax.set_title("Final TRUE score (KL from SFT in brackets)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "dpo_vs_rlhf.png"), dpi=110, bbox_inches="tight")
    print("\n  Chart saved to dpo_vs_rlhf.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
