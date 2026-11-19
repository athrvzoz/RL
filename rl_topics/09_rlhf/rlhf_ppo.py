"""
Stage 4 of RLHF: fine-tune the policy with PPO, using the learned reward model
as the judge, plus a KL penalty that keeps it close to the SFT model.

For every reply y the policy writes (one episode), the reward is

    R(y) = r_RM(y)  -  beta * log( pi(y) / pi_ref(y) )
           ▲ judge         ▲ "how much more likely is this reply now than
             says            it was for the SFT model?"  (averages to the KL)

The reward arrives only at the END of the reply, so every word in the reply
shares the same advantage (like a Monte Carlo return in a game with one prize
at the finish line).

We run several beta values. With beta = 0 nothing stops the policy from
exploiting the reward model's blind spot (it can't see repetition), and the
TRUE quality collapses while the reward model's score keeps climbing.

Run:  python rlhf_ppo.py                (about 30 seconds)
      python rlhf_ppo.py --no-show
Saves: rlhf_beta_sweep.png, reward_hacking.png, rlhf_results.npz (used by dpo.py)
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from reward_model import load
from toy_world import (P, PROMPTS, evaluate, features, flatten, grad_of_logprob,
                       sample_replies, to_words, token_logprobs, true_score)

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))


def ppo_with_kl(theta_ref, rm_w, beta, iters=400, batch=96, lr=0.3, epochs=4, clip=0.2,
                eval_every=10, seed=0, snapshots=()):
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(1000 + seed)
    theta = theta_ref.copy()                       # start FROM the SFT model
    history, saved = [], {}
    for it in range(iters + 1):
        if it % eval_every == 0:
            history.append(dict(it=it, **evaluate(theta, theta_ref, rm_w, eval_rng, 300)))
        if it in snapshots:
            saved[it] = theta.copy()
        if it == iters:
            break

        # 1. ROLL OUT: the policy writes a batch of replies (32 per prompt)
        prompts = np.repeat(np.arange(P), batch // P)
        replies = sample_replies(theta, prompts, rng)
        flat = flatten(prompts, replies)
        n = len(replies)

        # 2. SCORE: reward model judges each finished reply, minus the KL penalty
        logp_old = token_logprobs(theta, flat)
        logp_ref = token_logprobs(theta_ref, flat)
        log_ratio = np.bincount(flat[3], weights=logp_old - logp_ref, minlength=n)
        r_rm = np.array([features(p, to_words(r)) @ rm_w for p, r in zip(prompts, replies)])
        R = r_rm - beta * log_ratio

        # 3. ADVANTAGE: better or worse than the other replies to the same prompt?
        adv = R.copy()
        for p in range(P):
            adv[prompts == p] -= R[prompts == p].mean()        # baseline = average reply
        adv /= adv.std() + 1e-8
        adv_tok = adv[flat[3]]                                 # every word shares its reply's advantage

        # 4. PPO UPDATE: a few gradient steps on the clipped objective
        for _ in range(epochs):
            ratio = np.exp(token_logprobs(theta, flat) - logp_old)   # pi_new / pi_old per word
            # min(ratio*A, clip(ratio)*A): once a word's probability moved more than 20%
            # in the helpful direction, stop pushing it (its gradient becomes 0)
            active = np.where(adv_tok > 0, ratio < 1 + clip, ratio > 1 - clip)
            theta += lr * grad_of_logprob(theta, flat, adv_tok * ratio * active) / n
    return theta, history, saved


def curves(histories, key):
    return np.array([[h[key] for h in hist] for hist in histories])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--betas", default="0,0.1,0.2,0.3,1.0")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=400)
    args = parser.parse_args()
    betas = [float(b) for b in args.betas.split(",")]

    data = load()
    theta_ref, rm_w = data["theta_ref"], data["rm_w"]
    start = evaluate(theta_ref, theta_ref, rm_w, np.random.default_rng(5), 2000)
    print("=" * 70)
    print("START · the SFT model (pi_ref)")
    print("=" * 70)
    print(f"  reward-model score {start['rm']:+.2f}   TRUE score {start['true']:+.2f}   "
          f"length {start['length']:.2f}   repeated words {start['repeats']:.2f}")

    results, final_thetas = {}, {}
    snaps = (0, 50, 100, 200, 400)
    for beta in betas:
        print("\n" + "=" * 70)
        print(f"PPO with KL penalty, beta = {beta}  ({args.seeds} seeds x {args.iters} iterations)")
        print("=" * 70)
        hists = []
        for seed in range(args.seeds):
            theta, hist, saved = ppo_with_kl(theta_ref, rm_w, beta, iters=args.iters, seed=seed,
                                             snapshots=snaps)
            hists.append(hist)
            if seed == 0:
                final_thetas[beta] = theta
                rng = np.random.default_rng(3)
                for it in snaps:
                    if it in saved:
                        texts = [" ".join(to_words(sample_replies(saved[it], np.array([p]), rng)[0]))
                                 or "(empty)" for p in range(P)]
                        print(f"  iter {it:>3} seed 0 replies: " + "  |  ".join(texts))
        results[beta] = hists
        rm, tr, kl = (curves(hists, k)[:, -1] for k in ("rm", "true", "kl"))
        tr_curve = curves(hists, "true").mean(axis=0)
        best = int(np.argmax(tr_curve))
        print(f"  FINAL (mean of seeds): RM {rm.mean():+.2f}   TRUE {tr.mean():+.2f} "
              f"(seeds {', '.join(f'{x:+.2f}' for x in tr)})   KL {kl.mean():.2f}")
        print(f"  best TRUE along the way: {tr_curve[best]:+.2f} at iteration {hists[0][best]['it']}")

    print("\n" + "=" * 70)
    print("SUMMARY (final iteration, mean of seeds)")
    print("=" * 70)
    print(f"  {'beta':>6} {'RM score':>9} {'TRUE':>7} {'KL':>7} {'length':>7} {'repeats':>8}")
    print(f"  {'SFT':>6} {start['rm']:>+9.2f} {start['true']:>+7.2f} {0:>7.2f} "
          f"{start['length']:>7.2f} {start['repeats']:>8.2f}")
    for beta in betas:
        m = {k: curves(results[beta], k)[:, -1].mean() for k in ("rm", "true", "kl", "length", "repeats")}
        print(f"  {beta:>6} {m['rm']:>+9.2f} {m['true']:>+7.2f} {m['kl']:>7.2f} "
              f"{m['length']:>7.2f} {m['repeats']:>8.2f}")

    its = [h["it"] for h in results[betas[0]][0]]
    np.savez(os.path.join(HERE, "rlhf_results.npz"), its=its, betas=betas,
             **{f"{k}_{b}": curves(results[b], k) for b in betas for k in ("rm", "true", "kl")},
             **{f"theta_{b}": final_thetas[b] for b in betas})

    # ---- chart 1: the beta sweep ------------------------------------------------
    colors = ["#C2413A", "#A86A08", "#2C4BD1", "#1E8A5A", "#7B3FB0"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for k, (key, title) in enumerate([("rm", "Reward-model score (what PPO maximises)"),
                                      ("true", "TRUE human score (hidden)"),
                                      ("kl", "KL from the SFT model")]):
        ax = axes[k]
        for beta, c in zip(betas, colors):
            ax.plot(its, curves(results[beta], key).mean(axis=0), color=c, label=f"β = {beta}")
        if key == "true":
            ax.axhline(start["true"], color="gray", ls="--", lw=1, label="SFT model")
        ax.set_title(title)
        ax.set_xlabel("PPO iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[2].set_yscale("symlog", linthresh=1)
    fig.suptitle(f"RLHF with PPO + KL penalty (mean of {args.seeds} seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "rlhf_beta_sweep.png"), dpi=110, bbox_inches="tight")

    # ---- chart 2: reward hacking at beta = 0 --------------------------------------
    b0 = betas[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    ax.plot(its, curves(results[b0], "rm")[0], color="#2C4BD1", label="reward-model score")
    ax.set_ylabel("reward-model score", color="#2C4BD1")
    ax.set_xlabel("PPO iteration")
    ax2 = ax.twinx()
    ax2.plot(its, curves(results[b0], "true")[0], color="#C2413A", label="TRUE score")
    ax2.set_ylabel("TRUE human score", color="#C2413A")
    ax.set_title(f"β = {b0}: the judge is happy, the humans are not (seed 0)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(its, curves(results[b0], "length")[0], color="#151B2F", label="reply length (words)")
    ax.plot(its, curves(results[b0], "repeats")[0], color="#A86A08", label="repeated words per reply")
    ax.set_xlabel("PPO iteration")
    ax.set_title("How it cheats: longer replies, full of repeats")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reward_hacking.png"), dpi=110, bbox_inches="tight")
    print("\n  Charts saved to rlhf_beta_sweep.png and reward_hacking.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
