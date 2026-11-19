"""
Project 1: Slot-machine casino.
Compare greedy, epsilon-greedy and UCB over 1000 pulls, averaged over
many randomly generated casinos (one casino alone is too noisy to judge).

Run:
    python experiment.py                 # 2000 casinos, takes about a minute
    python experiment.py --runs 200      # quick version
    python experiment.py --no-show       # save the chart without opening a window
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from agents import EpsilonGreedyAgent, UCBAgent
from bandit import SlotMachineCasino

# name -> function that builds a fresh agent
AGENTS = {
    "greedy (ε = 0)":     lambda k, rng: EpsilonGreedyAgent(k, 0.0, rng),
    "ε-greedy (ε = 0.01)": lambda k, rng: EpsilonGreedyAgent(k, 0.01, rng),
    "ε-greedy (ε = 0.1)":  lambda k, rng: EpsilonGreedyAgent(k, 0.1, rng),
    "UCB (c = 2)":         lambda k, rng: UCBAgent(k, 2.0, rng),
}


def play_one_casino(make_agent, k, steps, run, seed):
    # Every agent meets the SAME casino on run number `run`, so the
    # comparison is fair. The agent gets its own random stream.
    casino = SlotMachineCasino(k, np.random.default_rng([seed, run]))
    agent = make_agent(k, np.random.default_rng([seed, run, 1]))

    rewards = np.zeros(steps)
    picked_best = np.zeros(steps)
    for t in range(steps):
        arm = agent.choose()            # 1. agent picks a machine
        reward = casino.pull(arm)       # 2. casino pays out
        agent.update(arm, reward)       # 3. agent updates Q[arm]
        rewards[t] = reward
        picked_best[t] = arm == casino.best_arm
    return rewards, picked_best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2000, help="number of casinos to average over")
    parser.add_argument("--steps", type=int, default=1000, help="pulls per casino")
    parser.add_argument("--k", type=int, default=10, help="machines per casino")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true", help="don't open the chart window")
    args = parser.parse_args()

    results = {}
    for name, make_agent in AGENTS.items():
        start = time.time()
        total_rewards = np.zeros(args.steps)
        total_best = np.zeros(args.steps)
        for run in range(args.runs):
            r, b = play_one_casino(make_agent, args.k, args.steps, run, args.seed)
            total_rewards += r
            total_best += b
        # Average over all casinos -> one learning curve per agent.
        results[name] = (total_rewards / args.runs, 100 * total_best / args.runs)
        print(f"  finished {name:<20} in {time.time() - start:5.1f}s")

    print(f"\nResults over {args.runs} casinos x {args.steps} pulls")
    print(f"{'agent':<22}{'avg reward':>12}{'last 100 avg':>14}{'% best (last 100)':>20}")
    for name, (avg_reward, pct_best) in results.items():
        print(f"{name:<22}{avg_reward.mean():>12.3f}{avg_reward[-100:].mean():>14.3f}{pct_best[-100:].mean():>19.1f}%")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name, (avg_reward, pct_best) in results.items():
        ax1.plot(avg_reward, label=name, linewidth=1.2)
        ax2.plot(pct_best, label=name, linewidth=1.2)
    ax1.set_ylabel("Average reward")
    ax1.set_title(f"Slot-machine casino: {args.k} machines, averaged over {args.runs} casinos")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("% of pulls on the best machine")
    ax2.set_xlabel("Pull number")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results.png", dpi=120)
    print("\nChart saved to results.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
