"""
Tabular Q-learning (RL From Zero, chapter 11).

Works for any Gymnasium game whose states and actions are plain numbers
(FrozenLake, Taxi, CliffWalking ...).

The big difference from project 2: the agent gets NO rule book.
It can't call transitions(). It only sees what env.step() hands back:
the next state, the reward, and whether the game ended.
"""
import numpy as np


def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))         # explore: random action
    best = np.flatnonzero(Q[s] == Q[s].max())         # exploit: best action,
    return int(rng.choice(best))                      # ties broken randomly


def train(env, episodes, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05,
          eps_decay=0.999, seed=0):
    """
    Returns the learned Q-table and the total reward of every training game.

    Q has one row per state and one column per action:
        Q[s, a] = "if I'm in state s and do a, then play well,
                   how much total reward do I expect?"
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    episode_rewards = np.zeros(episodes)
    eps = eps_start
    env.reset(seed=seed)

    for ep in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = epsilon_greedy(Q, s, eps, rng)
            s2, r, terminated, truncated, _ = env.step(a)

            # The Q-learning update: Bellman optimality as a small nudge.
            #   target = reward now + gamma * (best Q in the next state)   <- "Q-max"
            # If the game ENDED (fell in a hole, reached the goal, delivered
            # the passenger) there is no future, so the target is just r.
            target = r + gamma * np.max(Q[s2]) * (not terminated)
            Q[s, a] += alpha * (target - Q[s, a])

            episode_rewards[ep] += r
            s = s2
            done = terminated or truncated            # truncated = hit the step limit

        eps = max(eps_end, eps * eps_decay)           # explore a bit less every game

    return Q, episode_rewards


def evaluate(env, Q, games=1000, seed=12345):
    """Play with the learned policy, NO exploration. Returns (avg reward, avg steps)."""
    env.reset(seed=seed)
    totals, lengths = [], []
    for _ in range(games):
        s, _ = env.reset()
        done, total, steps = False, 0.0, 0
        while not done:
            s, r, terminated, truncated, _ = env.step(int(np.argmax(Q[s])))
            total += r
            steps += 1
            done = terminated or truncated
        totals.append(total)
        lengths.append(steps)
    return float(np.mean(totals)), float(np.mean(lengths))


def moving_average(x, window=100):
    # Smooth a noisy learning curve: average of the last `window` games.
    return np.convolve(x, np.ones(window) / window, mode="valid")
