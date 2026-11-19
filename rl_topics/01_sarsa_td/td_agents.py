"""
SARSA, Expected SARSA and Q-learning in ONE small file (RL From Zero, TD chapter).

All three are "temporal-difference" (TD) methods. They share everything:
the Q-table, the epsilon-greedy choice, the "nudge by alpha" update.
They differ in ONE line only: what number they use for "the future" in the target.

    Q-learning      future = max Q[s2]                 (the BEST action in s2)
    SARSA           future = Q[s2, a2]                 (the action we will REALLY take next)
    Expected SARSA  future = sum of pi(a|s2) * Q[s2,a] (the AVERAGE over what we might take)

The name SARSA is just the five things one update needs:
    S, A, R, S', A'   =   s, a, r, s2, a2
"""
import numpy as np


def policy_probs(Q, s, eps):
    """
    The agent's OWN action probabilities in state s under epsilon-greedy.
    (Not the world's probabilities! These describe the agent's dice.)

    Every action gets eps / n_actions (the "explore" share).
    The best action(s) also share the remaining 1 - eps (the "exploit" share).
    """
    n = Q.shape[1]
    probs = np.full(n, eps / n)
    best = np.flatnonzero(Q[s] == Q[s].max())
    probs[best] += (1 - eps) / len(best)
    return probs


def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))         # explore: random action
    best = np.flatnonzero(Q[s] == Q[s].max())         # exploit: best action,
    return int(rng.choice(best))                      # ties broken randomly


def future_value(method, Q, s2, a2, eps):
    """The ONLY line where the three methods differ."""
    if method == "sarsa":
        return Q[s2, a2]                              # what I will actually do next
    if method == "expected_sarsa":
        return np.dot(policy_probs(Q, s2, eps), Q[s2])  # average over my own dice
    if method == "qlearning":
        return np.max(Q[s2])                          # the best I could do next
    raise ValueError(method)


def train(env, method, episodes=500, alpha=0.5, gamma=1.0, eps=0.1,
          eps_decay=1.0, eps_end=0.0, seed=0):
    """
    Returns the Q-table plus, for every training game: total reward, number of
    steps, and how many times the agent fell off the cliff (reward -100).

    eps_decay = 1.0 means epsilon stays fixed (the classic cliff experiment).
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    falls = np.zeros(episodes, dtype=int)
    env.reset(seed=seed)

    for ep in range(episodes):
        s, _ = env.reset()
        a = epsilon_greedy(Q, s, eps, rng)            # SARSA needs the first action before the loop
        done = False
        while not done:
            s2, r, terminated, truncated, _ = env.step(a)

            # Choose the NEXT action now, with the same epsilon-greedy rule.
            # SARSA puts it into the target; all three methods then really take it.
            a2 = epsilon_greedy(Q, s2, eps, rng)

            # TD target = reward now + gamma * (an estimate of the future from s2).
            # If the game ENDED (reached the goal) there is no future: target = r.
            target = r + gamma * future_value(method, Q, s2, a2, eps) * (not terminated)
            td_error = target - Q[s, a]               # the "surprise", called delta
            Q[s, a] += alpha * td_error               # update the state we LEFT

            returns[ep] += r
            lengths[ep] += 1
            falls[ep] += (r == -100)
            s, a = s2, a2                             # the next action is already chosen
            done = terminated or truncated

        eps = max(eps_end, eps * eps_decay)
    return Q, returns, lengths, falls


def greedy_path(env, Q, max_steps=100):
    """Play ONE game with no exploration. Returns (total reward, list of states visited)."""
    s, _ = env.reset()
    path, total = [s], 0
    for _ in range(max_steps):                        # cap, in case the policy walks in circles
        s, r, terminated, truncated, _ = env.step(int(np.argmax(Q[s])))
        total += r
        path.append(s)
        if terminated or truncated:
            break
    return total, path


def evaluate_with_eps(env, Q, eps, games=1000, seed=12345, max_steps=500):
    """Play with the learned Q but KEEP exploring with probability eps (no learning).
    Returns (average reward, average cliff falls per game)."""
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    totals, falls = [], []
    for _ in range(games):
        s, _ = env.reset()
        total, fell = 0, 0
        for _ in range(max_steps):
            s, r, terminated, truncated, _ = env.step(epsilon_greedy(Q, s, eps, rng))
            total += r
            fell += (r == -100)
            if terminated or truncated:
                break
        totals.append(total)
        falls.append(fell)
    return float(np.mean(totals)), float(np.mean(falls))


def moving_average(x, window=10):
    # Smooth a noisy learning curve: average of the last `window` games.
    return np.convolve(x, np.ones(window) / window, mode="valid")
