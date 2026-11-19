"""
Monte Carlo methods for small games (RL From Zero, topic 2).

The one idea: PLAY A WHOLE GAME FIRST. Only when it has ended do we know the
real total reward that followed every state we visited. Then walk BACKWARDS
through the game and update every visited (state, action) towards that real total.

    Q-learning (project 3):  update after EVERY step, using a guess:  r + γ·max Q[s2]
    Monte Carlo (this file): update after the WHOLE game, using the real return G

No guesses of other states are used (no "bootstrapping"), so MC can only be
used when games END.

States here are tuples, e.g. Blackjack's (player_sum, dealer_card, usable_ace),
so the tables are Python dicts instead of numpy arrays.
"""
from collections import defaultdict

import numpy as np

STICK, HIT = 0, 1           # Blackjack's two actions


def as_state(obs):
    # Gymnasium hands back numpy integers; plain ints make nicer dict keys and prints.
    return tuple(int(x) for x in obs) if isinstance(obs, tuple) else int(obs)


def play_episode(env, policy, rng):
    """Play ONE complete game. Nothing is learned here, we only RECORD what happened.

    Returns a list of (state, action, reward) for every step:
        reward = what env.step() gave right AFTER taking that action.
    """
    episode = []
    s = as_state(env.reset()[0])
    done = False
    while not done:
        a = policy(s, rng)
        obs, r, terminated, truncated, _ = env.step(a)
        episode.append((s, a, r))
        s = as_state(obs)
        done = terminated or truncated
    return episode


def returns_backwards(episode, gamma):
    """The return G for every step, computed by walking BACKWARDS from the end.

        G_last = r_last
        G_t    = r_t + γ · G_(t+1)          (one multiply-and-add per step)
    """
    G, out = 0.0, []
    for s, a, r in reversed(episode):
        G = r + gamma * G
        out.append(G)
    return out[::-1]                         # back into normal time order


def first_visits(keys):
    # Index of the FIRST time each key appears in the game.
    first = {}
    for t, k in enumerate(keys):
        first.setdefault(k, t)
    return first


def mc_prediction(env, policy, episodes, gamma=1.0, first_visit=True, seed=0):
    """Estimate V(s) for a FIXED policy by averaging real returns.

    V[s] = (sum of returns seen from s) / (number of times we counted s)
    written incrementally, exactly like the bandit:  V += (G - V) / N
    """
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    V = defaultdict(float)
    N = defaultdict(int)
    for _ in range(episodes):
        episode = play_episode(env, policy, rng)          # 1. play to the END
        first = first_visits([s for s, _, _ in episode])
        G = 0.0
        for t in reversed(range(len(episode))):           # 2. walk BACKWARDS
            s, a, r = episode[t]
            G = r + gamma * G                             #    real return from step t
            if first_visit and first[s] != t:
                continue                                  #    first-visit: count s once per game
            N[s] += 1
            V[s] += (G - V[s]) / N[s]                     # 3. running average (bandit rule)
    return V, N


def epsilon_greedy_policy(Q, eps, n_actions=2):
    """Make a policy function from a Q-table: random with chance eps, else best Q."""
    def policy(s, rng):
        if rng.random() < eps:
            return int(rng.integers(n_actions))          # explore
        q = Q[s]
        best = np.flatnonzero(q == q.max())
        return int(rng.choice(best))                     # exploit, ties broken randomly
    return policy


def mc_control(env, episodes, eps=0.1, gamma=1.0, seed=0, exploring_starts=False):
    """On-policy first-visit MC control with an epsilon-greedy policy.

    Every game:  play with ε-greedy(Q)  ->  walk backwards  ->  Q[s,a] += (G - Q[s,a]) / N[s,a]
    The policy improves automatically, because it always reads the latest Q.

    exploring_starts=True: the FIRST action of every game is random, after that the agent
    is fully greedy (eps is ignored). Blackjack's start states are already random.
    Returns Q, N and the final reward of every training game.
    """
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    Q = defaultdict(lambda: np.zeros(2))
    N = defaultdict(lambda: np.zeros(2))
    game_rewards = np.zeros(episodes)
    explore = epsilon_greedy_policy(Q, 0.0 if exploring_starts else eps)

    for ep in range(episodes):
        if exploring_starts:
            first_step = [True]

            def policy(s, rng):
                if first_step[0]:
                    first_step[0] = False
                    return int(rng.integers(2))           # random first action
                return explore(s, rng)
        else:
            policy = explore
        episode = play_episode(env, policy, rng)          # 1. play the WHOLE game, learn nothing yet
        game_rewards[ep] = sum(r for _, _, r in episode)

        first = first_visits([(s, a) for s, a, _ in episode])
        G = 0.0
        for t in reversed(range(len(episode))):           # 2. game over: walk backwards
            s, a, r = episode[t]
            G = r + gamma * G
            if first[(s, a)] != t:
                continue
            N[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / N[s][a]            # 3. average of real returns
    return Q, N, game_rewards


def greedy(Q):
    """The final policy: always the action with the biggest Q (no exploration)."""
    def policy(s, rng):
        return int(np.argmax(Q[s])) if s in Q else HIT
    return policy


def stick_on_20(s, rng):
    # The "simple" policy: hit until your sum is 20 or 21, then stick.
    return STICK if s[0] >= 20 else HIT


def stick_on_17(s, rng):
    # Copy the dealer: hit until 17 or more.
    return STICK if s[0] >= 17 else HIT


def random_policy(s, rng):
    return int(rng.integers(2))


def win_draw_loss(env, policy, games=100_000, seed=12345):
    """Play many games with NO learning. Returns (win %, draw %, loss %, average reward)."""
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    finals = np.array([sum(r for _, _, r in play_episode(env, policy, rng)) for _ in range(games)])
    return (100 * np.mean(finals > 0), 100 * np.mean(finals == 0),
            100 * np.mean(finals < 0), float(finals.mean()))


def moving_average(x, window=1000):
    return np.convolve(x, np.ones(window) / window, mode="valid")
