"""
The players: epsilon-greedy and UCB (RL From Zero, chapter 10).

Both keep the same two lists:
    Q[a] = estimated average payout of machine a
    N[a] = how many times machine a has been pulled
and both learn with the same update. They differ only in HOW they choose.
"""
import numpy as np


def argmax_random_tie(values, rng):
    # np.argmax always returns the FIRST maximum. At the start every Q is 0,
    # so a greedy agent would always pick machine 0. Break ties randomly.
    best = np.flatnonzero(values == values.max())
    return int(rng.choice(best))


class Agent:
    def __init__(self, k, rng):
        self.k = k
        self.rng = rng
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def update(self, arm, reward):
        # The pattern from chapter 9:
        #   NewEstimate = Old + step * (Target - Old)
        # With step = 1/N this is exactly a running average of all
        # rewards seen from this machine.
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


class EpsilonGreedyAgent(Agent):
    """With probability epsilon pick a random machine, otherwise the best-looking one."""

    def __init__(self, k, epsilon, rng):
        super().__init__(k, rng)
        self.epsilon = epsilon

    def choose(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))       # explore
        return argmax_random_tie(self.Q, self.rng)      # exploit


class UCBAgent(Agent):
    """
    Upper Confidence Bound:
        a_t = argmax_a [ Q(a) + c * sqrt( ln t / N(a) ) ]
    Machines tried only a few times get a big bonus ("give it a chance").
    The bonus shrinks as N(a) grows, so exploration fades on its own.
    """

    def __init__(self, k, c, rng):
        super().__init__(k, rng)
        self.c = c
        self.t = 0

    def choose(self):
        self.t += 1
        # N(a) = 0 makes the bonus infinite, so try every machine once first.
        untried = np.flatnonzero(self.N == 0)
        if untried.size:
            return int(untried[0])
        bonus = self.c * np.sqrt(np.log(self.t) / self.N)
        return argmax_random_tie(self.Q + bonus, self.rng)
