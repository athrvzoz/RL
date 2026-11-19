"""
The casino: a k-armed bandit (RL From Zero, chapter 10).

There are k slot machines ("arms"). Each one has a hidden true average
payout q*(a). A single pull pays that average plus random noise, so one
pull tells you very little; only many pulls reveal which machine is best.

There is only ONE state (you're always standing in front of the same
machines), so there is no Bellman equation here, just the pure
explore-vs-exploit problem.
"""
import numpy as np


class SlotMachineCasino:
    def __init__(self, k=10, rng=None):
        self.k = k
        self.rng = rng if rng is not None else np.random.default_rng()
        # Hidden true averages q*(a), drawn from a bell curve N(0, 1).
        # The agent never sees these numbers.
        self.true_means = self.rng.normal(0.0, 1.0, size=k)
        self.best_arm = int(np.argmax(self.true_means))

    def pull(self, arm):
        # Reward = true average + noise from N(0, 1).
        return self.rng.normal(self.true_means[arm], 1.0)
