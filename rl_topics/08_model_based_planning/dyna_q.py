"""
Dyna-Q: Q-learning that also LEARNS A MODEL and PLANS with it.

Plain Q-learning (project 3) uses every real step exactly once, then throws it away.
Dyna-Q keeps a notebook (the "model") of what happened:

    model[(s, a)] = (r, s2, done)     "last time I did a in s, I got r and landed in s2"

After every REAL step it does three things:
    (a) direct RL   : the normal Q-learning update with the real step
    (b) model update: write the step into the notebook
    (c) planning    : n times, pick a remembered (s, a) at random, ask the notebook
                      what happens, and do the SAME Q-learning update on that
                      imagined step. No new game moves needed.

So one real step becomes 1 + n updates. The notebook is a rule book the agent
wrote ITSELF from experience (value iteration was GIVEN its rule book).

This file has the maze world and the agent. The scripts that use it:
    python dyna_maze.py           (the main demo: n = 0, 5, 50)
    python dyna_experiments.py    (more planning-step values + a maze that changes)
"""
import math
from collections import deque

import numpy as np

# The maze from Sutton & Barto, "Reinforcement Learning: An Introduction", figure 8.2.
#   S = start, G = goal (+1, game over), # = wall. Every other move gives reward 0.
DYNA_MAZE = [
    ".......#G",
    "..#....#.",
    "S.#....#.",
    "..#......",
    ".....#...",
    ".........",
]

# action id -> (name, row change, column change, arrow)
ACTIONS = {
    0: ("up",    -1,  0, "↑"),
    1: ("right",  0,  1, "→"),
    2: ("down",   1,  0, "↓"),
    3: ("left",   0, -1, "←"),
}


class Maze:
    """A small deterministic gridworld. States are numbered row * cols + col."""

    def __init__(self, layout=DYNA_MAZE):
        self.set_layout(layout)

    def set_layout(self, layout):
        # Used again by the "changing maze" experiment: the walls move mid-game.
        self.layout = layout
        self.rows, self.cols = len(layout), len(layout[0])
        self.n_states = self.rows * self.cols
        for r, line in enumerate(layout):
            for c, ch in enumerate(line):
                if ch == "S":
                    self.start = r * self.cols + c
                if ch == "G":
                    self.goal = r * self.cols + c

    def is_wall(self, r, c):
        return not (0 <= r < self.rows and 0 <= c < self.cols) or self.layout[r][c] == "#"

    def step(self, s, a):
        """The real world. Returns (s2, reward, done). The agent never reads this code."""
        r, c = divmod(s, self.cols)
        _, dr, dc, _ = ACTIONS[a]
        if not self.is_wall(r + dr, c + dc):          # bump a wall or edge: stay put
            r, c = r + dr, c + dc
        s2 = r * self.cols + c
        if s2 == self.goal:
            return s2, 1.0, True
        return s2, 0.0, False

    def shortest_path(self):
        """Breadth-first search, only to print 'the best possible' for comparison."""
        dist = {self.start: 0}
        todo = deque([self.start])
        while todo:
            s = todo.popleft()
            for a in ACTIONS:
                s2, _, _ = self.step(s, a)
                if s2 not in dist:
                    dist[s2] = dist[s] + 1
                    todo.append(s2)
        return dist[self.goal]


class DynaQ:
    def __init__(self, n_states, n_planning=5, alpha=0.1, gamma=0.95, eps=0.1,
                 kappa=0.0, seed=0, n_actions=4):
        """
        n_planning: n, how many imagined updates after each real step (n = 0 is plain Q-learning)
        kappa:      0 = Dyna-Q. > 0 = Dyna-Q+ (adds a small bonus to moves not tried for a
                    long time, so the agent goes back and checks if the world has changed)
        """
        self.Q = np.zeros((n_states, n_actions))
        self.n_planning, self.alpha, self.gamma, self.eps, self.kappa = n_planning, alpha, gamma, eps, kappa
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.model = {}         # (s, a) -> (r, s2, done): the notebook, written from experience
        self.seen = []          # list of the notebook's keys, so we can pick one at random fast
        self.last_tried = {}    # (s, a) -> real time step it was last really tried (Dyna-Q+)
        self.t = 0              # real steps taken so far
        self.updates = 0        # total Q-updates (real + imagined), just for counting

    def act(self, s):
        # epsilon-greedy, ties broken randomly (exactly like q_learning.py)
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        best = np.flatnonzero(self.Q[s] == self.Q[s].max())
        return int(self.rng.choice(best))

    def q_update(self, s, a, r, s2, done):
        # The same Bellman nudge as Q-learning. Returns the numbers so we can print a trace.
        old = self.Q[s, a]
        target = r + self.gamma * np.max(self.Q[s2]) * (not done)
        self.Q[s, a] = old + self.alpha * (target - old)
        self.updates += 1
        return old, target, self.Q[s, a]

    def remember(self, s, a, r, s2, done):
        if self.kappa > 0 and not any((s, b) in self.model for b in range(self.n_actions)):
            # Dyna-Q+ only: a brand-new square. Pretend every untried action "stays here,
            # reward 0", so planning can still give them a curiosity bonus.
            for b in range(self.n_actions):
                self.model[(s, b)] = (0.0, s, False)
                self.seen.append((s, b))
                self.last_tried[(s, b)] = 0
        if (s, a) not in self.model:
            self.seen.append((s, a))
        self.model[(s, a)] = (r, s2, done)            # overwrite: the world is deterministic
        self.last_tried[(s, a)] = self.t

    def plan_once(self):
        # Pick a (state, action) we have REALLY tried before, at random...
        s, a = self.seen[self.rng.integers(len(self.seen))]
        # ...ask the notebook what happens (no real move!)...
        r, s2, done = self.model[(s, a)]
        if self.kappa > 0:                            # Dyna-Q+: bonus for "not checked lately"
            r += self.kappa * math.sqrt(self.t - self.last_tried[(s, a)])
        # ...and learn from that imagined step as if it were real.
        return (s, a, r, s2) + self.q_update(s, a, r, s2, done)

    def learn(self, s, a, r, s2, done):
        """Called once per REAL step. Returns the list of planning updates (for traces)."""
        self.t += 1
        self.q_update(s, a, r, s2, done)              # (a) direct RL
        self.remember(s, a, r, s2, done)              # (b) model learning
        return [self.plan_once() for _ in range(self.n_planning)]   # (c) planning


def run_episodes(maze, agent, episodes, max_steps=100_000):
    """Play whole games. Returns the number of real steps each game took."""
    steps = []
    for _ in range(episodes):
        s, done, k = maze.start, False, 0
        while not done and k < max_steps:
            a = agent.act(s)
            s2, r, done = maze.step(s, a)
            agent.learn(s, a, r, s2, done)
            s, k = s2, k + 1
        steps.append(k)
    return steps
