"""
The gridworld from the playground (RL From Zero, chapters 3 and 7).

    . . . . G       S = start
    . # . . P       G = goal  (+1, game over)
    . # . # .       P = pit   (-1, game over)
    S . . . .       # = wall  (can't enter; bumping leaves you in place)

Unlike the casino, this world HAS states: every open square is one.
Your action changes which square you're on next, so rewards in the future
depend on what you do now. That's what the Bellman equation is for.

This class is the full MDP rule book (S, A, P, R): value iteration is
allowed to read it. In later projects the agent won't get this.
"""

LAYOUT = [
    "....G",
    ".#..P",
    ".#.#.",
    "S....",
]

# action id -> (name, row change, column change, arrow)
ACTIONS = {
    0: ("up",    -1,  0, "↑"),
    1: ("right",  0,  1, "→"),
    2: ("down",   1,  0, "↓"),
    3: ("left",   0, -1, "←"),
}


class GridWorld:
    def __init__(self, layout=LAYOUT, slip=0.0, step_reward=-0.04):
        """
        slip: chance of sliding sideways instead of going where you aimed.
              slip=0.2 means 80% intended direction, 10% each perpendicular side.
        step_reward: small cost for every ordinary step, so wandering is bad.
        """
        self.layout = layout
        self.rows, self.cols = len(layout), len(layout[0])
        self.slip = slip
        self.step_reward = step_reward
        # Every square that isn't a wall is a state.
        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)
                       if layout[r][c] != "#"]
        self.start = next(s for s in self.states if self.kind(s) == "S")

    def kind(self, s):
        r, c = s
        return self.layout[r][c]

    def is_terminal(self, s):
        return self.kind(s) in "GP"

    def reward(self, s2):
        # Reward for ARRIVING in square s2.
        return {"G": 1.0, "P": -1.0}.get(self.kind(s2), self.step_reward)

    def _move(self, s, a):
        _, dr, dc, _ = ACTIONS[a]
        r, c = s[0] + dr, s[1] + dc
        if not (0 <= r < self.rows and 0 <= c < self.cols) or self.layout[r][c] == "#":
            return s                       # bumped into edge or wall: stay put
        return (r, c)

    def transitions(self, s, a):
        """
        P(s' | s, a) and R as a list of (probability, next_state, reward).
        This is the "world rolls dice" layer of the backup diagram.
        """
        if self.slip == 0:
            outcomes = [(1.0, a)]
        else:
            left_of, right_of = (a - 1) % 4, (a + 1) % 4   # the two perpendicular directions
            outcomes = [(1 - self.slip, a), (self.slip / 2, left_of), (self.slip / 2, right_of)]
        result = []
        for p, actual in outcomes:
            s2 = self._move(s, actual)
            result.append((p, s2, self.reward(s2)))
        return result
