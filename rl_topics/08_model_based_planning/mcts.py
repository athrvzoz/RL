"""
Monte Carlo Tree Search (MCTS), the plain "UCT" version, in plain Python.

Planning at DECISION time: before every single move, imagine many games
starting from the current board, and play the move that worked best in imagination.

One "simulation" = four steps:
    1. SELECTION       walk down the tree we already built, choosing children with
                       UCB = (wins / visits) + c * sqrt( ln(parent visits) / visits )
                       -- the SAME formula as your UCB bandit! Each node is a little bandit.
    2. EXPANSION       reach a node with an untried move: add ONE new child for it
    3. SIMULATION      from that new child, play random moves until the game ends (a "rollout")
    4. BACKPROPAGATION walk back up to the root: every node on the path gets visits + 1,
                       and wins + 1 (or + 0.5 for a draw) if its player won

After all simulations: play the root's MOST VISITED move.

Also here: puct_search(), the AlphaZero version of the same loop, where a
"network" gives priors P (which moves look promising) and a value (who is
winning), instead of trying moves blindly and finishing with random rollouts.

Used by:  python tictactoe_mcts.py      python mcts_experiments.py
"""
import math
import random


class Node:
    def __init__(self, game, state, parent=None, move=None, prior=1.0):
        self.state, self.parent, self.move = state, parent, move
        self.children = []
        self.untried = [] if game.winner(state) else game.legal_moves(state)
        # The player who made the move INTO this node. W counts results for THAT player,
        # because that is the player who picks this child when standing at the parent.
        self.just_moved = "O" if game.to_move(state) == "X" else "X"
        self.N = 0            # visits
        self.W = 0.0          # wins for just_moved (draw = 0.5)
        self.P = prior        # prior from a "network" (only used by puct_search)


def ucb_score(child, parent_N, c):
    exploit = child.W / child.N                                 # average result so far
    explore = c * math.sqrt(math.log(parent_N) / child.N)       # bonus for "not tried much"
    return exploit + explore


def rollout(game, s, rng):
    # The model is game.play(): we KNOW the rules, so we can imagine whole games.
    while (w := game.winner(s)) is None:
        s = game.play(s, rng.choice(game.legal_moves(s)))
    return w


def score_for(player, result):
    return 1.0 if result == player else 0.5 if result == "draw" else 0.0


def search(game, root_state, n_sims, c=1.4, seed=0, log=None):
    """Run n_sims simulations from root_state. Returns the root node (with the whole tree)."""
    rng = random.Random(seed)
    root = Node(game, root_state)
    for _ in range(n_sims):
        node = root
        # 1. SELECTION: while every move here has been tried, follow the best UCB child
        while not node.untried and node.children:
            node = max(node.children, key=lambda ch: ucb_score(ch, node.N, c))
        # 2. EXPANSION: add one new child (unless the game is already over here)
        if node.untried:
            m = node.untried.pop(rng.randrange(len(node.untried)))
            child = Node(game, game.play(node.state, m), parent=node, move=m)
            node.children.append(child)
            node = child
        # 3. SIMULATION: random moves to the end of the game
        result = rollout(game, node.state, rng)
        if log is not None:
            log.append((path_of(node), result))
        # 4. BACKPROPAGATION: update every node on the path back to the root
        while node is not None:
            node.N += 1
            node.W += score_for(node.just_moved, result)
            node = node.parent
    return root


def path_of(node):
    moves = []
    while node.parent is not None:
        moves.append(node.move)
        node = node.parent
    return moves[::-1]


def best_move(root):
    # The most VISITED child, not the highest win rate: visits are the more reliable vote.
    return max(root.children, key=lambda ch: ch.N).move


# ───────────────────────── AlphaZero-style search (PUCT) ─────────────────────────
def puct_search(game, root_state, n_sims, prior_fn, value_fn=None, c_puct=1.5, seed=0):
    """
    Same loop, two changes (this is the AlphaZero idea):
      * a leaf is expanded with ALL its children at once, each with a prior P from prior_fn
      * the leaf is scored by value_fn (a "value network") instead of a random rollout
        (value_fn=None: still use one random rollout)
    Selection uses PUCT:  Q + c_puct * P * sqrt(parent visits) / (1 + child visits)
    """
    rng = random.Random(seed)
    root = Node(game, root_state)
    for _ in range(n_sims):
        node = root
        while node.children:                                   # SELECTION
            sq = math.sqrt(node.N)
            node = max(node.children, key=lambda ch: (ch.W / ch.N if ch.N else 0.5)
                       + c_puct * ch.P * sq / (1 + ch.N))
        result = game.winner(node.state)
        if result is not None:                                 # game over: the true result
            leaf_score = score_for(game.to_move(node.state), result)
        else:
            for m, p in prior_fn(node.state).items():          # EXPANSION: all children + priors
                node.children.append(Node(game, game.play(node.state, m), node, m, prior=p))
            if value_fn is None:                               # EVALUATION
                leaf_score = score_for(game.to_move(node.state), rollout(game, node.state, rng))
            else:                                              # value in [-1, 1] -> [0, 1]
                leaf_score = (1 + value_fn(node.state, rng)) / 2
        leaf_player = game.to_move(node.state)                 # leaf_score is for this player
        while node is not None:                                # BACKPROPAGATION
            node.N += 1
            node.W += leaf_score if node.just_moved == leaf_player else 1 - leaf_score
            node = node.parent
    return root


# ───────────────────────── Players and a referee ─────────────────────────
def random_player(game, s, rng):
    return rng.choice(game.legal_moves(s))


def mcts_player(n_sims, c=1.4):
    def choose(game, s, rng):
        return best_move(search(game, s, n_sims, c, seed=rng.randrange(1 << 30)))
    return choose


def play_game(game, player_x, player_o, rng):
    s = game.start
    while game.winner(s) is None:
        player = player_x if game.to_move(s) == "X" else player_o
        s = game.play(s, player(game, s, rng))
    return game.winner(s)


def match(game, player_x, player_o, games, seed=0):
    """Returns how many games X won, drew, and O won."""
    rng = random.Random(seed)
    results = [play_game(game, player_x, player_o, rng) for _ in range(games)]
    return results.count("X"), results.count("draw"), results.count("O")
