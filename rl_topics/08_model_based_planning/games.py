"""
Two board games for the MCTS demos, plus a perfect tic-tac-toe solver (minimax).

Every game here has the same 5 little functions, so mcts.py works on both:
    start                 the empty board
    to_move(s)            whose turn: "X" or "O"
    legal_moves(s)        list of moves you may play
    play(s, m)            the board after playing move m   (this IS the rule book / model)
    winner(s)             "X", "O", "draw", or None if the game is still going

A state is just a string, so it can be copied, compared and used as a dict key.
"""
from functools import lru_cache

# ───────────────────────── Tic-tac-toe ─────────────────────────
#   squares are numbered   0 1 2
#                          3 4 5
#                          6 7 8
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


class TicTacToe:
    name = "tic-tac-toe"
    start = "." * 9

    @staticmethod
    def to_move(s):
        return "X" if s.count("X") == s.count("O") else "O"

    @staticmethod
    def legal_moves(s):
        return [i for i, ch in enumerate(s) if ch == "."]

    @staticmethod
    def play(s, m):
        return s[:m] + TicTacToe.to_move(s) + s[m + 1:]

    @staticmethod
    def winner(s):
        for a, b, c in LINES:
            if s[a] != "." and s[a] == s[b] == s[c]:
                return s[a]
        return "draw" if "." not in s else None

    @staticmethod
    def show(s, indent="     "):
        return "\n".join(indent + " ".join(s[r * 3:r * 3 + 3]) for r in range(3))


# ───────────────────────── Minimax: the perfect answer key ─────────────────────────
@lru_cache(maxsize=None)
def minimax_value(s):
    """+1 = the player to move WINS with perfect play, 0 = draw, -1 = loses."""
    w = TicTacToe.winner(s)
    if w == "draw":
        return 0
    if w is not None:
        return -1                      # the player who just moved won, so the player to move lost
    # My value = the best of (minus the opponent's value after each of my moves)
    return max(-minimax_value(TicTacToe.play(s, m)) for m in TicTacToe.legal_moves(s))


def perfect_moves(s):
    """All moves that keep the best possible result (there can be several)."""
    best = minimax_value(s)
    return [m for m in TicTacToe.legal_moves(s) if -minimax_value(TicTacToe.play(s, m)) == best]


def count_full_tree(s=TicTacToe.start):
    """Plain minimax WITHOUT memory visits every node of the game tree. Count them."""
    if TicTacToe.winner(s) is not None:
        return 1, 1                    # (nodes, finished games)
    nodes, games = 1, 0
    for m in TicTacToe.legal_moves(s):
        n, g = count_full_tree(TicTacToe.play(s, m))
        nodes, games = nodes + n, games + g
    return nodes, games


# ───────────────────────── Connect-4 (6 rows x 7 columns) ─────────────────────────
ROWS, COLS = 6, 7


def _windows():
    # For every cell, all lines of 4 cells that pass through it.
    through = [[] for _ in range(ROWS * COLS)]
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + k * dr, c + k * dc) for k in range(4)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in cells):
                    idx = tuple(rr * COLS + cc for rr, cc in cells)
                    for i in idx:
                        through[i].append(idx)
    return through


WINDOWS = _windows()


class Connect4:
    """State = (board string of 42 cells, top row first; index of the last piece dropped)."""
    name = "Connect-4"
    start = ("." * (ROWS * COLS), -1)

    @staticmethod
    def to_move(s):
        board = s[0]
        return "X" if board.count("X") == board.count("O") else "O"

    @staticmethod
    def legal_moves(s):
        return [c for c in range(COLS) if s[0][c] == "."]      # column not full

    @staticmethod
    def play(s, col):
        board = s[0]
        for r in range(ROWS - 1, -1, -1):                      # the piece falls to the lowest gap
            i = r * COLS + col
            if board[i] == ".":
                return board[:i] + Connect4.to_move(s) + board[i + 1:], i
        raise ValueError("column full")

    @staticmethod
    def winner(s):
        board, last = s
        if last >= 0:                                          # only the last piece can make a new line
            p = board[last]
            for w in WINDOWS[last]:
                if all(board[i] == p for i in w):
                    return p
        return "draw" if "." not in board[:COLS] else None
