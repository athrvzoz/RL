"""
Project 3c: YOU play FrozenLake and Taxi, with a trained Q-learning agent as your coach.

Run:
    python play.py                    # menu
    python play.py slippery           # jump straight in: frozen | slippery | taxi
    python play.py taxi --terminal    # no window: type a key + Enter

Controls (in the window, click it first; in the terminal, type the letter + Enter):
    arrow keys / W A S D    move
    P / O                   pick up / drop off the passenger (Taxi only)
    H                       hint: show the coach's Q-values for where you are
    C                       let the coach make this one move
    R                       restart the game        Esc / Q   quit
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import gymnasium as gym
import numpy as np

from frozen_lake import MAP
from q_learning import evaluate, train
from taxi import ACTION_NAMES as TAXI_NAMES, describe, strip_colors

MOVE_KEYS = {"up", "down", "left", "right"}
GAMES = {
    "frozen": dict(
        title="FrozenLake (normal ice)", id="FrozenLake-v1", kwargs=dict(is_slippery=False),
        train=dict(episodes=2000, alpha=0.1, eps_decay=0.995),
        names=["left", "down", "right", "up"], keys={"left": 0, "down": 1, "right": 2, "up": 3},
        rules="Walk from S to G without falling in a hole (H). You go exactly where you aim."),
    "slippery": dict(
        title="FrozenLake (SLIPPERY ice)", id="FrozenLake-v1", kwargs=dict(is_slippery=True),
        train=dict(episodes=20000, alpha=0.05, eps_decay=0.9995),
        names=["left", "down", "right", "up"], keys={"left": 0, "down": 1, "right": 2, "up": 3},
        rules="Same lake, but you only go where you aim 1/3 of the time. The rest you slide sideways!"),
    "taxi": dict(
        title="Taxi", id="Taxi-v4", kwargs={},
        train=dict(episodes=5000, alpha=0.1, eps_decay=0.999),
        names=TAXI_NAMES, keys={"down": 0, "up": 1, "right": 2, "left": 3, "p": 4, "o": 5},
        rules="Drive to the passenger, press P to pick up, drive to the destination, press O to drop off.\n"
              "  Every move costs -1, delivery gives +20, a wrong pickup/drop-off costs -10."),
}


# ── reading your keys ────────────────────────────────────────────────────
def read_key_window():
    import pygame
    names = {pygame.K_UP: "up", pygame.K_w: "up", pygame.K_DOWN: "down", pygame.K_s: "down",
             pygame.K_LEFT: "left", pygame.K_a: "left", pygame.K_RIGHT: "right", pygame.K_d: "right",
             pygame.K_p: "p", pygame.K_o: "o", pygame.K_h: "h", pygame.K_c: "c", pygame.K_r: "r",
             pygame.K_ESCAPE: "quit", pygame.K_q: "quit"}
    while True:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN and event.key in names:
            return names[event.key]


TERMINAL_KEYS = {"w": "up", "a": "left", "s": "down", "d": "right", "p": "p", "o": "o",
                 "h": "h", "c": "c", "r": "r", "q": "quit"}


def read_key_terminal():
    while True:
        k = input("  your move> ").strip().lower()[:1]
        if k in TERMINAL_KEYS:
            return TERMINAL_KEYS[k]
        print("  keys: w a s d move, p pick up, o drop off, h hint, c coach moves, r restart, q quit")


# ── showing the board ────────────────────────────────────────────────────
def show_board(game, env, s):
    """Terminal mode only; in window mode the picture updates by itself."""
    if game == "taxi":
        print(strip_colors(env.render()))
        print(f"  {describe(env, s)}")
    else:
        for r in range(4):
            print("     " + " ".join("(☺)" if r * 4 + c == s else f" {MAP[r][c]} " for c in range(4)))
        print()


def set_caption(text):
    import pygame
    pygame.display.set_caption(text)


# ── what just happened, in words ─────────────────────────────────────────
def explain_frozen(names, s, s2, a, r, terminated):
    dr, dc = s2 // 4 - s // 4, s2 % 4 - s % 4
    went = {(0, -1): "left", (1, 0): "down", (0, 1): "right", (-1, 0): "up"}.get((dr, dc))
    if went is None:
        text = "bumped the edge, stayed put"
    elif went == names[a]:
        text = f"moved {went}"
    else:
        text = f"SLID {went.upper()} on the ice!"
    if terminated:
        text += "   GOAL! +1" if r == 1 else "   fell in a HOLE, game over"
    return text


def explain_taxi(r, terminated):
    if terminated:
        return "delivered the passenger! +20"
    return "illegal pickup / drop-off, -10" if r == -10 else f"{r:+}"


def hint(Q, s, names):
    if not Q[s].any():
        print("  Coach: I never visited this spot during training, I have no idea!")
        return
    order = np.argsort(-Q[s])
    print("  Coach's Q-values here (higher = better):  "
          + "   ".join(f"{names[a]} {Q[s, a]:+.2f}" for a in order))
    print(f"  Coach says: {names[order[0]].upper()}")


# ── the game loop ────────────────────────────────────────────────────────
def play(game, use_window):
    spec = GAMES[game]
    names = spec["names"]
    print(f"\nTraining your AI coach on {spec['title']} (a few seconds)...")
    coach_env = gym.make(spec["id"], **spec["kwargs"])
    Q, _ = train(coach_env, **spec["train"])
    coach_score, coach_steps = evaluate(coach_env, Q)
    print(f"Coach is ready. Over 1000 test games it scores {coach_score:.2f} on average "
          f"in {coach_steps:.1f} steps.\n")
    print(f"  {spec['rules']}")
    print("  Keys: arrows/WASD move" + (", P pick up, O drop off" if game == "taxi" else "")
          + ", H hint, C coach moves, R restart, Esc/Q quit")
    if use_window:
        print("  >>> Click the game window, then use your keyboard there. Messages appear here. <<<")

    env = gym.make(spec["id"], render_mode="human" if use_window else "ansi", **spec["kwargs"])
    read_key = read_key_window if use_window else read_key_terminal
    scores = []

    while True:
        s, _ = env.reset()
        total, steps, done, quit_now = 0, 0, False, False
        print(f"\n─── Game {len(scores) + 1} ───")
        if use_window:
            set_caption(f"{spec['title']}  |  game {len(scores) + 1}")
        else:
            show_board(game, env, s)

        while not done:
            key = read_key()
            if key == "quit":
                quit_now = True
                break
            if key == "r":
                print("  (restarted)")
                break
            if key == "h":
                hint(Q, s, names)
                continue
            if key == "c":
                a, who = int(np.argmax(Q[s])), "Coach"
            elif key in spec["keys"]:
                a, who = spec["keys"][key], "You"
            else:
                continue                                  # e.g. P pressed in FrozenLake

            s2, r, terminated, truncated, _ = env.step(a)
            total += r
            steps += 1
            what = explain_frozen(names, s, s2, a, r, terminated) if game != "taxi" else explain_taxi(r, terminated)
            print(f"  step {steps:>3}: {who} chose {names[a]:<8} -> {what}")
            s = s2
            done = terminated or truncated
            if truncated:
                print("  Time's up! The game stops itself after "
                      f"{steps} steps.")
            if use_window:
                set_caption(f"{spec['title']}  |  steps {steps}  |  score {total:+g}")
            else:
                show_board(game, env, s)

        if quit_now:
            break
        if done:
            scores.append(total)
            print(f"  Game over: score {total:+g} in {steps} steps.")
            print("  Press any move key to play again, Esc/Q to quit.")
            if read_key() == "quit":
                break

    env.close()
    print("\n═══ Final scoreboard ═══")
    if not scores:
        print("  You didn't finish a game. Come back soon!")
    elif game == "taxi":
        print(f"  You:   average score {np.mean(scores):+.1f} over {len(scores)} game(s)")
        print(f"  Coach: average score {coach_score:+.1f} over 1000 games")
    else:
        wins = int(sum(scores))
        print(f"  You:   reached the goal {wins} of {len(scores)} time(s) ({100 * wins / len(scores):.0f}%)")
        print(f"  Coach: reaches the goal {100 * coach_score:.0f}% of the time (1000 games)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("game", nargs="?", choices=list(GAMES))
    parser.add_argument("--terminal", action="store_true", help="no window, type keys + Enter")
    args = parser.parse_args()

    game = args.game
    if game is None:
        print("Which game?\n  1  FrozenLake, normal ice\n  2  FrozenLake, SLIPPERY ice\n  3  Taxi")
        game = {"1": "frozen", "2": "slippery", "3": "taxi"}.get(input("Pick 1, 2 or 3: ").strip(), "frozen")
    play(game, use_window=not args.terminal)


if __name__ == "__main__":
    main()
