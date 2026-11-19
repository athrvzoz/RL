"""
Dyna-Q on the Sutton & Barto maze: does "thinking between moves" save real games?

    . . . . . . . # G       S = start,  G = goal (+1, game over)
    . . # . . . . # .       # = wall (bump: stay put)
    S . # . . . . # .       every other move: reward 0
    . . # . . . . . .       gamma = 0.95, alpha = 0.1, epsilon = 0.1
    . . . . . # . . .
    . . . . . . . . .

Part 1: a 4-square corridor, traced by hand: one real step + 2 planning steps.
Part 2: the maze with n = 0 (plain Q-learning), 5 and 50 planning steps per real step.
        30 runs each, 50 games each. How many real steps does each game take?
Part 3: what the agent "knows" after its FIRST game, with n = 0 vs n = 50.

Run:  python dyna_maze.py            (about 30 seconds)
      python dyna_maze.py --no-show
"""
import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from dyna_q import ACTIONS, DynaQ, Maze, run_episodes

sys.stdout.reconfigure(encoding="utf-8")

ARROW = {a: ACTIONS[a][3] for a in ACTIONS}
NAME = {a: ACTIONS[a][0] for a in ACTIONS}


def part1_trace():
    print("=" * 66)
    print("PART 1 · Corridor  S(0) . (1) . (2) G(3)   alpha 0.5, gamma 0.9")
    print("=" * 66)
    world = Maze(["S..G"])
    RIGHT = 1
    agent = DynaQ(world.n_states, n_planning=0, alpha=0.5, gamma=0.9, seed=12)
    plain = DynaQ(world.n_states, n_planning=0, alpha=0.5, gamma=0.9, seed=3)

    print("\n  Game 1: walk right, right, right (no planning yet)")
    for s in (0, 1, 2):
        s2, r, done = world.step(s, RIGHT)
        agent.learn(s, RIGHT, r, s2, done)
        plain.learn(s, RIGHT, r, s2, done)
        print(f"    real  {s} -right-> {s2}  r={r:.0f}   Q[{s},right] = {agent.Q[s, RIGHT]:.4f}")
    print("  Notebook (model) now holds:",
          ", ".join(f"({s},{NAME[a]})->({r:.0f},{s2})" for (s, a), (r, s2, _) in agent.model.items()))

    print("\n  Game 2, first real step, now WITH n = 2 planning steps:")
    agent.n_planning = 2
    s2, r, done = world.step(0, RIGHT)
    old, best_next = agent.Q[0, RIGHT], np.max(agent.Q[1])
    plans = agent.learn(0, RIGHT, r, s2, done)
    print(f"    REAL  0 -right-> 1  r=0   target = 0 + 0.9 x max Q[1] = 0.9 x {best_next:.4f}"
          f" = {0.9 * best_next:.4f}   Q[0,right]: {old:.4f} -> {old:.4f}")
    for k, (ps, pa, pr, ps2, pold, ptarget, pnew) in enumerate(plans, 1):
        print(f"    PLAN {k}: picked ({ps},{NAME[pa]}) from the notebook -> ({pr:.0f}, {ps2})   "
              f"target = {pr:.0f} + 0.9 x {(ptarget - pr) / 0.9:.4f} = {ptarget:.4f}   "
              f"Q[{ps},{NAME[pa]}]: {pold:.4f} -> {pnew:.4f}")
    print(f"  After ONE real step of game 2: Q[0,right] = {agent.Q[0, RIGHT]:.4f}  (the START already knows)")

    # Plain Q-learning: keep walking right and see when the start square first gets a value.
    game = 1
    while plain.Q[0, RIGHT] == 0:
        game += 1
        for s in (0, 1, 2):
            s2, r, done = world.step(s, RIGHT)
            plain.learn(s, RIGHT, r, s2, done)
    print(f"  Plain Q-learning (n = 0): the start square first gets a value in game {game} "
          f"(Q[0,right] = {plain.Q[0, RIGHT]:.4f})")


def part2_curves(runs=30, episodes=50):
    print("\n" + "=" * 66)
    print(f"PART 2 · The maze: {runs} runs x {episodes} games for each n")
    print("=" * 66)
    maze = Maze()
    print(f"  Shortest possible path: {maze.shortest_path()} steps")
    curves = {}
    for n in (0, 5, 50):
        t0 = time.time()
        all_steps, all_updates = [], []
        for run in range(runs):
            agent = DynaQ(maze.n_states, n_planning=n, seed=run)
            all_steps.append(run_episodes(maze, agent, episodes))
            all_updates.append(agent.updates)
        steps = np.array(all_steps)
        curves[n] = steps.mean(axis=0)
        first_good = next((i + 1 for i, v in enumerate(curves[n]) if v < 20), None)
        print(f"\n  n = {n:>2}:  game 1 = {curves[n][0]:6.1f} steps, game 2 = {curves[n][1]:6.1f}, "
              f"game 5 = {curves[n][4]:5.1f}, game 10 = {curves[n][9]:5.1f}, game 50 = {curves[n][-1]:5.1f}")
        print(f"          first game averaging under 20 steps: game {first_good}")
        print(f"          real steps in all {episodes} games: {steps.sum(axis=1).mean():,.0f}"
              f"   Q-updates (real + imagined): {np.mean(all_updates):,.0f}"
              f"   time per run: {(time.time() - t0) / runs:.2f} s")
    return curves


def known_squares(agent, maze):
    # A square "knows the way" if at least one of its Q-values is above 0.
    return [s for s in range(maze.n_states) if agent.Q[s].max() > 0]


def part3_after_one_game():
    print("\n" + "=" * 66)
    print("PART 3 · What is known after game 1 + the first 20 real steps of game 2 (seed 0)")
    print("=" * 66)
    maze = Maze()
    agents = {}
    for n in (0, 50):
        agent = DynaQ(maze.n_states, n_planning=n, seed=0)
        steps = run_episodes(maze, agent, 1)[0]
        s = maze.start
        for _ in range(20):                            # 20 real steps into game 2
            a = agent.act(s)
            s2, r, done = maze.step(s, a)
            agent.learn(s, a, r, s2, done)
            s = s2
            if done:
                break
        known = known_squares(agent, maze)
        open_squares = sum(ch != "#" for line in maze.layout for ch in line) - 1
        print(f"\n  n = {n:>2}: first game took {steps} steps; squares with a value > 0: "
              f"{len(known)} of {open_squares}")
        for r in range(maze.rows):
            row = []
            for c in range(maze.cols):
                s = r * maze.cols + c
                ch = maze.layout[r][c]
                row.append(ch if ch in "#G" else (ARROW[int(np.argmax(agent.Q[s]))] if s in known else "·"))
            print("     " + " ".join(row))
        agents[n] = agent
    return maze, agents


def draw_maze(ax, maze, agent, title):
    known = set(known_squares(agent, maze))
    for r in range(maze.rows):
        for c in range(maze.cols):
            s, ch = r * maze.cols + c, maze.layout[r][c]
            color = {"#": "#555b6e", "G": "#2e9e68", "S": "#e8ecfb"}.get(ch, "#f5f6fa")
            if ch == "." and s in known:
                color = "#cfe8d9"
            ax.add_patch(plt.Rectangle((c, maze.rows - 1 - r), 1, 1, color=color, ec="white"))
            label = "G" if ch == "G" else (ARROW[int(np.argmax(agent.Q[s]))] if s in known else "")
            if ch != "#":
                ax.text(c + 0.5, maze.rows - 0.5 - r, label, ha="center", va="center", fontsize=13,
                        color="white" if ch == "G" else "black")
    ax.set_xlim(0, maze.cols)
    ax.set_ylim(0, maze.rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")                     # draw straight to the PNG file, no window

    part1_trace()
    curves = part2_curves()
    maze, agents = part3_after_one_game()

    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(2, 1, 1)
    games = np.arange(2, 51)
    for n, color in ((0, "#c2413a"), (5, "#a86a08"), (50, "#2c4bd1")):
        ax.plot(games, curves[n][1:], color=color, lw=2, label=f"n = {n} planning steps")
    ax.axhline(maze.shortest_path(), color="gray", ls="--", lw=1, label="shortest path (14)")
    ax.set_xlabel("Game (episode)")
    ax.set_ylabel("Real steps in that game\n(average of 30 runs)")
    ax.set_ylim(0, 800)
    ax.set_title("Dyna-Q on the maze: more planning = far fewer real steps (game 1 left out)")
    ax.legend()
    ax.grid(alpha=0.3)
    draw_maze(fig.add_subplot(2, 2, 3), maze, agents[0], "n = 0: game 1 + 20 steps of game 2")
    draw_maze(fig.add_subplot(2, 2, 4), maze, agents[50], "n = 50: game 1 + 20 steps of game 2")
    fig.tight_layout()
    fig.savefig("dyna_maze_results.png", dpi=110, bbox_inches="tight")
    print("\nChart saved to dyna_maze_results.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
