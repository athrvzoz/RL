"""
cartpole_game.py - CartPole as a game you can play, and a DQN bot that learns it while you watch.

  python cartpole_game.py play              you play: LEFT / RIGHT arrows (no key = no push)
  python cartpole_game.py train             watch DQN learn live, all fixes ON
  python cartpole_game.py train --naive     start with every fix OFF and watch it struggle
  python cartpole_game.py watch             the best saved bot plays; tap the pole with LEFT / RIGHT
  python cartpole_game.py compare           headless experiment: remove each fix in turn -> compare.png

Keys while training
  SPACE pause      UP / DOWN speed (real-time ... turbo)      1-6 switch a fix ON/OFF live
  G  show the bot's current skill (no randomness, no learning) - press G again to go back to training
  R  restart from a brand-new bot with the current fixes        S save      ESC quit
"""
import argparse
import csv
import math
import os
import sys
import time
import warnings

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
torch.set_num_threads(1)  # tiny network: one thread is fastest


# ═════════════════════════════════════════════════════════════════════════════
# 1. The game: classic cart-pole physics (Barto, Sutton & Anderson 1983)
# ═════════════════════════════════════════════════════════════════════════════
class CartPole:
    GRAVITY, M_CART, M_POLE, HALF_LEN = 9.8, 1.0, 0.1, 0.5
    FORCE, DT = 10.0, 0.02                      # 10 N push, 50 physics steps per second
    X_LIMIT, THETA_LIMIT = 2.4, math.radians(12)
    MAX_STEPS = 500                             # the game is cut off after 500 steps (10 s)

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, 4)
        self.steps = 0
        return self.obs()

    def obs(self):
        return self.state.astype(np.float32)

    def step(self, action):
        """action 0 = push left, 1 = push right, None = no push (human mode only)."""
        x, v, th, w = self.state
        f = 0.0 if action is None else (self.FORCE if action == 1 else -self.FORCE)
        total = self.M_CART + self.M_POLE
        pml = self.M_POLE * self.HALF_LEN
        cos, sin = math.cos(th), math.sin(th)
        temp = (f + pml * w * w * sin) / total
        th_acc = (self.GRAVITY * sin - cos * temp) / (self.HALF_LEN * (4 / 3 - self.M_POLE * cos * cos / total))
        x_acc = temp - pml * th_acc * cos / total
        x, v, th, w = x + self.DT * v, v + self.DT * x_acc, th + self.DT * w, w + self.DT * th_acc
        self.state = np.array([x, v, th, w])
        self.steps += 1
        terminated = abs(x) > self.X_LIMIT or abs(th) > self.THETA_LIMIT   # it fell / left the track
        truncated = not terminated and self.steps >= self.MAX_STEPS        # time is up (not a failure!)
        return self.obs(), 1.0, terminated, truncated


# ═════════════════════════════════════════════════════════════════════════════
# 2. The bot: DQN with every fix switchable
# ═════════════════════════════════════════════════════════════════════════════
FIX_KEYS = ["explore", "replay", "target", "double", "huber", "timeout"]
FIX_INFO = {  # name, the problem, the fix
    "explore": ("Exploration (e-greedy)",
                "a new net keeps repeating the same move, so it never finds out what works",
                "act randomly with probability e, fading 1.0 -> 0.02 over 10k steps"),
    "replay": ("Experience replay",
               "back-to-back frames are almost identical; learning them in order overfits the moment",
               "keep the last 50k moves and learn from random batches of 64"),
    "target": ("Target network",
               "the target r + g*maxQ(s') shifts after every update: the net chases its own tail",
               "compute targets with a slow-moving copy of the net (soft update tau=0.005)"),
    "double": ("Double DQN",
               "the max over noisy estimates is biased upward, so Q values overestimate",
               "the online net picks the next move, the target net scores it"),
    "huber": ("Huber loss + gradient clipping",
              "a few huge TD errors make huge gradients that wreck the weights",
              "Huber loss grows linearly for big errors; gradient norm clipped to 10"),
    "timeout": ("Time limit is not a fall",
                "at 500 steps the game is cut off; calling that 'game over' says good play has no future",
                "only a real fall is terminal; keep bootstrapping when time runs out"),
}

GAMMA, LR, BATCH, BUFFER, WARMUP = 0.99, 5e-4, 64, 50_000, 1000
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.02, 10_000
TAU, CLIP = 0.005, 10.0


class QNet(nn.Module):
    """4 numbers in (cart pos, cart speed, pole angle, pole spin) -> 2 numbers out: Q(left), Q(right)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, fixes, seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.fixes = fixes  # shared dict: switching a key changes behaviour immediately
        self.q, self.tq = QNet(), QNet()
        self.tq.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=LR)
        self.S = np.zeros((BUFFER, 4), np.float32)
        self.S2 = np.zeros((BUFFER, 4), np.float32)
        self.A = np.zeros(BUFFER, np.int64)
        self.R = np.zeros(BUFFER, np.float32)
        self.D = np.zeros(BUFFER, np.float32)
        self.ptr = self.size = 0
        self.steps = self.updates = 0

    def epsilon(self):
        if not self.fixes["explore"]:
            return 0.0
        return max(EPS_END, EPS_START - (EPS_START - EPS_END) * self.steps / EPS_DECAY)

    @torch.no_grad()
    def q_values(self, obs):
        return self.q(torch.as_tensor(obs)[None])[0].numpy()

    def act(self, obs, greedy=False):
        q = self.q_values(obs)
        eps = 0.0 if greedy else self.epsilon()
        if not greedy:
            self.steps += 1
        if self.rng.random() < eps:
            return int(self.rng.integers(2)), q, True
        return int(q.argmax()), q, False

    def remember(self, s, a, r, s2, done):
        i = self.ptr
        self.S[i], self.A[i], self.R[i], self.S2[i], self.D[i] = s, a, r, s2, float(done)
        self.ptr = (i + 1) % BUFFER
        self.size = min(self.size + 1, BUFFER)

    def learn(self):
        f = self.fixes
        if f["replay"]:
            if self.size < WARMUP:
                return None
            idx = self.rng.integers(0, self.size, BATCH)                 # random moves from anywhere in memory
        else:
            if self.size < BATCH:
                return None
            idx = (self.ptr - 1 - np.arange(BATCH)) % BUFFER           # only the latest 64 moves, in order
        s, s2 = torch.from_numpy(self.S[idx]), torch.from_numpy(self.S2[idx])
        a, r, d = torch.from_numpy(self.A[idx]), torch.from_numpy(self.R[idx]), torch.from_numpy(self.D[idx])

        qs = self.q(s)
        q = qs.gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            judge = self.tq if f["target"] else self.q
            if f["double"]:
                best = self.q(s2).argmax(1, keepdim=True)              # online net chooses...
                nq = judge(s2).gather(1, best).squeeze(1)              # ...target net evaluates
            else:
                nq = judge(s2).max(1).values
            y = r + GAMMA * (1 - d) * nq                               # Bellman target
        loss = F.smooth_l1_loss(q, y) if f["huber"] else F.mse_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        if f["huber"]:
            nn.utils.clip_grad_norm_(self.q.parameters(), CLIP)
        self.opt.step()
        if f["target"]:
            with torch.no_grad():
                for p, tp in zip(self.q.parameters(), self.tq.parameters()):
                    tp.mul_(1 - TAU).add_(TAU * p)
        self.updates += 1
        return loss.item(), qs.max(1).values.mean().item()

    def save(self, path, **meta):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model": self.q.state_dict(), "fixes": dict(self.fixes), **meta}, path)

    @staticmethod
    def load(path):
        ck = torch.load(path, weights_only=False)
        ag = Agent(dict.fromkeys(FIX_KEYS, True))
        ag.q.load_state_dict(ck["model"])
        return ag, ck


def greedy_episode(agent, env):
    obs, n = env.reset(), 0
    while True:
        obs, _, term, trunc = env.step(agent.act(obs, greedy=True)[0])
        n += 1
        if term or trunc:
            return n


# ═════════════════════════════════════════════════════════════════════════════
# 3. Training loop, one step at a time (so the screen can draw between steps)
# ═════════════════════════════════════════════════════════════════════════════
class Trainer:
    EVAL_EVERY, EVAL_EPISODES, AVG = 10, 3, 20

    def __init__(self, fixes, seed=0, save_dir=MODELS):
        self.fixes, self.seed, self.save_dir = fixes, seed, save_dir
        self.agent = Agent(fixes, seed)
        self.env, self.eval_env = CartPole(seed), CartPole(seed + 1000)
        self.obs = self.env.reset()
        self.episode, self.ep_ret = 0, 0.0
        self.ep_loss, self.ep_q = [], []
        self.hist = {"score": [], "loss": [], "maxq": [], "eps": [], "eval": []}  # eval: (episode, score)
        self.best_eval, self.best_avg, self.solved_at = -1.0, 0.0, None
        self.last = {"action": 0, "q": np.zeros(2), "explored": False}
        self.events = []  # (time, text) shown briefly on screen

    def step(self):
        a, q, explored = self.agent.act(self.obs)
        s2, r, term, trunc = self.env.step(a)
        done = term if self.fixes["timeout"] else (term or trunc)
        self.agent.remember(self.obs, a, r, s2, done)
        out = self.agent.learn()
        if out:
            self.ep_loss.append(out[0])
            self.ep_q.append(out[1])
        self.last = {"action": a, "q": q, "explored": explored}
        self.obs, self.ep_ret = s2, self.ep_ret + r
        if term or trunc:
            self.end_episode(trunc)
            return True
        return False

    def end_episode(self, timed_out):
        h = self.hist
        self.episode += 1
        h["score"].append(self.ep_ret)
        h["loss"].append(float(np.mean(self.ep_loss)) if self.ep_loss else float("nan"))
        h["maxq"].append(float(np.mean(self.ep_q)) if self.ep_q else float("nan"))
        h["eps"].append(self.agent.epsilon())
        self.last_end = "time up (500)" if timed_out else "fell"
        self.ep_ret, self.ep_loss, self.ep_q = 0.0, [], []
        self.obs = self.env.reset()
        avg = float(np.mean(h["score"][-self.AVG:]))
        self.best_avg = max(self.best_avg, avg)
        if self.episode % self.EVAL_EVERY == 0:
            score = float(np.mean([greedy_episode(self.agent, self.eval_env) for _ in range(self.EVAL_EPISODES)]))
            h["eval"].append((self.episode, score))
            if score >= self.best_eval and self.save_dir:
                self.best_eval = score
                self.agent.save(os.path.join(self.save_dir, "best.pt"), episode=self.episode, eval=score)
                self.events.append((time.time(), f"new best bot saved (test score {score:.0f})"))
            self.best_eval = max(self.best_eval, score)
            if score >= 475 and self.solved_at is None:
                self.solved_at = self.episode
                self.events.append((time.time(), f"SOLVED at episode {self.episode}!"))

    def diagnosis(self):
        """Plain-language read-out of what is going wrong (or right) right now."""
        h, ag, f, out = self.hist, self.agent, self.fixes, []
        n = len(h["score"])
        avg = float(np.mean(h["score"][-self.AVG:])) if n else 0.0
        recent_q = [v for v in h["maxq"][-10:] if v == v]
        losses = [v for v in h["loss"] if v == v]
        if f["replay"] and ag.size < WARMUP:
            out.append(("info", f"Filling the replay memory with random play first: {ag.size}/{WARMUP} moves"))
        elif ag.epsilon() > 0.3:
            out.append(("info", f"Mostly exploring: {ag.epsilon():.0%} of moves are random (orange arrow)"))
        if not f["explore"] and n > 20 and avg < 60:
            out.append(("bad", "No exploration: the bot repeats its first habits and never tries anything else"))
        if recent_q and max(recent_q) > 110:
            out.append(("bad", f"Q = {max(recent_q):.0f} but the most it can ever earn is 100 "
                               f"(1/(1-g)): Q values are overestimating / diverging"))
        if len(losses) > 30 and np.mean(losses[-5:]) > 20 * max(np.median(losses), 1e-3):
            out.append(("bad", "Loss is exploding: TD errors are growing instead of shrinking"))
        if self.best_avg > 250 and avg < 0.5 * self.best_avg:
            out.append(("bad", f"Collapse: average fell from {self.best_avg:.0f} to {avg:.0f} (catastrophic "
                               f"forgetting) - the best bot is still saved in models/best.pt"))
        if n > 120 and self.best_avg < 60:
            out.append(("bad", "Not learning: after 120+ games it is still near the random score (~20)"))
        if f["double"] and not f["target"]:
            out.append(("info", "Double DQN needs the target network: with it OFF, double does nothing"))
        if self.solved_at:
            out.append(("good", f"Solved at episode {self.solved_at}: the bot balanced 475+ steps with no randomness"))
        elif n and not out:
            out.append(("good", f"Learning: average score of the last {min(n, self.AVG)} games is {avg:.0f}"))
        return out


# ═════════════════════════════════════════════════════════════════════════════
# 4. Drawing
# ═════════════════════════════════════════════════════════════════════════════
BG, PANEL, GRID = (17, 19, 27), (28, 31, 43), (52, 57, 75)
TEXT, DIM, FAINT = (228, 231, 238), (146, 152, 168), (95, 101, 120)
BLUE, GREEN, RED = (88, 160, 255), (78, 205, 125), (238, 92, 92)
ORANGE, YELLOW, PURPLE = (255, 168, 60), (240, 212, 84), (182, 134, 255)
W, H = 1360, 800
SPEEDS = [("Real-time 1x", 1, 50), ("Fast 5x", 5, 60), ("Very fast 40x", 40, 0), ("Turbo (draw once per game)", None, 0)]


def mix(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


class Painter:
    def __init__(self, surf):
        import pygame
        self.pg, self.surf, self.fonts = pygame, surf, {}

    def font(self, size, bold=False, mono=False):
        key = (size, bold, mono)
        if key not in self.fonts:
            name = "consolas" if mono else "segoeui,arial"
            self.fonts[key] = self.pg.font.SysFont(name, size, bold=bold)
        return self.fonts[key]

    def text(self, s, pos, size=16, color=TEXT, bold=False, mono=False, anchor="topleft"):
        img = self.font(size, bold, mono).render(str(s), True, color)
        r = img.get_rect(**{anchor: pos})
        self.surf.blit(img, r)
        return r

    def panel(self, rect, radius=10):
        self.pg.draw.rect(self.surf, PANEL, rect, border_radius=radius)

    def dashed(self, color, p1, p2, dash=6, width=1):
        (x1, y1), (x2, y2) = p1, p2
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / dash))
        for i in range(0, n, 2):
            a, b = i / n, min(1.0, (i + 1) / n)
            self.pg.draw.line(self.surf, color, (x1 + (x2 - x1) * a, y1 + (y2 - y1) * a),
                              (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), width)

    def pill(self, pos, on):
        r = self.pg.Rect(pos[0], pos[1], 40, 20)
        self.pg.draw.rect(self.surf, GREEN if on else (90, 50, 55), r, border_radius=10)
        self.text("ON" if on else "OFF", r.center, 12, (10, 30, 15) if on else (240, 170, 170), True, anchor="center")

    # ── the game scene ──────────────────────────────────────────────────────
    def scene(self, rect, env, decision=None, title="", subtitle="", hint=None):
        pg, s = self.pg, self.surf
        self.panel(rect)
        x, v, th, w = env.state
        scale = (rect.w - 80) / (2 * env.X_LIMIT)            # pixels per metre
        ground = rect.bottom - 92
        mid = rect.centerx
        # track with metre ticks and the walls at +-2.4 m
        pg.draw.line(s, GRID, (mid - env.X_LIMIT * scale, ground), (mid + env.X_LIMIT * scale, ground), 3)
        for m in np.arange(-2.0, 2.01, 0.5):
            px = mid + m * scale
            pg.draw.line(s, GRID, (px, ground), (px, ground + (8 if m % 1 == 0 else 4)), 1)
            if m % 1 == 0:
                self.text(f"{m:+.0f} m" if m else "0", (px, ground + 10), 12, FAINT, anchor="midtop")
        for side in (-1, 1):
            px = mid + side * env.X_LIMIT * scale
            pg.draw.rect(s, mix(PANEL, RED, 0.55), (px - 3, ground - 60, 6, 60), border_radius=2)
        # cart, coloured by how close it is to a wall
        danger_x = abs(x) / env.X_LIMIT
        cx = mid + x * scale
        cart = pg.Rect(0, 0, 84, 36)
        cart.midbottom = (cx, ground - 9)
        pg.draw.rect(s, mix((70, 110, 190), RED, (danger_x - 0.6) / 0.4), cart, border_radius=6)
        for dx in (-24, 24):
            pg.draw.circle(s, (40, 44, 58), (cx + dx, ground - 8), 9)
            pg.draw.circle(s, DIM, (cx + dx, ground - 8), 3)
        # angle limits +-12 deg and the pole, coloured by how close it is to falling
        pivot = (cx, cart.top + 2)
        L = 2 * env.HALF_LEN * scale
        for lim in (-env.THETA_LIMIT, env.THETA_LIMIT):
            self.dashed(mix(PANEL, RED, 0.6), pivot, (pivot[0] + math.sin(lim) * (L + 20), pivot[1] - math.cos(lim) * (L + 20)))
        danger_t = abs(th) / env.THETA_LIMIT
        tip = (pivot[0] + math.sin(th) * L, pivot[1] - math.cos(th) * L)
        pg.draw.line(s, mix(GREEN, RED, (danger_t - 0.3) / 0.7), pivot, tip, 11)
        pg.draw.circle(s, mix(GREEN, RED, (danger_t - 0.3) / 0.7), tip, 7)
        pg.draw.circle(s, TEXT, pivot, 6)
        # the push this step
        if decision is not None:
            a, explored = decision["action"], decision["explored"]
            col = ORANGE if explored else BLUE
            d = 1 if a == 1 else -1
            bx = cart.right + 8 if d > 0 else cart.left - 8
            y0 = cart.centery
            pg.draw.polygon(s, col, [(bx, y0 - 13), (bx, y0 + 13), (bx + d * 26, y0)])
        # titles
        self.text(title, (rect.x + 18, rect.y + 12), 24, TEXT, True)
        self.text(subtitle, (rect.x + 18, rect.y + 46), 15, DIM)
        # inputs the network sees
        readout = (f"inputs ->  x = {x:+.2f} m    speed = {v:+.2f} m/s    "
                   f"angle = {math.degrees(th):+.1f} deg    spin = {w:+.2f} rad/s")
        self.text(readout, (rect.x + 18, rect.bottom - 30), 14, DIM, mono=True)
        # decision box
        if decision is not None:
            self.decision_box(pg.Rect(rect.right - 330, rect.y + 12, 312, 128), decision)
        if hint is not None:
            self.text(hint, (rect.right - 18, rect.y + 16), 15, PURPLE, True, anchor="topright")

    def decision_box(self, r, d):
        pg = self.pg
        pg.draw.rect(self.surf, (22, 25, 35), r, border_radius=8)
        q = d["q"]
        self.text("Network output  Q(state, move)", (r.x + 12, r.y + 8), 13, DIM)
        top = max(1.0, float(np.max(np.abs(q))))
        for i, (name, val) in enumerate((("left", q[0]), ("right", q[1]))):
            y = r.y + 32 + i * 26
            chosen = d["action"] == i
            self.text(f"Q({name})", (r.x + 12, y), 14, TEXT if chosen else DIM, chosen, mono=True)
            bw = int(150 * max(0.0, val) / top)
            pg.draw.rect(self.surf, GRID, (r.x + 92, y + 3, 150, 14), border_radius=4)
            pg.draw.rect(self.surf, BLUE if chosen else FAINT, (r.x + 92, y + 3, bw, 14), border_radius=4)
            self.text(f"{val:6.2f}", (r.right - 12, y), 14, TEXT, mono=True, anchor="topright")
        move = "RIGHT" if d["action"] == 1 else "LEFT"
        if d["explored"]:
            msg, col = f"push {move}: RANDOM (exploring, e={d.get('eps', 0):.2f})", ORANGE
        else:
            msg, col = f"push {move}: network's choice (+{abs(q[1] - q[0]):.2f})", BLUE
        self.text(msg, (r.x + 12, r.y + 92), 14, col, True)

    # ── charts ──────────────────────────────────────────────────────────────
    def chart(self, rect, title, series, ylim=None, hlines=(), log=False, note=""):
        """series: list of (ys, color, kind) kind in bars|line|dots; dots take (x, y) pairs."""
        pg = self.pg
        self.panel(rect)
        self.text(title, (rect.x + 12, rect.y + 7), 14, TEXT, True)
        if note:
            self.text(note, (rect.right - 12, rect.y + 9), 12, DIM, anchor="topright")
        plot = pg.Rect(rect.x + 50, rect.y + 32, rect.w - 64, rect.h - 52)
        n = max([len(ys) for ys, _, k in series if k != "dots"] + [1])
        tf = (lambda v: math.log10(max(v, 1e-5))) if log else (lambda v: v)
        vals = [tf(v) for ys, _, k in series for v in ((p[1] for p in ys) if k == "dots" else ys) if v == v]
        if not vals:
            self.text("waiting for the first games...", plot.center, 13, FAINT, anchor="center")
            return
        lo, hi = ylim if ylim else (min(vals), max(vals))
        if hi - lo < 1e-6:
            hi = lo + 1
        if not ylim:
            pad = (hi - lo) * 0.1
            lo, hi = lo - pad, hi + pad
        X = lambda i: plot.x + plot.w * (i / max(n - 1, 1))
        Y = lambda v: plot.bottom - plot.h * (min(max(tf(v), lo), hi) - lo) / (hi - lo)
        for k in range(3):
            gv = lo + (hi - lo) * k / 2
            gy = plot.bottom - plot.h * k / 2
            pg.draw.line(self.surf, GRID, (plot.x, gy), (plot.right, gy), 1)
            label = f"{10 ** gv:.0e}" if log else (f"{gv:.0f}" if abs(hi - lo) > 4 else f"{gv:.2f}")
            self.text(label, (plot.x - 6, gy), 11, FAINT, anchor="midright")
        for v, col, lab in hlines:
            if lo <= tf(v) <= hi:
                self.dashed(col, (plot.x, Y(v)), (plot.right, Y(v)), 5)
                self.text(lab, (plot.right - 4, Y(v) - 2), 11, col, anchor="bottomright")
        for ys, col, kind in series:
            if kind == "bars":
                bw = max(1, plot.w / n - 1)
                for i, v in enumerate(ys):
                    if v == v:
                        y = Y(v)
                        pg.draw.rect(self.surf, col, (X(i) - bw / 2, y, bw, plot.bottom - y))
            elif kind == "line":
                stride = max(1, len(ys) // plot.w)
                pts = [(X(i), Y(v)) for i, v in enumerate(ys) if i % stride == 0 and v == v]
                if len(pts) > 1:
                    pg.draw.lines(self.surf, col, False, pts, 2)
            elif kind == "dots":
                for ep, v in ys:
                    pg.draw.circle(self.surf, col, (X(ep - 1), Y(v)), 4)
        self.text(f"games played: {n}", (plot.right, plot.bottom + 4), 11, FAINT, anchor="topright")


def moving_avg(ys, k):
    out, c = [], 0.0
    for i, v in enumerate(ys):
        c += v
        if i >= k:
            c -= ys[i - k]
        out.append(c / min(i + 1, k))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 5. The app: play / train / watch
# ═════════════════════════════════════════════════════════════════════════════
class App:
    def __init__(self, args):
        import pygame
        self.pg, self.args = pygame, args
        pygame.init()
        pygame.display.set_caption("CartPole - DQN learning lab")
        self.screen = pygame.display.set_mode((W, H))
        self.p = Painter(self.screen)
        self.clock = pygame.time.Clock()
        self.t0 = time.time()

    def shot_due(self):
        a = self.args
        if a.shot and time.time() - self.t0 > a.shot_at:
            self.pg.image.save(self.screen, a.shot)
            return True
        return False

    def header(self, mode, right):
        self.pg.draw.rect(self.screen, (12, 14, 20), (0, 0, W, 44))
        self.p.text("CartPole", (16, 8), 20, TEXT, True)
        self.p.text(mode, (118, 11), 16, BLUE, True)
        self.p.text(right, (W - 16, 13), 14, DIM, anchor="topright")

    # ── train ───────────────────────────────────────────────────────────────
    def train(self):
        pg, a = self.pg, self.args
        fixes = dict.fromkeys(FIX_KEYS, not a.naive)
        tr = Trainer(fixes, a.seed)
        speed, paused, show = a.speed, False, None   # show = env for greedy demo (G)
        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    return self.finish(tr)
                if e.type != pg.KEYDOWN:
                    continue
                if e.key == pg.K_SPACE:
                    paused = not paused
                elif e.key == pg.K_UP:
                    speed = min(speed + 1, len(SPEEDS) - 1)
                elif e.key == pg.K_DOWN:
                    speed = max(speed - 1, 0)
                elif pg.K_1 <= e.key <= pg.K_6:
                    k = FIX_KEYS[e.key - pg.K_1]
                    fixes[k] = not fixes[k]
                    tr.events.append((time.time(), f"{FIX_INFO[k][0]} switched {'ON' if fixes[k] else 'OFF'}"))
                elif e.key == pg.K_r:
                    tr = Trainer(fixes, tr.seed + 1)
                    tr.events.append((time.time(), "restarted with a brand-new bot"))
                elif e.key == pg.K_g:
                    show = None if show else CartPole(int(time.time()))
                elif e.key == pg.K_s:
                    tr.agent.save(os.path.join(MODELS, "last.pt"), episode=tr.episode)
                    tr.events.append((time.time(), "saved models/last.pt"))

            if show is not None:                           # demo: current policy, no randomness, no learning
                act, q, _ = tr.agent.act(show.obs(), greedy=True)
                _, _, term, trunc = show.step(act)
                decision = {"action": act, "q": q, "explored": False}
                if term or trunc:
                    tr.events.append((time.time(), f"demo game: {show.steps} steps"))
                    show.reset()
                self.draw_train(tr, speed, paused, show, decision)
                self.clock.tick(50)
            else:
                if not paused:
                    n = SPEEDS[speed][1]
                    if n is None:
                        while not tr.step():
                            pass
                    else:
                        for _ in range(n):
                            tr.step()
                decision = dict(tr.last, eps=tr.agent.epsilon())
                self.draw_train(tr, speed, paused, None, decision)
                self.clock.tick(SPEEDS[speed][2])
            pg.display.flip()
            if self.shot_due():
                return self.finish(tr)

    def draw_train(self, tr, speed, paused, show, decision):
        p, pg = self.p, self.pg
        self.screen.fill(BG)
        state = "PAUSED" if paused else ("DEMO: bot's current skill, no randomness, not learning" if show else SPEEDS[speed][0])
        self.header("TRAINING DQN", f"{state}    |   SPACE pause   UP/DOWN speed   1-6 fixes   "
                                    f"G demo   R restart   ESC quit")
        env = show or tr.env
        h = tr.hist
        best_score = max(h["score"]) if h["score"] else 0
        if show:
            title, sub = f"Demo game  -  {show.steps} steps", "press G to go back to training"
        else:
            title = f"Game {tr.episode + 1}   -   balanced {env.steps} / {env.MAX_STEPS} steps"
            sub = (f"best game so far {best_score:.0f}    |    replay memory {tr.agent.size:,}    |    "
                   f"learning updates {tr.agent.updates:,}    |    random moves {tr.agent.epsilon():.0%}")
        p.scene(pg.Rect(16, 56, 800, 404), env, decision, title, sub)

        # issues & fixes
        r = pg.Rect(16, 470, 800, 314)
        p.panel(r)
        p.text("Problems DQN runs into -> the fix (press the number to switch it live)", (r.x + 16, r.y + 10), 15, TEXT, True)
        for i, k in enumerate(FIX_KEYS):
            name, problem, fix = FIX_INFO[k]
            y = r.y + 38 + i * 36
            on = tr.fixes[k]
            p.text(str(i + 1), (r.x + 26, y + 9), 15, YELLOW, True, anchor="center")
            p.pill((r.x + 44, y), on)
            p.text(name, (r.x + 94, y), 14, TEXT if on else DIM, True)
            p.text(("fix: " + fix) if on else ("problem: " + problem), (r.x + 300, y + 1), 13, GREEN if on else ORANGE)
            p.text(("solves: " + problem) if on else "", (r.x + 94, y + 17), 12, FAINT)
        y = r.y + 38 + 6 * 36 + 4
        colors = {"good": GREEN, "bad": RED, "info": YELLOW}
        for kind, msg in tr.diagnosis()[:2]:
            p.text(("! " if kind == "bad" else "> ") + msg, (r.x + 16, y), 13, colors[kind], True)
            y += 19
        now = time.time()
        tr.events = [(t, m) for t, m in tr.events if now - t < 4]
        for j, (t, m) in enumerate(tr.events[-2:]):
            p.text(m, (816 - 16, 470 - 26 - 22 * j), 14, PURPLE, True, anchor="bottomright")

        # charts
        x, cw = 832, W - 832 - 16
        scores = h["score"]
        p.chart(pg.Rect(x, 56, cw, 214), "Score per game (steps balanced)",
                [(scores, (55, 70, 105), "bars"), (moving_avg(scores, 20), BLUE, "line"), (h["eval"], GREEN, "dots")],
                ylim=(0, 520), hlines=[(475, GREEN, "solved = 475")],
                note="bars: each game   blue: avg of 20   green: test (no randomness)")
        p.chart(pg.Rect(x, 278, cw, 170), "Average max Q - what the bot expects to earn",
                [(h["maxq"], PURPLE, "line")], hlines=[(100, RED, "ceiling 1/(1-g) = 100")],
                note="above the red line = overestimation")
        p.chart(pg.Rect(x, 456, cw, 160), "TD loss (log scale) - how wrong its predictions are",
                [(h["loss"], ORANGE, "line")], log=True)
        p.chart(pg.Rect(x, 624, cw, 160), "Exploration e - share of random moves",
                [(h["eps"], YELLOW, "line")], ylim=(0, 1.05))

    def finish(self, tr):
        if tr.episode:
            tr.agent.save(os.path.join(MODELS, "last.pt"), episode=tr.episode)
            os.makedirs(MODELS, exist_ok=True)
            with open(os.path.join(MODELS, "training_log.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "score", "loss", "avg_max_q", "epsilon"])
                for i, row in enumerate(zip(tr.hist["score"], tr.hist["loss"], tr.hist["maxq"], tr.hist["eps"]), 1):
                    w.writerow([i, *row])
            print(f"{tr.episode} games, best test score {tr.best_eval:.0f}"
                  f"{f', solved at game {tr.solved_at}' if tr.solved_at else ''}. Saved models/best.pt, "
                  f"models/last.pt, models/training_log.csv")
        self.pg.quit()

    # ── watch the best bot ──────────────────────────────────────────────────
    def watch(self):
        pg = self.pg
        path = os.path.join(MODELS, self.args.model)
        if not os.path.exists(path):
            sys.exit(f"No models/{self.args.model} yet - run:  python cartpole_game.py train")
        agent, ck = Agent.load(path)
        env, scores, pushes = CartPole(), [], 0
        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    return pg.quit()
                if e.type == pg.KEYDOWN and e.key in (pg.K_LEFT, pg.K_RIGHT):
                    env.state[3] += 0.6 if e.key == pg.K_RIGHT else -0.6   # tap the pole
                    pushes += 1
            a, q, _ = agent.act(env.obs(), greedy=True)
            _, _, term, trunc = env.step(a)
            if term or trunc:
                scores.append(env.steps)
                env.reset()
                pushes = 0
            self.screen.fill(BG)
            self.header("WATCH THE TRAINED BOT", "LEFT / RIGHT tap the pole to test it    ESC quit")
            sub = (f"model from game {ck.get('episode', '?')} (test score {ck.get('eval', 0):.0f})   |   "
                   f"games here: {len(scores)}   average {np.mean(scores) if scores else 0:.0f}   |   taps survived: {pushes}")
            self.p.scene(pg.Rect(16, 56, W - 32, H - 72), env, {"action": a, "q": q, "explored": False},
                         f"Balanced {env.steps} / {env.MAX_STEPS} steps", sub)
            pg.display.flip()
            self.clock.tick(50)
            if self.shot_due():
                return pg.quit()

    # ── you play ────────────────────────────────────────────────────────────
    def play(self):
        pg = self.pg
        path = os.path.join(MODELS, "best.pt")
        bot = Agent.load(path)[0] if os.path.exists(path) else None
        env, best, last, fps, started = CartPole(), 0, None, 25, False
        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    return pg.quit()
                if e.type == pg.KEYDOWN and e.key == pg.K_UP:
                    fps = min(50, fps + 5)
                if e.type == pg.KEYDOWN and e.key == pg.K_DOWN:
                    fps = max(10, fps - 5)
                if e.type == pg.KEYDOWN and e.key in (pg.K_LEFT, pg.K_RIGHT):
                    started = True
            keys = pg.key.get_pressed()
            act = 0 if keys[pg.K_LEFT] else 1 if keys[pg.K_RIGHT] else None
            if started:
                _, _, term, trunc = env.step(act)
                if term or trunc:
                    last = env.steps
                    best = max(best, last)
                    env.reset()
                    started = False
            hint = None
            if bot is not None:
                ba, bq, _ = bot.act(env.obs(), greedy=True)
                hint = f"bot would push {'RIGHT' if ba else 'LEFT'}"
            self.screen.fill(BG)
            self.header("YOU PLAY", f"hold LEFT / RIGHT to push   UP / DOWN game speed ({fps * 2}%)   ESC quit")
            sub = (f"your best {best}   last game {last if last is not None else '-'}   |   "
                   f"press an arrow to start; no key = no push (the bot must ALWAYS push)")
            self.p.scene(pg.Rect(16, 56, W - 32, H - 72), env,
                         None if act is None else {"action": act, "q": np.zeros(2), "explored": False, "human": True},
                         f"Balanced {env.steps} / {env.MAX_STEPS} steps", sub, hint)
            pg.display.flip()
            self.clock.tick(fps)
            if self.shot_due():
                return pg.quit()


# ═════════════════════════════════════════════════════════════════════════════
# 6. Headless experiment: what happens when each fix is removed
# ═════════════════════════════════════════════════════════════════════════════
def run_config(job):
    name, fixes, episodes, seed = job
    torch.set_num_threads(1)
    tr = Trainer(dict(fixes), seed, save_dir=None)
    t = time.time()
    while tr.episode < episodes:
        tr.step()
    return {"name": name, "seed": seed, "score": tr.hist["score"], "eval": tr.hist["eval"],
            "maxq": tr.hist["maxq"], "solved_at": tr.solved_at, "secs": time.time() - t}


def compare(args):
    from multiprocessing import Pool
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = [("all fixes", dict.fromkeys(FIX_KEYS, True))]
    for k in FIX_KEYS:
        configs.append((f"no {FIX_INFO[k][0].split(' (')[0].lower()}", {**dict.fromkeys(FIX_KEYS, True), k: False}))
    configs.append(("naive (no fixes)", dict.fromkeys(FIX_KEYS, False)))
    jobs = [(n, f, args.episodes, s) for n, f in configs for s in range(args.seeds)]
    workers = min(len(jobs), os.cpu_count(), 4)  # each worker loads its own torch (~1 GB); more runs out of memory
    print(f"running {len(jobs)} trainings x {args.episodes} games, {workers} at a time ...")
    with Pool(workers) as pool:
        results = pool.map(run_config, jobs)

    rows, by = [], {}
    for r in results:
        by.setdefault(r["name"], []).append(r)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    for ax, (name, _) in zip(axes.flat, configs):
        rs = by[name]
        for r in rs:
            ax.plot(moving_avg(r["score"], 20), color="#5a9cff", lw=1.2, alpha=0.8)
            if r["eval"]:
                ex, ey = zip(*r["eval"])
                ax.scatter(ex, ey, s=6, color="#2e9e5b", alpha=0.6)
        last = [np.mean([v for _, v in r["eval"][-5:]]) for r in rs]
        solved = [r["solved_at"] for r in rs]
        ax.set_title(f"{name}\nlast tests {np.mean(last):.0f}, solved: "
                     f"{', '.join(str(s) if s else 'no' for s in solved)}", fontsize=10)
        ax.axhline(475, color="#2e9e5b", ls="--", lw=0.8)
        ax.set_ylim(0, 520)
        ax.set_xlabel("game")
        rows.append([name, f"{np.mean(last):.0f}", " ".join(str(s or "-") for s in solved),
                     f"{np.nanmax([np.nanmax(r['maxq']) for r in rs]):.0f}", f"{np.mean([r['secs'] for r in rs]):.0f}"])
    axes[0, 0].set_ylabel("score (blue: avg of 20 games, green: tests)")
    fig.suptitle(f"DQN on CartPole: removing one fix at a time ({args.episodes} games, {args.seeds} seed(s))")
    fig.tight_layout()
    out = os.path.join(HERE, "compare.png")
    fig.savefig(out, dpi=110)
    with open(os.path.join(HERE, "compare.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "last_5_tests_avg", "solved_at_game", "peak_avg_max_q", "seconds"])
        w.writerows(rows)
    print(f"\n{'config':28} {'last tests':>10} {'solved at':>12} {'peak Q':>8} {'secs':>6}")
    for r in rows:
        print(f"{r[0]:28} {r[1]:>10} {r[2]:>12} {r[3]:>8} {r[4]:>6}")
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["play", "train", "watch", "compare"])
    ap.add_argument("--naive", action="store_true", help="train: start with every fix OFF")
    ap.add_argument("--speed", type=int, default=1, help="train: 0 real-time, 1 fast, 2 very fast, 3 turbo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="best.pt", help="watch: best.pt (best test score) or last.pt (when you quit)")
    ap.add_argument("--episodes", type=int, default=300, help="compare: games per run")
    ap.add_argument("--seeds", type=int, default=1, help="compare: runs per config")
    ap.add_argument("--shot", help=argparse.SUPPRESS)
    ap.add_argument("--shot-at", type=float, default=5, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.mode == "compare":
        return compare(args)
    getattr(App(args), args.mode)()


if __name__ == "__main__":
    main()
