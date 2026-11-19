"""
DQN: Deep Q-Network (RL From Zero, topic 3).

Same idea as q_learning.py, with ONE big change:
the Q-table is replaced by a small neural network.

    Q-table:   Q[s, a]          look up a number in a box
    DQN:       Q_net(s)[a]      feed the state's numbers in, get one number per action out

Why? CartPole's state is 4 real numbers (cart position, cart speed,
pole angle, pole spin). There are infinitely many states, so there is
no table to fill in. The network GUESSES Q for any state with a formula,
and similar states get similar guesses.

The update line is the same Bellman target you already know,
    target = r + gamma * max Q(s2)     (0 future if the game ENDED)
but instead of  Q[s,a] += alpha * (target - Q[s,a])  we now
shrink the LOSS  (target - Q(s,a))^2  by gradient descent.

Two tricks make it stable (both can be switched off for experiments):
    replay buffer   : learn from a random mix of OLD moves, not just the last one
    target network  : a frozen copy of the net computes the targets,
                      so the target doesn't move every single step

This file is a library; run cartpole_dqn.py, ablations.py or experiments.py.
"""
import copy
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)      # be polite: other programs share the CPU


class QNetwork(nn.Module):
    """state (4 numbers) -> 64 hidden -> 64 hidden -> one Q-value per action (2 numbers)."""

    def __init__(self, n_inputs, n_actions, hidden=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),     # layer 1: weights W1, bias b1, then ReLU
            nn.Linear(hidden, hidden), nn.ReLU(),       # layer 2
            nn.Linear(hidden, n_actions),               # output: Q(s, left), Q(s, right)
        )

    def forward(self, states):
        return self.layers(states)


class ReplayBuffer:
    """A notebook of past moves (s, a, r, s2, ended). Old pages fall out when it's full."""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, s, a, r, s2, terminated):
        self.memory.append((s, a, r, s2, terminated))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)  # a RANDOM mix of old moves
        return to_tensors(batch)

    def __len__(self):
        return len(self.memory)


def to_tensors(batch):
    s, a, r, s2, term = zip(*batch)
    return (torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(term, dtype=torch.float32))


def epsilon_greedy(net, s, eps, n_actions):
    if random.random() < eps:
        return random.randrange(n_actions)                      # explore: random action
    with torch.no_grad():                                       # exploit: ask the network
        q = net(torch.tensor(s, dtype=torch.float32))
    return int(torch.argmax(q))


def compute_targets(online, target_net, r, s2, term, gamma, double):
    """The Bellman target  r + gamma * max Q(s2),  for a whole batch at once."""
    with torch.no_grad():                                       # targets are NOT trained
        if double:
            # Double DQN: the ONLINE net picks the best action in s2,
            # the TARGET net says how good that action is.
            best_a = online(s2).argmax(dim=1, keepdim=True)
            q_next = target_net(s2).gather(1, best_a).squeeze(1)
        else:
            q_next = target_net(s2).max(dim=1).values           # plain DQN: max of the target net
        return r + gamma * q_next * (1 - term)                  # no future if the game ENDED


def evaluate(net, env_id="CartPole-v1", games=10, seed=10_000):
    """Play with NO exploration and NO learning. Returns the average score."""
    env = gym.make(env_id)
    scores = []
    for g in range(games):
        s, _ = env.reset(seed=seed + g)
        done, total = False, 0.0
        while not done:
            s, r, terminated, truncated, _ = env.step(epsilon_greedy(net, s, 0.0, 2))
            total += r
            done = terminated or truncated
        scores.append(total)
    env.close()
    return float(np.mean(scores))


def train(env_id="CartPole-v1", total_steps=50_000, use_replay=True, use_target=True,
          double=False, loss_name="huber", lr=5e-4, gamma=0.99, batch_size=64,
          buffer_size=50_000, learning_starts=1_000, target_every=500,
          eps_start=1.0, eps_end=0.05, eps_decay=0.9995, eval_every=2_500, hidden=64,
          tau=1.0, train_every=1, watch_states=None, seed=0, verbose=True):
    """
    Train a DQN. Returns (best_net, history).

    history["returns"]      score of every training game
    history["ends"]         the step number at which each training game ended
    history["evals"]        (step, average greedy score) every eval_every steps
    history["watch_q"]      (step, average max-Q on watch_states)  -> to spot overestimation
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_id)
    n_inputs, n_actions = env.observation_space.shape[0], env.action_space.n
    online = QNetwork(n_inputs, n_actions, hidden)                     # the network we train
    target_net = copy.deepcopy(online) if use_target else online   # its frozen copy (or itself)
    optimizer = torch.optim.Adam(online.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss() if loss_name == "huber" else nn.MSELoss()
    buffer = ReplayBuffer(buffer_size)

    history = {"returns": [], "ends": [], "evals": [], "watch_q": [], "losses": []}
    best_score, best_net = -1.0, copy.deepcopy(online)
    eps = eps_start
    s, _ = env.reset(seed=seed)
    total = 0.0

    for step in range(1, total_steps + 1):
        # 1. choose, 2. act, 3. see -- exactly like tabular Q-learning
        a = epsilon_greedy(online, s, eps, n_actions)
        s2, r, terminated, truncated, _ = env.step(a)
        buffer.add(s, a, r, s2, terminated)
        total += r
        s = s2
        eps = max(eps_end, eps * eps_decay)                     # explore a bit less every step

        if terminated or truncated:
            history["returns"].append(total)
            history["ends"].append(step)
            total = 0.0
            s, _ = env.reset()

        # 4-6. learn: Bellman target -> loss -> gradient step
        if step >= learning_starts and step % train_every == 0:
            if use_replay:
                bs, ba, br, bs2, bterm = buffer.sample(batch_size)        # random OLD moves
            else:
                bs, ba, br, bs2, bterm = to_tensors([buffer.memory[-1]])  # only the move just made
            q_taken = online(bs).gather(1, ba.unsqueeze(1)).squeeze(1)    # Q(s, a) for the action taken
            target = compute_targets(online, target_net, br, bs2, bterm, gamma, double)
            loss = loss_fn(q_taken, target)                               # how wrong were we?

            optimizer.zero_grad()
            loss.backward()                                   # backprop: chain rule, done for us
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()                                  # gradient descent: nudge every weight
            if step % 100 == 0:
                history["losses"].append((step, float(loss)))

        if use_target and tau < 1.0:                          # "soft" version: copy 0.5% every step
            with torch.no_grad():
                for p_t, p_o in zip(target_net.parameters(), online.parameters()):
                    p_t.mul_(1 - tau).add_(tau * p_o)
        elif use_target and step % target_every == 0:
            target_net.load_state_dict(online.state_dict())   # refresh the frozen copy

        if step % eval_every == 0:
            score = evaluate(online, env_id)
            history["evals"].append((step, score))
            if watch_states is not None:
                with torch.no_grad():
                    q = online(torch.tensor(watch_states, dtype=torch.float32)).max(dim=1).values
                history["watch_q"].append((step, float(q.mean())))
            if score >= best_score:                           # keep the best version seen
                best_score, best_net = score, copy.deepcopy(online)
            if verbose:
                recent = np.mean(history["returns"][-20:]) if history["returns"] else 0.0
                print(f"  step {step:>6}  eps {eps:.3f}  games {len(history['returns']):>4}  "
                      f"train avg(last 20) {recent:6.1f}  test (greedy) {score:6.1f}")

    env.close()
    history["final_net"] = online
    return best_net, history


def moving_average(x, window=20):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")
