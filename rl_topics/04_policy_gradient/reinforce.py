"""
REINFORCE: the simplest policy-gradient algorithm (shared by cartpole.py and experiments.py).

The big difference from Q-learning:
    Q-learning learns SCORES  Q[s, a], then picks the action with the biggest score.
    REINFORCE  learns PROBABILITIES  π(a | s) directly, and picks an action by rolling
               its own dice with those probabilities.

The update, after every finished game:
    for every step t of the game:
        θ ← θ + α · (G_t − b) · ∇ log π(a_t | s_t)
    G_t = returns-to-go: reward collected from step t until the end (discounted by γ)
    b   = baseline: "what return is normal?" (0 = no baseline)

In words: actions that came before a BETTER-than-normal future get their probability
pushed UP; actions before a WORSE-than-normal future get pushed DOWN.
"""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)          # other programs share this CPU


class Policy(nn.Module):
    """
    The policy network: state in, action PROBABILITIES out.
        4 numbers (cart position, cart speed, pole angle, pole spin speed)
        → 32 hidden units → 2 logits → softmax → [P(push left), P(push right)]
    The weights of this network ARE θ.
    """

    def __init__(self, n_inputs=4, n_actions=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_inputs, hidden), nn.Tanh(), nn.Linear(hidden, n_actions))

    def forward(self, state):
        logits = self.net(state)                      # raw scores, any size, can be negative
        return torch.softmax(logits, dim=-1)          # probabilities: positive, add up to 1


def returns_to_go(rewards, gamma):
    """
    G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...  for every step t, computed backwards.
    Same idea as Monte Carlo: wait until the game is over, then add up what came AFTER each step.
    Example (γ = 0.9, rewards 1, 1, 1):  G = [1 + 0.9·1.9, 1 + 0.9·1, 1] = [2.71, 1.9, 1]
    """
    G = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def play_episode(env, policy):
    """Play ONE full game with the current policy. Returns log π(a_t|s_t) of each step and the rewards."""
    s, _ = env.reset()
    log_probs, rewards = [], []
    done = False
    while not done:
        probs = policy(torch.as_tensor(s, dtype=torch.float32))
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()                             # ROLL THE AGENT'S DICE (its own probabilities)
        log_probs.append(dist.log_prob(a))            # remember log π(a|s): needed for the gradient
        s, r, terminated, truncated, _ = env.step(int(a))
        rewards.append(r)
        done = terminated or truncated
    return log_probs, rewards


def train(env, episodes=600, lr=0.01, gamma=0.99, baseline="none", rtg=True, seed=0, hidden=32):
    """
    baseline = "none"  : weight each step by G_t
             = "mean"  : weight each step by G_t − (average G_t of this game)
    rtg      = True    : use returns-to-go G_t (only rewards AFTER step t)
             = False   : use the whole-game return G_0 for every step
    Returns the trained policy and the score (total reward) of every training game.
    """
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    policy = Policy(hidden=hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scores = np.zeros(episodes)

    for ep in range(episodes):
        log_probs, rewards = play_episode(env, policy)      # 1. play a whole game, no learning yet
        scores[ep] = sum(rewards)

        G = returns_to_go(rewards, gamma)                    # 2. how good was the future after each step?
        if not rtg:
            G = np.full(len(rewards), G[0])                  #    (variant: credit every step with the total)
        if baseline == "mean":
            G = G - G.mean()                                 # 3. better or worse than normal?

        # 4. REINFORCE: loss = −Σ (G_t − b)·log π(a_t|s_t).
        #    Minimising this loss = gradient ASCENT on expected return.
        #    PyTorch's backward() computes ∇log π for us (the chain rule, automatically).
        loss = -(torch.as_tensor(G, dtype=torch.float32) * torch.stack(log_probs)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()                                     # θ ← θ + α·Σ (G_t − b)·∇log π  (Adam flavour)

    return policy, scores


def evaluate(env, policy, games=20, seed=12345, greedy=False):
    """Play test games with no learning. greedy=True: always take the most likely action."""
    totals = []
    env.reset(seed=seed)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(games):
        s, _ = env.reset()
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                probs = policy(torch.as_tensor(s, dtype=torch.float32))
            a = int(torch.argmax(probs)) if greedy else int(torch.multinomial(probs, 1, generator=gen))
            s, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
        totals.append(total)
    return float(np.mean(totals))


def moving_average(x, window=20):
    # Smooth a noisy learning curve: average of the last `window` games.
    return np.convolve(x, np.ones(window) / window, mode="valid")
