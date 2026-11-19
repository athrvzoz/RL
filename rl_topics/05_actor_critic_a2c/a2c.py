"""
A2C: Advantage Actor-Critic (the shared engine for every script in this folder).

Two small neural networks work together:
    ACTOR  pi(a|s)  the PLAYER: looks at the state, gives a PROBABILITY for each action.
    CRITIC V(s)     the COACH:  looks at the state, guesses "how much total reward
                                from here?" (the same V(s) as value iteration,
                                but learned from experience, like Q-learning).

After every few steps the coach compares what really happened with what it
expected. The difference (the "advantage", like Q-learning's surprise) tells
the player: "that move was better / worse than usual, do it more / less".

"A2C" = the synchronous version: N copies of the game run side by side
(gymnasium vector envs) and we learn from all of them together.

This file only defines things; run cartpole_a2c.py, compare_reinforce.py, ...
"""
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)          # other programs share the CPU


def mlp(n_in, n_out, hidden=64):
    # A small neural network: input -> 64 -> 64 -> output, tanh in between.
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


class ActorCritic(nn.Module):
    """
    shared=False: two separate networks (actor and critic share nothing).
    shared=True : one common "body" and two small "heads" on top of it.
    """
    def __init__(self, n_obs, n_actions, shared=False, hidden=64):
        super().__init__()
        self.shared = shared
        if shared:
            self.body = nn.Sequential(nn.Linear(n_obs, hidden), nn.Tanh(),
                                      nn.Linear(hidden, hidden), nn.Tanh())
            self.actor_head = nn.Linear(hidden, n_actions)
            self.critic_head = nn.Linear(hidden, 1)
        else:
            self.actor = mlp(n_obs, n_actions, hidden)
            self.critic = mlp(n_obs, 1, hidden)

    def forward(self, obs):
        """Returns (logits, V). logits -> softmax -> action probabilities."""
        if self.shared:
            h = self.body(obs)
            return self.actor_head(h), self.critic_head(h).squeeze(-1)
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def value(self, obs):
        return self.forward(obs)[1]


def to_t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def train_a2c(env_id="CartPole-v1", total_steps=200_000, n_envs=8, n_steps=5,
              gamma=0.99, lr=7e-4, ent_coef=0.01, vf_coef=0.5, shared=False,
              seed=0, checkpoints=(), log_every=0):
    """
    Trains an actor-critic. Returns (model, history).

    total_steps : environment steps in total, counted over ALL parallel games
    n_envs      : how many copies of the game run side by side
    n_steps     : how many steps each copy plays before we learn (the "n" in n-step)
    ent_coef    : entropy bonus weight (keeps the actor from becoming certain too early)
    checkpoints : step counts at which to save a copy of the model (for charts)
    """
    torch.manual_seed(seed)
    envs = gym.make_vec(env_id, num_envs=n_envs, vectorization_mode="sync",
                        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    n_obs = envs.single_observation_space.shape[0]
    n_act = envs.single_action_space.n
    model = ActorCritic(n_obs, n_act, shared=shared)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {"ep_end_step": [], "ep_return": [], "update_step": [], "entropy": [],
            "actor_loss": [], "critic_loss": [], "adv_std": [], "snapshots": {}}
    obs, _ = envs.reset(seed=seed)
    running = np.zeros(n_envs)                 # score of the game each copy is playing now
    step, start = 0, time.time()
    checkpoints = sorted(checkpoints)

    while step < total_steps:
        # ---------- 1. PLAY n_steps in every copy (no learning yet) ----------
        logps, values, ents, rewards, dones = [], [], [], [], []
        for _ in range(n_steps):
            logits, v = model(to_t(obs))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()                                  # actor ROLLS a die
            obs, r, terminated, truncated, info = envs.step(a.numpy())
            game_r = r.copy()                                  # the real game reward (for scores)
            r = r.astype(np.float64)
            # Time limit hit (truncated): the game did NOT really end, the future still
            # counts. Add gamma * V(last state) to the reward, then cut the chain.
            if truncated.any():
                for i in np.flatnonzero(truncated & ~terminated):
                    with torch.no_grad():
                        r[i] += gamma * model.value(to_t(info["final_obs"][i])).item()
            logps.append(dist.log_prob(a))
            values.append(v)
            ents.append(dist.entropy())
            rewards.append(to_t(r))
            dones.append(to_t(terminated | truncated))
            step += n_envs
            # score keeping for the learning curve
            running += game_r
            for i in np.flatnonzero(terminated | truncated):
                hist["ep_end_step"].append(step)
                hist["ep_return"].append(running[i])
                running[i] = 0.0

        # ---------- 2. n-step RETURNS, built backwards from the critic's guess ----------
        with torch.no_grad():
            R = model.value(to_t(obs))                         # bootstrap: V(state we stopped in)
        returns = []
        for t in reversed(range(n_steps)):
            R = rewards[t] + gamma * R * (1 - dones[t])        # game over -> no future
            returns.append(R)
        returns = torch.stack(returns[::-1])                   # shape (n_steps, n_envs)
        values = torch.stack(values)
        logps = torch.stack(logps)
        entropy = torch.stack(ents).mean()

        # ---------- 3. ADVANTAGE = what happened - what the critic expected ----------
        advantage = returns - values
        actor_loss = -(logps * advantage.detach()).mean()      # detach: don't move the critic here
        critic_loss = advantage.pow(2).mean()                  # MSE: (target - V)^2
        loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy

        # ---------- 4. LEARN: one gradient step for both networks ----------
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

        hist["update_step"].append(step)
        hist["entropy"].append(entropy.item())
        hist["actor_loss"].append(actor_loss.item())
        hist["critic_loss"].append(critic_loss.item())
        hist["adv_std"].append(advantage.detach().std().item())
        while checkpoints and step >= checkpoints[0]:
            hist["snapshots"][checkpoints.pop(0)] = {k: v.clone() for k, v in model.state_dict().items()}
        if log_every and len(hist["update_step"]) % log_every == 0:
            last = hist["ep_return"][-20:]
            print(f"   step {step:>7,}  avg return (last 20 games) {np.mean(last) if last else 0:7.1f}"
                  f"  entropy {entropy.item():.3f}  ({time.time() - start:.0f}s)")
    envs.close()
    return model, hist


@torch.no_grad()
def evaluate(model, env_id="CartPole-v1", games=20, seed=1000, greedy=True):
    """Play full games with the trained actor, no learning. Returns list of total rewards."""
    env = gym.make(env_id)
    scores = []
    for g in range(games):
        obs, _ = env.reset(seed=seed + g)
        done, total = False, 0.0
        while not done:
            logits, _ = model(to_t(obs))
            a = int(torch.argmax(logits)) if greedy else int(torch.distributions.Categorical(logits=logits).sample())
            obs, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
        scores.append(total)
    env.close()
    return scores


def moving_average(x, window=50):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")
