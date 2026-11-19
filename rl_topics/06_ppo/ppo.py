"""
PPO (Proximal Policy Optimization), the shared engine.

Used by cartpole_ppo.py, pendulum_ppo.py, ablation.py and knobs.py.

What PPO learns (two small neural networks):
    ACTOR  : state -> ACTION PROBABILITIES   ("70% push left, 30% push right")
    CRITIC : state -> V(s), one number        ("from here I expect ~45 more reward")

The loop, over and over:
    1. ROLLOUT : play num_steps moves in every copy of the game with the CURRENT policy.
                 Remember state, action, prob of that action, reward, ended?, V(s).
    2. GAE     : for every move, compute the ADVANTAGE A = "how much better than
                 expected did this move turn out?"  (Bellman-style, like Q-learning's
                 target - old guess, but chained over several steps).
    3. UPDATE  : several EPOCHS over the same batch, in MINIBATCHES.
                 Push up probs of moves with A > 0, push down moves with A < 0,
                 BUT stop pushing once a prob has changed by more than ±clip (20%).

Nothing here is specific to one game: CartPole (2 discrete actions) and
Pendulum (1 continuous action) both use it.
"""
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

torch.set_num_threads(2)        # other programs share this CPU


# ───────────────────────── the two networks ─────────────────────────
def mlp(n_in, n_out, last_gain):
    """A small network: n_in numbers -> 64 -> 64 -> n_out numbers."""
    layers = [nn.Linear(n_in, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, n_out)]
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight, last_gain)   # small last layer = start with ~equal probs
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions=None, act_dim=None):
        super().__init__()
        self.discrete = n_actions is not None
        self.critic = mlp(obs_dim, 1, 1.0)                       # state -> V(s)
        if self.discrete:
            self.actor = mlp(obs_dim, n_actions, 0.01)           # state -> one score per action
        else:
            self.actor = mlp(obs_dim, act_dim, 0.01)             # state -> mean of a bell curve
            self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))   # width of the bell curve

    def dist(self, obs):
        """The policy π(·|s): a probability distribution over actions."""
        if self.discrete:
            return Categorical(logits=self.actor(obs))           # softmax of the scores
        return Normal(self.actor(obs), self.log_std.exp())

    def log_prob(self, d, actions):
        lp = d.log_prob(actions)
        return lp if self.discrete else lp.sum(-1)               # several action numbers: add logs

    def entropy(self, d):
        e = d.entropy()
        return e if self.discrete else e.sum(-1)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def make_agent(envs):
    obs_dim = envs.single_observation_space.shape[0]
    space = envs.single_action_space
    if isinstance(space, gym.spaces.Discrete):
        return ActorCritic(obs_dim, n_actions=int(space.n))
    return ActorCritic(obs_dim, act_dim=space.shape[0])


def make_envs(env_id, num_envs, seed):
    """num_envs copies of the game, stepped together. A finished game restarts by itself
    in the same step; the real last state is kept in info["final_obs"]."""
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(num_envs)],
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    envs.reset(seed=seed)
    envs.action_space.seed(seed)
    return envs


# ───────────────────────── 1. ROLLOUT ─────────────────────────
class EpisodeTracker:
    """Adds up rewards per game copy, and records each finished game's total."""
    def __init__(self, num_envs):
        self.running = np.zeros(num_envs)
        self.finished = []                      # list of (env step when it ended, total reward)

    def add(self, rewards, ended, step):
        self.running += rewards
        for i in np.flatnonzero(ended):
            self.finished.append((step, self.running[i]))
            self.running[i] = 0.0


def collect_rollout(envs, agent, obs, num_steps, gamma, reward_scale, tracker, step0):
    n = envs.num_envs
    buf = {k: [] for k in ["obs", "actions", "logprobs", "rewards", "dones", "values"]}
    for t in range(num_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():                                    # just PLAYING, not learning yet
            d = agent.dist(obs_t)
            action = d.sample()                                  # roll the policy's dice
            logprob = agent.log_prob(d, action)                  # log π_old(a|s), saved for later
            value = agent.value(obs_t)                           # critic's guess V(s)
        a = action.numpy()
        if not agent.discrete:
            low, high = envs.single_action_space.low, envs.single_action_space.high
            a = np.clip(a, low, high)                            # the game only accepts [-2, 2]
        next_obs, reward, terminated, truncated, info = envs.step(a)
        tracker.add(reward, terminated | truncated, step0 + (t + 1) * n)

        reward = reward * reward_scale
        # Time limit hit (truncated) but the pole/pendulum did NOT really fail:
        # the future still counts, so add γ·V(real last state) - like (not terminated)
        # in your Q-learning target.
        for i in np.flatnonzero(truncated & ~terminated):
            with torch.no_grad():
                v_last = agent.value(torch.as_tensor(info["final_obs"][i], dtype=torch.float32))
            reward[i] += gamma * v_last.item()

        buf["obs"].append(obs_t)
        buf["actions"].append(action)
        buf["logprobs"].append(logprob)
        buf["rewards"].append(torch.as_tensor(reward, dtype=torch.float32))
        buf["dones"].append(torch.as_tensor(terminated | truncated, dtype=torch.float32))
        buf["values"].append(value)
        obs = next_obs

    with torch.no_grad():
        last_value = agent.value(torch.as_tensor(obs, dtype=torch.float32))
    batch = {k: torch.stack(v) for k, v in buf.items()}          # shape (num_steps, num_envs, ...)
    return batch, last_value, obs


# ───────────────────────── 2. GAE ─────────────────────────
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """
    Walk BACKWARDS through the rollout.
        delta_t = r_t + γ·V(s_{t+1}) - V(s_t)           one-step surprise (a TD error)
        A_t     = delta_t + γ·λ·A_{t+1}                  add the later surprises, faded by γλ
    If the game ended at step t, nothing after t belongs to this game: cut the chain.
    Works on arrays (num_steps,) or (num_steps, num_envs).
    """
    T = len(rewards)
    adv = torch.zeros_like(rewards)
    next_adv = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(T)):
        alive = 1.0 - dones[t]                           # 0 if the game ended at step t
        delta = rewards[t] + gamma * next_value * alive - values[t]
        next_adv = delta + gamma * lam * alive * next_adv
        adv[t] = next_adv
        next_value = values[t]
    returns = adv + values                               # target for the critic: V should be A + V
    return adv, returns


# ───────────────────────── 3. UPDATE ─────────────────────────
def ppo_loss(agent, obs, actions, old_logprobs, adv, returns, clip, vf_coef, ent_coef):
    d = agent.dist(obs)
    new_logprobs = agent.log_prob(d, actions)
    ratio = torch.exp(new_logprobs - old_logprobs)       # r = π_new(a|s) / π_old(a|s)

    # The clipped surrogate: take the SMALLER (more pessimistic) of the two.
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    policy_loss = -torch.min(unclipped, clipped).mean()   # minus: optimisers go DOWNHILL

    value_loss = ((agent.value(obs) - returns) ** 2).mean()   # critic: plain squared error
    entropy = agent.entropy(d).mean()                         # how "spread out" the probs are
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return total, policy_loss, value_loss, entropy


def ppo_update(agent, optimizer, data, cfg, rng):
    """Several epochs of minibatch gradient steps on ONE batch of experience."""
    n = len(data["adv"])
    mb_size = n // cfg["minibatches"]
    for epoch in range(cfg["epochs"]):
        order = rng.permutation(n)                                # shuffle every epoch
        for start in range(0, n, mb_size):
            idx = order[start:start + mb_size]
            adv = data["adv"][idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)          # normalise: mean 0, std 1
            total, pl, vl, ent = ppo_loss(agent, data["obs"][idx], data["actions"][idx],
                                          data["logprobs"][idx], adv, data["returns"][idx],
                                          cfg["clip"], cfg["vf_coef"], cfg["ent_coef"])
            optimizer.zero_grad()
            total.backward()                                      # chain rule, done for us
            nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"])
            optimizer.step()

    # After the update: how far did the policy move on this batch?
    with torch.no_grad():
        d = agent.dist(data["obs"])
        log_ratio = agent.log_prob(d, data["actions"]) - data["logprobs"]
        ratio = log_ratio.exp()
        approx_kl = ((ratio - 1) - log_ratio).mean().item()       # ≈ KL(old ‖ new), always ≥ 0
        clip_frac = ((ratio - 1).abs() > cfg["clip_measure"]).float().mean().item()
        a = (data["adv"] - data["adv"].mean()) / (data["adv"].std() + 1e-8)
        _, pl, vl, ent = ppo_loss(agent, data["obs"], data["actions"], data["logprobs"], a,
                                  data["returns"], cfg["clip"], cfg["vf_coef"], cfg["ent_coef"])
    return {"approx_kl": approx_kl, "clip_frac": clip_frac, "policy_loss": pl.item(),
            "value_loss": vl.item(), "entropy": ent.item(),
            "max_ratio": ratio.max().item(), "min_ratio": ratio.min().item()}


DEFAULTS = dict(num_envs=8, num_steps=128, epochs=10, minibatches=8, lr=3e-4, gamma=0.99,
                lam=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                anneal_lr=True, reward_scale=1.0, clip_measure=0.2)


def train(env_id, total_steps, seed=0, verbose=True, snapshot=False, **overrides):
    """
    Returns (agent, history, episodes).
        history  : one dict per update (step, approx_kl, clip_frac, entropy, losses, ...)
        episodes : list of (env step when the game ended, total reward of that game)
    snapshot=True also records 6 moves of the FIRST update: π_old, advantage, π_new, ratio.
    """
    cfg = {**DEFAULTS, **overrides}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    envs = make_envs(env_id, cfg["num_envs"], seed)
    agent = make_agent(envs)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg["lr"], eps=1e-5)
    tracker = EpisodeTracker(cfg["num_envs"])
    obs, _ = envs.reset(seed=seed)

    batch_size = cfg["num_envs"] * cfg["num_steps"]
    n_updates = total_steps // batch_size
    history, start = [], time.time()
    for u in range(n_updates):
        if cfg["anneal_lr"]:                                      # slowly lower the step size
            optimizer.param_groups[0]["lr"] = cfg["lr"] * (1 - u / n_updates)
        step0 = u * batch_size
        batch, last_value, obs = collect_rollout(envs, agent, obs, cfg["num_steps"], cfg["gamma"],
                                                 cfg["reward_scale"], tracker, step0)
        adv, returns = compute_gae(batch["rewards"], batch["values"], batch["dones"], last_value,
                                   cfg["gamma"], cfg["lam"])
        # flatten (num_steps, num_envs) -> one long list of moves
        data = {"obs": batch["obs"].flatten(0, 1), "actions": batch["actions"].flatten(0, 1),
                "logprobs": batch["logprobs"].flatten(0, 1),
                "adv": adv.flatten(0, 1), "returns": returns.flatten(0, 1)}
        stats = ppo_update(agent, optimizer, data, cfg, rng)

        stats["step"] = (u + 1) * batch_size
        recent = [r for s, r in tracker.finished[-20:]]
        stats["avg_return"] = float(np.mean(recent)) if recent else float("nan")
        if snapshot and u == 0:
            stats["snapshot"] = first_update_snapshot(agent, data)
        history.append(stats)
        if verbose and (u % max(1, n_updates // 15) == 0 or u == n_updates - 1):
            print(f"  step {stats['step']:>7,}  avg return (last 20 games) {stats['avg_return']:7.1f}  "
                  f"approx_kl {stats['approx_kl']:.4f}  clip_frac {stats['clip_frac']:.3f}  "
                  f"entropy {stats['entropy']:.3f}")
    envs.close()
    if verbose:
        print(f"  ({time.time() - start:.0f} s)")
    return agent, history, tracker.finished


def first_update_snapshot(agent, data, n=6):
    """Pick a few moves and show what the update did to their probabilities."""
    idx = torch.arange(n) * 37                                    # spread out through the batch
    with torch.no_grad():
        d = agent.dist(data["obs"][idx])
        new = agent.log_prob(d, data["actions"][idx]).exp()
    old = data["logprobs"][idx].exp()
    return [{"obs": data["obs"][i].tolist(), "action": data["actions"][i].tolist(),
             "adv": data["adv"][i].item(), "p_old": old[k].item(), "p_new": new[k].item(),
             "ratio": (new[k] / old[k]).item()} for k, i in enumerate(idx.tolist())]


def evaluate(env_id, agent, games=20, seed=12345):
    """Play with NO randomness: most likely action (CartPole) or the bell-curve's centre
    (Pendulum). No learning. Returns the list of game totals."""
    env = gym.make(env_id)
    totals = []
    for g in range(games):
        obs, _ = env.reset(seed=seed + g)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                d = agent.dist(torch.as_tensor(obs, dtype=torch.float32))
                a = d.probs.argmax().item() if agent.discrete else d.mean.numpy()
            if not agent.discrete:
                a = np.clip(a, env.action_space.low, env.action_space.high)
            obs, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
        totals.append(total)
    env.close()
    return totals


def smooth(x, window=10):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")
