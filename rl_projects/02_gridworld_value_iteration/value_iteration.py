"""
Value iteration (RL From Zero, chapter 8).

Repeat the Bellman optimality equation on every square until nothing changes:

    V(s) <- max over a of  sum over s' of  P(s'|s,a) * [ R + gamma * V(s') ]
            ───────────    ──────────────────────────────────────────────
            "you choose"   "world rolls dice": Q(s, a), the value of action a

Then the best policy is: in each square, take the action with the biggest Q.
"""
from gridworld import ACTIONS


def q_value(env, V, s, a, gamma):
    # Expected (reward now + discounted value of where I land), averaged over the slip dice.
    return sum(p * (r + gamma * V[s2]) for p, s2, r in env.transitions(s, a))


def bellman_sweep(env, V, gamma):
    """One sweep: apply Bellman once to every square. Returns (new V, biggest change)."""
    new_V = dict(V)
    biggest_change = 0.0
    for s in env.states:
        if env.is_terminal(s):
            continue                      # game over: no future, value stays 0
        new_V[s] = max(q_value(env, V, s, a, gamma) for a in ACTIONS)
        biggest_change = max(biggest_change, abs(new_V[s] - V[s]))
    return new_V, biggest_change


def greedy_policy(env, V, gamma):
    # pi*(s) = argmax_a Q(s, a)
    return {s: max(ACTIONS, key=lambda a: q_value(env, V, s, a, gamma))
            for s in env.states if not env.is_terminal(s)}


def value_iteration(env, gamma=0.9, theta=1e-6, on_sweep=None):
    """
    Start from V = 0 everywhere and sweep until the biggest change is below theta.
    on_sweep(sweep_number, V, change) is called after each sweep, for printing.
    """
    V = {s: 0.0 for s in env.states}
    sweep = 0
    while True:
        V, change = bellman_sweep(env, V, gamma)
        sweep += 1
        if on_sweep:
            on_sweep(sweep, V, change)
        if change < theta:
            break
    return V, greedy_policy(env, V, gamma), sweep
