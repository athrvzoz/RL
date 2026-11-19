"""
The toy "chat assistant" world used by every script in this folder.

A real chat model has ~100,000 words in its vocabulary and billions of numbers.
Ours has 10 words and a small table, but it is the SAME kind of RL problem:

    state   = (the customer's message, the last word written so far)
    action  = the next word (or <end> to stop)
    episode = one whole reply, at most 8 words
    reward  = ONE number, given only when the reply is finished

The policy is a table of "logits" (scores), one row per state, like a Q-table:
    theta[prompt, last_word, next_word]
softmax turns a row of scores into probabilities, and the reply is written by
rolling dice with those probabilities, one word at a time.

Also in here: the HIDDEN "human" taste (true_score) that the learner never sees
directly, only through comparisons "reply A is better than reply B".
"""
import numpy as np

WORDS = ["<end>", "hi", "sorry", "please", "tracking", "refund", "reset", "link", "great", "ok"]
END = 0
V = len(WORDS)              # 10 possible actions
START = V                   # extra row index meaning "nothing written yet"
MAX_LEN = 8                 # a reply is cut off after 8 words

PROMPTS = ["Where is my parcel?", "I was charged twice!", "I forgot my password."]
P = len(PROMPTS)

# What a helpful reply should mention, per customer message.
RELEVANT = [{"tracking", "link"}, {"sorry", "refund"}, {"reset", "link"}]
CONTENT = {"tracking", "refund", "reset", "link"}
POLITE = {"hi", "please", "sorry"}
FILLER = {"great", "ok"}

# A few replies written by "human demonstrators" for supervised fine-tuning (stage 2).
DEMOS = [
    ["hi tracking link", "tracking link please", "hi please tracking", "ok tracking link",
     "link", "great tracking"],
    ["sorry refund", "hi sorry refund", "refund please", "sorry ok refund", "sorry",
     "great refund"],
    ["reset link", "hi reset link please", "reset please", "ok reset link", "link",
     "great reset"],
]


def true_score(prompt, words):
    """
    The hidden human taste. The learner NEVER calls this. We only use it to
    (1) simulate the labellers' choices and (2) check the real quality at the end.
    """
    if len(words) == 0:
        return -3.0                                   # an empty reply is useless
    score, seen = 0.0, set()
    for w in words:
        if w in seen:
            score -= 1.5                              # repeating yourself is annoying
            continue
        seen.add(w)
        if w in RELEVANT[prompt]:
            score += 2.0                              # actually helps the customer
        elif w in CONTENT:
            score -= 1.0                              # off-topic ("refund" for a password)
        elif w in POLITE:
            score += 0.5                              # a little politeness is nice
        elif w in FILLER:
            score -= 0.3                              # empty filler
    score -= 0.5 * max(0, len(words) - 4)             # too long
    return score


def human_prefers_first(prompt, words_a, words_b, rng):
    """
    A simulated labeller compares two replies. Real people are noisy, so we use
    the Bradley-Terry rule itself:  P(A preferred) = sigmoid(score_A - score_B).
    """
    d = true_score(prompt, words_a) - true_score(prompt, words_b)
    return rng.random() < sigmoid(d)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(logits):
    z = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def to_words(tokens):
    return [WORDS[t] for t in tokens if t != END]


def sample_replies(theta, prompts, rng):
    """
    Write one reply for every prompt in `prompts` (all at once, word by word).
    Returns a list of token lists. A token list includes the <end> token if
    the policy chose to stop (that choice is an action too!).
    """
    n = len(prompts)
    last = np.full(n, START)
    alive = np.ones(n, dtype=bool)
    replies = [[] for _ in range(n)]
    for _ in range(MAX_LEN):                                      # after 8 words: cut off
        probs = softmax(theta[prompts, last])                     # (n, V) action probabilities
        u = rng.random(n)[:, None]
        tok = (probs.cumsum(axis=1) < u).sum(axis=1).clip(0, V - 1)   # roll the dice
        for i in np.flatnonzero(alive):
            replies[i].append(int(tok[i]))
            if tok[i] == END:                                     # the policy chose to stop
                alive[i] = False
        last = np.where(alive, tok, last)
        if not alive.any():
            break
    return replies


def flatten(prompts, replies):
    """
    Turn a batch of replies into one long list of (prompt, state, action, which_reply)
    so we can compute log-probabilities and gradients for every word at once.
    """
    pp, ss, aa, rid = [], [], [], []
    for i, (p, toks) in enumerate(zip(prompts, replies)):
        last = START
        for a in toks:
            pp.append(p); ss.append(last); aa.append(a); rid.append(i)
            last = a
    return np.array(pp), np.array(ss), np.array(aa), np.array(rid)


def token_logprobs(theta, flat):
    """log pi(action | state) for every word in the flattened batch."""
    pp, ss, aa, _ = flat
    logits = theta[pp, ss]                                        # (T, V)
    logits = logits - logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return logp[np.arange(len(aa)), aa]


def reply_logprobs(theta, flat, n):
    """log pi(whole reply) = SUM of the log-probabilities of its words."""
    return np.bincount(flat[3], weights=token_logprobs(theta, flat), minlength=n)


def grad_of_logprob(theta, flat, coef):
    """
    Gradient of  SUM_t coef_t * log pi(a_t | s_t)  with respect to the table theta.
    For a softmax row the derivative of log pi(a) is simply  onehot(a) - pi.
    (Each word gets its own coefficient coef_t.)
    """
    pp, ss, aa, _ = flat
    probs = softmax(theta[pp, ss])                                # (T, V)
    g_rows = -probs * coef[:, None]
    g_rows[np.arange(len(aa)), aa] += coef
    grad = np.zeros_like(theta)
    np.add.at(grad, (pp, ss), g_rows)
    return grad


def features(prompt, words):
    """
    What the REWARD MODEL can see: how many times each word appears (per prompt).
    This is its built-in blind spot: it can't tell "refund" once from "refund" x8
    except by counting, and it adds the counts up in a straight line.
    """
    phi = np.zeros(P * V)
    for w in words:
        phi[prompt * V + WORDS.index(w)] += 1.0
    return phi


def sft(demos=DEMOS, epochs=60, lr=0.5):
    """
    Stage 2, supervised fine-tuning: copy the demonstrators.
    Plain "make the demo words more likely" (maximum likelihood), started from
    all-zero scores (= every word equally likely, our stand-in for a pretrained
    model that can write words but has no idea what a good reply is).
    We stop early on purpose, so the SFT model is still a bit random.
    """
    theta = np.zeros((P, V + 1, V))
    prompts, replies = [], []
    for p, lst in enumerate(demos):
        for text in lst:
            prompts.append(p)
            replies.append([WORDS.index(w) for w in text.split()] + [END])
    flat = flatten(prompts, replies)
    for _ in range(epochs):
        theta += lr * grad_of_logprob(theta, flat, np.ones(len(flat[2]))) / len(replies)
    return theta


def evaluate(theta, theta_ref, rm_w, rng, n_per_prompt=400):
    """
    Write many replies and measure:
      rm   = average score from the learned reward model   (what RL optimises)
      true = average hidden human score                    (what we actually want)
      kl   = how far the policy has drifted from the SFT model (average log-ratio)
      length, repeats = simple statistics of the replies
    """
    prompts = np.repeat(np.arange(P), n_per_prompt)
    replies = sample_replies(theta, prompts, rng)
    flat = flatten(prompts, replies)
    n = len(replies)
    kl = reply_logprobs(theta, flat, n) - reply_logprobs(theta_ref, flat, n)
    words = [to_words(r) for r in replies]
    rm = np.array([features(p, w) @ rm_w for p, w in zip(prompts, words)])
    true = np.array([true_score(p, w) for p, w in zip(prompts, words)])
    length = np.array([len(w) for w in words])
    repeats = np.array([len(w) - len(set(w)) for w in words])
    return dict(rm=rm.mean(), true=true.mean(), kl=kl.mean(), length=length.mean(),
                repeats=repeats.mean())
