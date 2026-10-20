"""Reinforcement aspect: active learning where a multi-armed bandit chooses the query strategy.

Labels are expensive in all three thesis topics (wet-lab solubility, destructive egg breaking,
sensory panel + 48 h storage). Each round the agent picks one *arm* (a query strategy), the
oracle labels a batch, the model is retrained and the reward is the improvement of the
validation score. This is the "Active Learning by Learning" idea of Hsu & Lin (AAAI 2015),
here with the epsilon-greedy / Monte-Carlo value estimate and UCB1 from the lecture.

The test set is never used for the reward or for any decision; it is only for reporting.
"""
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

ARMS = ["random", "uncertainty", "diversity"]


# --------------------------------------------------------------------- query strategies
def q_random(ctx, k, rng):
    return rng.choice(ctx["pool"], size=k, replace=False)


def q_uncertainty(ctx, k, rng):
    u = ctx["uncertainty"](ctx["model"], ctx["X"][ctx["pool"]])
    return ctx["pool"][np.argsort(-u)[:k]]


def q_diversity(ctx, k, rng):
    """Greedy k-center: repeatedly take the pool point farthest from everything labelled."""
    Z = ctx["Z"]
    pool = ctx["pool"]
    d = ctx["min_dist"][pool].copy()
    chosen = []
    for _ in range(k):
        j = int(np.argmax(d))
        chosen.append(pool[j])
        d = np.minimum(d, np.linalg.norm(Z[pool] - Z[pool[j]], axis=1))
    return np.array(chosen)


QUERY = {"random": q_random, "uncertainty": q_uncertainty, "diversity": q_diversity}


# --------------------------------------------------------------------- uncertainty helpers
def forest_std(model, X):
    """Regression uncertainty: spread of per-tree predictions of a random forest."""
    return np.stack([t.predict(X) for t in model.estimators_]).std(axis=0)


def proba_entropy(model, X):
    p = np.clip(model.predict_proba(X), 1e-12, 1)
    return -(p * np.log(p)).sum(axis=1)


# --------------------------------------------------------------------- bandit policies
class Fixed:
    def __init__(self, arm):
        self.arm = arm

    def select(self, t, rng):
        return self.arm

    def update(self, arm, reward):
        pass


class EpsilonGreedy:
    """Monte-Carlo average reward per arm; explore with probability epsilon."""

    def __init__(self, arms, epsilon=0.2):
        self.arms, self.eps = arms, epsilon
        self.n = {a: 0 for a in arms}
        self.q = {a: 0.0 for a in arms}

    def select(self, t, rng):
        untried = [a for a in self.arms if self.n[a] == 0]
        if untried:
            return untried[0]
        if rng.random() < self.eps:
            return self.arms[rng.integers(len(self.arms))]
        return max(self.arms, key=self.q.get)

    def update(self, arm, reward):
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]


class UCB1(EpsilonGreedy):
    def __init__(self, arms, c=1.0):
        super().__init__(arms)
        self.c = c

    def select(self, t, rng):
        untried = [a for a in self.arms if self.n[a] == 0]
        if untried:
            return untried[0]
        total = sum(self.n.values())
        return max(self.arms, key=lambda a: self.q[a] + self.c * np.sqrt(2 * np.log(total) / self.n[a]))


def make_policy(name):
    if name in ARMS:
        return Fixed(name)
    if name == "bandit-eps":
        return EpsilonGreedy(ARMS, epsilon=0.2)
    if name == "bandit-ucb":
        return UCB1(ARMS, c=1.0)
    raise ValueError(name)


# --------------------------------------------------------------------- main loop
def run_active_learning(model, X_pool, y_pool, X_val, y_val, X_test, y_test, *, policy,
                        score, uncertainty, n_init, batch, rounds, seed,
                        higher_is_better=True, stratify_init=False):
    """Return one record per round: labelled size, arm, val score, test score, reward."""
    rng = np.random.default_rng(seed)
    X_pool = np.asarray(X_pool)
    Z = StandardScaler().fit_transform(X_pool)
    n = len(X_pool)

    if stratify_init:
        # one example per class first so a classifier can be fitted at all
        classes = np.unique(y_pool)
        labelled = [rng.choice(np.flatnonzero(y_pool == c)) for c in classes]
        rest = np.setdiff1d(np.arange(n), labelled)
        labelled += list(rng.choice(rest, n_init - len(labelled), replace=False))
        labelled = np.array(labelled)
    else:
        labelled = rng.choice(n, n_init, replace=False)
    pool_mask = np.ones(n, bool)
    pool_mask[labelled] = False
    min_dist = np.full(n, np.inf)
    for i in labelled:
        min_dist = np.minimum(min_dist, np.linalg.norm(Z - Z[i], axis=1))

    pol = make_policy(policy)
    sign = 1 if higher_is_better else -1

    def fit_eval(idx):
        m = clone(model).fit(X_pool[idx], y_pool[idx])
        return m, score(y_val, m.predict(X_val)), score(y_test, m.predict(X_test))

    m, v, te = fit_eval(labelled)
    log = [{"round": 0, "n_labelled": len(labelled), "arm": "init", "val": v, "test": te,
            "reward": 0.0}]
    for t in range(1, rounds + 1):
        arm = pol.select(t, rng)
        ctx = {"pool": np.flatnonzero(pool_mask), "X": X_pool, "Z": Z, "model": m,
               "uncertainty": uncertainty, "min_dist": min_dist}
        new = QUERY[arm](ctx, min(batch, pool_mask.sum()), rng)
        pool_mask[new] = False
        for i in new:
            min_dist = np.minimum(min_dist, np.linalg.norm(Z - Z[i], axis=1))
        labelled = np.concatenate([labelled, new])
        m, v_new, te = fit_eval(labelled)
        reward = sign * (v_new - v) / (abs(v) + 1e-9)    # relative validation improvement
        pol.update(arm, reward)
        v = v_new
        log.append({"round": t, "n_labelled": len(labelled), "arm": arm, "val": v, "test": te,
                    "reward": reward})
    return log
