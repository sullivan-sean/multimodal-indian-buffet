import numpy as np
import scipy.stats as stats
from types import SimpleNamespace
from scipy.special import expit, gammaln
from scipy.linalg import block_diag
from tqdm import tqdm_notebook

# We will be taking log(0) = -Inf, so turn off this warning
np.seterr(divide='ignore')


def lnfact(x):
    return gammaln(x + 1)


def gamma(var, f=None):
    var, a, b = var if isinstance(var, tuple) else (var, None, None)
    return Gamma(var, a=a, b=b, f=f)


class Gamma(float):
    def __init__(self, val, a=None, b=None, f=None):
        super().__init__()
        f = (lambda x: x) if f is None else f
        self.a, self.b, self.f = a, b, f
        self.is_dist = self.a is not None and self.b is not None

    def sample(self, a, b):
        if self.is_dist:
            new_val = stats.gamma.rvs(self.a + a, scale=1 / self.b + b)
            return Gamma(self.f(new_val), a=self.a, b=self.b, f=self.f)
        return self


class Feature:
    def __init__(self, data, var_x, var_w):
        self.X = data
        self.N, self.D = self.X.shape

        self.var_x = gamma(var_x, f=lambda a: 1 / a)
        self.var_w = gamma(var_w, f=lambda a: 1 / a)

        self.W = None

    def set_weights(self, Z):
        # dist of W over P(W | Z, X)
        R = self.get_R(Z)

        mu = R @ Z.T @ self.X
        sigma = self.var_x * R
        info = np.linalg.inv(sigma)
        h = info @ mu

        self.W = SimpleNamespace(mu=mu, sigma=sigma, info=info, h=h)

    def update_weights(self, zi, xi, sub=False):
        zi, xi = zi.reshape(1, -1), xi.reshape(1, -1)
        sub = -1 if sub else 1

        self.W.info += sub * zi.T @ zi / self.var_x
        self.W.h += sub * zi.T @ xi / self.var_x

        self.W.sigma = np.linalg.inv(self.W.info)
        self.W.mu = self.W.sigma @ self.W.h

    def get_R(self, Z):
        # where R = (Z'Z + var_x / var_w * I)^-1
        ZZI = Z.T @ Z + self.var_x / self.var_w * np.eye(Z.shape[1])
        return np.linalg.inv(ZZI)

    def sample_vars(self, Z):
        if self.var_w.is_dist or self.var_x.is_dist:
            self.set_weights(Z)

        if self.var_x.is_dist:
            vs = (Z @ self.W.sigma @ Z.T).diagonal().sum()

            v_x = np.power(self.X - Z @ self.W.mu, 2).sum() + self.D * vs
            n_x = self.N * self.D
            self.var_x = self.var_x.sample(n_x / 2, v_x / 2)

        if self.var_w.is_dist:
            _, K = Z.shape
            v_a = self.W.sigma.trace() * self.D + np.power(self.W.mu, 2).sum()
            n_a = K * self.D
            self.var_w = self.var_w.sample(n_a / 2, v_a / 2)

    def likelihood_Xi(self, Z, i, diff=None):
        zi = Z[i]
        xi = self.X[i]

        mean = zi @ self.W.mu
        sigma = zi @ self.W.sigma @ zi.T + self.var_x

        sigmas = [sigma]

        if diff is not None:
            sigmas.append(sigma + diff * self.var_w)

        p = np.power(xi - mean, 2).sum()
        lls = [(self.D * np.log(sigma) + p / sigma) / -2 for sigma in sigmas]

        return lls if diff is not None else lls[0]


class IndianBuffet:
    def __init__(self, data, alpha, var_xs, var_ws):
        self.feats = [Feature(X, var_x, var_w) for X, var_x, var_w in zip(data, var_xs, var_ws)]
        self.N = data[0].shape[0]

        self.alpha = gamma(alpha)

        self.init_z()

    def init_z(self):
        Z = np.ones((0, 0))
        for i in range(1, self.N + 1):
            zi = np.random.uniform(0, 1, (1, Z.shape[1])) < Z.sum(axis=0) / i
            k = stats.poisson.rvs(self.alpha / i)
            Z = np.block([[Z, np.zeros((Z.shape[0], k))], [zi, np.ones((1, k))]])
        self.Z = Z
        self.K = Z.shape[1]
        self.M = Z.sum(axis=0)

    def gibbs_sample(self):
        self.sample_z()

        if self.alpha.is_dist:
            self.alpha = self.alpha.sample(self.Z.sum(), self.N)

        for feat in self.feats:
            feat.sample_vars(self.Z)

    def sample_z(self):
        """ Take single sample of latent features Z """
        # for each data point
        order = np.random.permutation(self.N)

        for c, i in enumerate(order):
            if c % 5 == 0:
                for feat in self.feats:
                    feat.set_weights(self.Z)

            # remove point from Z, M
            for feat in self.feats:
                feat.update_weights(self.Z[i], feat.X[i], sub=True)
            mi = self.M - self.Z[i, :]

            singletons = np.nonzero(mi <= 0)[0]

            lpz = np.log(np.stack((self.N - mi, mi)))

            for k in np.nonzero(mi > 0)[0]:
                prev = self.Z[i, k]

                self.Z[i, k] = 0
                lpz[0, k] += sum(f.likelihood_Xi(self.Z, i) for f in self.feats)

                self.Z[i, k] = 1
                lpz[1, k] += sum(f.likelihood_Xi(self.Z, i) for f in self.feats)

                on = np.random.uniform(0, 1) < expit(lpz[1, k] - lpz[0, k])
                self.Z[i, k] = int(on)
                self.M[k] += self.Z[i, k] - prev  # assert = self.Z[:, k].sum()

            print(self.M)
            diff = stats.poisson.rvs(self.alpha / self.N) - len(singletons)

            lpold, lpnew = 0, 0
            for f in self.feats:
                lpo, lpn = f.likelihood_Xi(self.Z, i, diff=diff)
                lpold, lpnew = lpold + lpo, lpnew + lpn

            lpaccept = min(0.0, lpnew - lpold)
            lpreject = np.log(max(1.0 - np.exp(lpaccept), 1e-100))

            # run metropolis hastings to add or remove features by diff
            if np.random.uniform(0, 1) < expit(lpaccept - lpreject):
                if diff > 0:
                    self.Z = np.hstack((self.Z, np.zeros((self.N, diff))))
                    self.Z[i, self.K:] = 1
                    self.M = np.hstack((self.M, np.ones(diff)))
                    for f in self.feats:
                        f.W.info = block_diag(f.W.info, np.eye(diff) / f.var_w)
                        f.W.h = np.vstack((f.W.h, np.zeros((diff, f.D))))
                    self.K += diff
                elif diff < 0:
                    dead = [ki for ki in singletons[:-diff]]
                    self.K -= len(dead)
                    self.Z = np.delete(self.Z, dead, axis=1)
                    self.M = np.delete(self.M, dead)
                    for f in self.feats:
                        f.W.info = np.delete(f.W.info, dead, axis=0)
                        f.W.info = np.delete(f.W.info, dead, axis=1)
                        f.W.h = np.delete(f.W.h, dead, axis=0)

            # add point back in
            for f in self.feats:
                f.update_weights(self.Z[i], f.X[i])

    def run_sampler(self, iters=5000, burn_in=3000, thin=10):
        Zs = []
        for i in tqdm_notebook(range(iters)):
            self.gibbs_sample()
            print(self.K)
            if i > burn_in and i % thin == 0:
                Zs.append(self.Z)
        return Zs

    def likelihood_Xs(self, Z):
        # P(X | Z)
        lp = 0
        for feat in self.feats:
            R = feat.get_R(Z)
            lp -= self.N * feat.D * np.log(2 * np.pi * feat.var_x)
            lp -= self.K * feat.D * np.log(feat.var_w / feat.var_x)
            lp -= feat.D * np.log(np.linalg.det(R))
            IZRZ = np.eye(self.N) - Z @ R @ Z.T
            lp -= np.trace(feat.X.T @ IZRZ @ feat.X) / feat.var_x
        return lp / 2

    def prior_Z(self):
        # P(Z | alpha)
        logp = self.K * np.log(self.alpha)
        logp -= lnfact(np.unique(self.Z, axis=1, return_counts=True)[1]).sum()
        logp -= (self.alpha / np.arange(1, self.N + 1)).sum()
        facts = lnfact(self.N - self.M) + lnfact(self.M - 1) - lnfact(self.N)
        return logp + facts.sum()

    def ll(self):
        return self.likelihood_Xs(self.Z) + self.prior_Z()
