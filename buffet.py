import numpy as np
from scipy.stats import poisson, gamma
from scipy.special import gammaln, expit
from tqdm import tqdm_notebook
from types import SimpleNamespace


def lnfact(x):
    return gammaln(x + 1)


class Gamma:
    def __init__(self, params):
        params = params if isinstance(params, tuple) else (params, None, None)
        val, a, b = params
        self.a = a
        self.b = b
        self.is_dist = a is not None and b is not None
        self.val = val

    def sample(self, a, b):
        return gamma.rvs(self.a + a, scale=1 / (self.b + b))


class Feature:
    def __init__(self, data, sigma_x, sigma_w):
        self.X = data
        self.XXT = self.X.T @ self.X
        self.N, self.D = data.shape

        self.sigma_x = Gamma(sigma_x)
        self.sigma_w = Gamma(sigma_w)

        self.var_x = self.sigma_x.val ** 2
        self.var_w = self.sigma_w.val ** 2

        self.W = SimpleNamespace(mu=None, sigma=None, h=None, info=None)

    def log_likelihood_xi(self, zi, i, noise=0):
        # mu and sigma for xi using dist of W
        mu = zi @ self.W.mu
        sigma = zi @ self.W.sigma @ zi.T + self.var_x + noise * self.var_w

        # log likelihood for single xi
        ll = self.D * np.log(sigma) + np.power(self.X[i] - mu, 2).sum() / sigma
        return -ll / 2

    def calc_M(self, Z):
        # M = (Z'Z + sigma_x^2 / sigma_w^2 * I)^-1
        K = Z.shape[1]
        return np.linalg.inv(Z.T @ Z + self.var_x / self.var_w * np.eye(K))

    def log_likelihood(self, Z):
        _, K = Z.shape
        M = self.calc_M(Z)
        lp = -self.N * np.log(2 * np.pi * self.var_x)
        lp -= K * np.log(self.var_w / self.var_x)
        lp += np.log(np.linalg.det(M))
        XZ = self.X.T @ Z
        lp = lp * self.D - np.trace(self.XXT - XZ @ M @ XZ.T) / self.var_x
        return lp / 2

    def sample_sigma(self, Z):
        if self.sigma_w.is_dist or self.sigma_x.is_dist:
            W_mu, W_sigma = self.post_W(Z)

        if self.sigma_x.is_dist:
            var_x = self.D * (Z @ W_sigma @ Z.T).diagonal().sum()
            var_x += ((self.X - Z @ W_mu) ** 2).sum()
            n = self.N * self.D / 2
            self.var_x = 1 / self.sigma_x.sample(n, var_x / 2)

        if self.sigma_w.is_dist:
            var_w = W_sigma.trace() * self.D + np.power(W_mu, 2).sum()
            n = Z.shape[1] * self.D / 2
            self.var_w = 1 / self.sigma_w.sample(n, var_w / 2)

    def post_W(self, Z):
        # mean and covariange of W
        M = self.calc_M(Z)
        return M @ Z.T @ self.X, self.var_x * M

    def weights(self, Z):
        # E(W|X,Z)
        return self.post_W(Z)[0]

    def update(self, zi, i, sub=False):
        sub = 1 if sub else -1
        self.W.info += sub * np.outer(zi, zi) / self.var_x
        self.W.h += sub * np.outer(zi, self.X[i]) / self.var_x


class IndianBuffet:
    def __init__(self, data, alpha, sigma_xs, sigma_ws):
        self.N = data[0].shape[0]
        self.feats = [Feature(d, sx, sw) for d, sx, sw in zip(data, sigma_xs, sigma_ws)]

        self.alpha_dist = Gamma(alpha)
        self.alpha = self.alpha_dist.val

        self.init_z()
        self.Zs = []

    def init_z(self):
        Z = np.ones((0, 0))
        for i in range(1, self.N + 1):
            zi = np.random.uniform(0, 1, Z.shape[1]) < Z.sum(axis=0) / i
            k = poisson.rvs(self.alpha / i)
            Z = np.block([[Z, np.zeros((Z.shape[0], k))], [zi, np.ones((1, k))]])
        self.Z = Z
        self.K = Z.shape[1]
        self.m = Z.sum(axis=0)

    def prior(self):
        logp = self.K * np.log(self.alpha)
        logp -= lnfact(np.unique(self.Z, axis=1, return_counts=True)[1]).sum()
        logp -= (self.alpha / np.arange(1, self.N + 1)).sum()
        facts = lnfact(self.N - self.m) + lnfact(self.m - 1) - lnfact(self.N)
        return logp + facts.sum()

    def log_posterior(self):
        return self.prior() + sum(f.log_likelihood(self.Z) for f in self.feats)

    def add_features(self, k, i):
        self.Z = np.hstack((self.Z, np.zeros((self.N, k))))
        self.Z[i, self.K:] = 1
        self.m = np.hstack((self.m, np.ones(k)))

        for feat in self.feats:
            zero = np.zeros((self.K, k))
            feat.W.info = np.block([[feat.W.info, zero], [zero.T, np.eye(k) / feat.var_w]])
            feat.W.h = np.vstack((feat.W.h, np.zeros((k, feat.D))))
        self.K += k

    def remove_features(self, cols):
        self.Z = np.delete(self.Z, cols, axis=1)
        self.m = np.delete(self.m, cols)
        self.K -= len(cols)

        for feat in self.feats:
            feat.W.info = np.delete(feat.W.info, cols, axis=0)
            feat.W.info = np.delete(feat.W.info, cols, axis=1)
            feat.W.h = np.delete(feat.W.h, cols, axis=0)

    def log_likelihood_xi(self, *args, **kwargs):
        return sum(f.log_likelihood_xi(*args, **kwargs) for f in self.feats)

    def sample_z(self):
        order = np.random.permutation(self.N)
        for c, i in enumerate(order):
            if c % 5 == 0:
                for feat in self.feats:
                    feat.W.mu, feat.W.sigma = feat.post_W(self.Z)
                    feat.W.info = np.linalg.inv(feat.W.sigma)
                    feat.W.h = feat.W.info @ feat.W.mu

            zi = self.Z[i]

            for feat in self.feats:
                feat.update(zi, i, sub=True)
                feat.W.sigma = np.linalg.inv(feat.W.info)
                feat.W.mu = feat.W.sigma @ feat.W.h

            # Remove point from counts
            mi = self.m - zi

            prev = np.copy(zi)
            for k in np.nonzero(mi > 0)[0]:
                zi[k] = 0
                lp0 = np.log(self.N - mi[k]) + self.log_likelihood_xi(zi, i)
                zi[k] = 1
                lp1 = np.log(mi[k]) + self.log_likelihood_xi(zi, i)
                zi[k] = int(np.random.uniform(0, 1) < expit(lp1 - lp0))
            self.m += zi - prev

            # Metropolis-Hastings step described in Meeds et al
            netk = poisson.rvs(self.alpha / self.N) - np.count_nonzero(mi <= 0)

            # Calculate the loglikelihoods
            lpold = self.log_likelihood_xi(zi, i)
            lpnew = self.log_likelihood_xi(zi, i, noise=netk)
            lpaccept = min(0.0, lpnew - lpold)
            lpreject = np.log(max(1.0 - np.exp(lpaccept), 1e-100))

            if np.random.uniform(0, 1) < expit(lpaccept - lpreject):
                if netk > 0:
                    self.add_features(netk, i)
                elif netk < 0:
                    empty = np.nonzero(mi <= 0)[0]
                    self.remove_features(empty[:-netk])

            for feat in self.feats:
                feat.update(self.Z[i], i)

    def gibbs_sample(self):
        self.sample_z()
        if self.alpha_dist.is_dist:
            self.alpha = self.alpha_dist.sample(self.m.sum(), self.N)
        for feat in self.feats:
            feat.sample_sigma(self.Z)

    def run_sampler(self, iters=5000, burn_in=3000, thin=10):
        self.Zs = []
        for i in tqdm_notebook(range(iters)):
            self.gibbs_sample()
            print(self.K)
            if i > burn_in and i % thin == 0:
                self.Zs.append(np.copy(self.Z))
        return self.Zs

    def predict(self, Xs, k=None):
        best = (-float('inf'), None, None)
        for Z in self.Zs:
            # calculate log prior over each row
            m = Z.sum(axis=0) - Z
            ps = np.where(Z == 1, m, self.N - m) / self.N
            lps = np.sum(np.log(ps), axis=1)

            for f, X in zip(self.feats[:-1], Xs[:-1]):
                # update prior with likelihood as mv normal centered at Z @ E(W | Z)
                X_bar = Z @ f.weights(Z)
                diffs = np.sum((X - X_bar) ** 2, axis=1) / (2 * f.var_x)
                lps -= diffs + np.log(2 * np.pi) / 2 + np.log(f.var_x) * f.D / 2
            i = np.argmax(lps)

            # Save result if z[i] is most probable generator of X
            if lps[i] > best[0]:
                best = lps[i], Z, i

        best_lp, best_Z, best_i = best
        W = self.feats[-1].weights(best_Z)
        X_w_pred = best_Z[best_i] @ W

        return Xs[-1][np.argmax(X_w_pred)] == 1
