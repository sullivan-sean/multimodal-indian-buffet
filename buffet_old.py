"""
PyIBP

Implements fast Gibbs sampling for the linear-Gaussian
infinite latent feature model (IBP).

Copyright (C) 2009 David Andrzejewski (andrzeje@cs.wisc.edu)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy.stats import poisson, gamma
from scipy.special import gammaln, expit
from tqdm import tqdm_notebook

# We will be taking log(0) = -Inf, so turn off this warning
np.seterr(divide='ignore')

def lnfact(x):
    return gammaln(x + 1)

class Feature:
    def __init__(self, data, sigma_x, sigma_a):
        self.X = data
        self.XXT = self.X.T @ self.X
        self.N, self.D = data.shape

        sigma_x = sigma_x if isinstance(sigma_x, tuple) else (sigma_x, None, None)
        sigma_a = sigma_a if isinstance(sigma_a, tuple) else (sigma_a, None, None)

        _, self.sigma_xa, self.sigma_xb = sigma_x
        _, self.sigma_aa, self.sigma_ab = sigma_a

        self.var_x = sigma_x[0] ** 2
        self.var_a = sigma_a[0] ** 2

        self.meanA, self.covarA, self.hA, self.infoA = (None,) * 4

    def logPxi(self, zi, i, noise=0):
        # Mean/covar of xi given posterior over A
        meanLike = np.dot(zi, self.meanA)
        covarLike = np.dot(zi, np.dot(self.covarA, zi.T)) + self.var_x

        # Calculate log-likelihood of a single xi
        covarLike += noise
        ll = self.D * np.log(covarLike) + np.power(self.X[i] - meanLike, 2).sum() / covarLike
        return -ll / 2

    def calcM(self, Z):
        """ Calculate M = (Z' * Z - (sigmax^2) / (sigmaa^2) * I)^-1 """
        K = Z.shape[1]
        return np.linalg.inv(np.dot(Z.T, Z) + self.var_x / self.var_a * np.eye(K))

    def logPX(self, Z):
        """ Calculate collapsed log likelihood of data"""
        M = self.calcM(Z)
        K = Z.shape[1]
        lp = -self.N * self.D * np.log(2 * np.pi * self.var_x)
        lp -= K * self.D * np.log(self.var_a / self.var_x)
        lp += self.D * np.log(np.linalg.det(M))
        XZ = self.X.T @ Z
        lp -= np.trace(self.XXT - XZ @ M @ XZ.T) / self.var_x
        return lp / 2

    def sampleSigma(self, Z):
        """ Sample feature/noise variances """
        K = Z.shape[1]
        # Posterior over feature weights A
        meanA, covarA = self.postA(Z)
        # var_x
        vars = np.dot(Z, np.dot(covarA, Z.T)).diagonal()
        var_x = (np.power(self.X - np.dot(Z, meanA), 2)).sum()
        var_x += self.D * vars.sum()
        n = float(self.N * self.D)
        postShape = self.sigma_xa + n / 2
        postScale = 1 / (self.sigma_xb + var_x / 2)
        tau_x = gamma.rvs(postShape, scale=postScale)
        self.var_x = 1 / tau_x
        # var_a
        var_a = covarA.trace() * self.D + np.power(meanA, 2).sum()
        n = float(K * self.D)
        postShape = self.sigma_aa + n / 2
        postScale = 1 / (self.sigma_ab + var_a / 2)
        tau_a = gamma.rvs(postShape, scale=postScale)
        self.var_a = 1 / tau_a

    def postA(self, Z):
        """ Mean/covar of posterior over weights A """
        M = self.calcM(Z)
        meanA = np.dot(M, np.dot(Z.T, self.X))
        covarA = self.var_x * self.calcM(Z)
        return meanA, covarA

    def weights(self, Z):
        """ Return E[A|X,Z] """
        return self.postA(Z)[0]

    def update(self, zi, xi, sub=False):
        """ Add/remove data i to/from information """
        xi, zi = xi.reshape(1, -1), zi.reshape(1, -1)
        sub = 1 if sub else -1
        self.infoA += sub * ((1 / self.var_x) * np.dot(zi.T, zi))
        self.hA += sub * ((1 / self.var_x) * np.dot(zi.T, xi))


class PyIBP(object):
    def __init__(self, data, alpha, sigma_xs, sigma_as):
        """ 
        data = NxD NumPy data matrix (should be centered)

        alpha = Fixed IBP hyperparam for OR (init,a,b) tuple where
        (a,b) are Gamma hyperprior shape and rate/inverse scale
        """
        self.N = data[0].shape[0]
        self.feats = [Feature(d, sigma_x, sigma_a) for d, sigma_x, sigma_a in zip(data, sigma_xs, sigma_as)]
        # IBP hyperparameter
        if (type(alpha) == tuple):
            (self.alpha, self.alpha_a, self.alpha_b) = alpha
        else:
            (self.alpha, self.alpha_a, self.alpha_b) = (alpha, None, None)

        self.initZ()
        self.Zs = []

    def initZ(self):
        """ Init latent features Z according to IBP(alpha) """
        Z = np.ones((0, 0))
        for i in range(1, self.N + 1):
            # Sample existing features
            zi = (np.random.uniform(0, 1, (1, Z.shape[1])) <
                  (Z.sum(axis=0).astype(np.float) / i))
            # Sample new features
            knew = poisson.rvs(self.alpha / i)
            zi = np.hstack((zi, np.ones((1, knew))))
            # Add to Z matrix
            Z = np.hstack((Z, np.zeros((Z.shape[0], knew))))
            Z = np.vstack((Z, zi))
        self.Z = Z
        self.K = self.Z.shape[1]
        # Calculate initial feature counts
        self.m = self.Z.sum(axis=0)

    #
    # Convenient external methods
    #

    def fullSample(self):
        """ Do all applicable samples """
        self.sampleZ()
        if self.alpha_a is not None:
            self.sampleAlpha()
        for feat in self.feats:
            if feat.sigma_xa is not None:
                feat.sampleSigma(self.Z)

    def logLike(self, flag=False):
        return self.logIBP() + sum(f.logPX(self.Z) for f in self.feats)

    def sampleAlpha(self):
        """ Sample alpha from conjugate posterior """
        postShape = self.alpha_a + self.m.sum()
        postScale = 1 / (self.alpha_b + self.N)
        self.alpha = gamma.rvs(postShape, scale=postScale)

    def sampleZ(self):
        """ Take single sample of latent features Z """
        # for each data point
        order = np.random.permutation(self.N)
        for (ctr, i) in enumerate(order):
            # Initially, and later occasionally,
            # re-cacluate information directly
            if ctr % 5 == 0:
                for feat in self.feats:
                    feat.meanA, feat.covarA = feat.postA(self.Z)
                    feat.infoA = np.linalg.inv(feat.covarA)
                    feat.hA = feat.infoA @ feat.meanA
            # Get (z,x) for this data point
            zi = self.Z[i]

            # Remove this point from information
            for feat in self.feats:
                feat.update(self.Z[i], feat.X[i], sub=True)
                feat.covarA = np.linalg.inv(feat.infoA)
                feat.meanA = np.dot(feat.covarA, feat.hA)

            # Remove this data point from feature cts
            mi = self.m - self.Z[i]

            lpz = np.log(np.stack((self.N - mi, mi)))
            # Find all singleton features
            singletons = np.nonzero(mi <= 0)[0]
            # Sample for each non-singleton feature
            #
            prev = np.copy(zi)
            for k in np.nonzero(mi > 0)[0]:
                zi[k] = 0
                lp0 = lpz[0, k] + sum(f.logPxi(zi, i) for f in self.feats)
                zi[k] = 1
                lp1 = lpz[1, k] + sum(f.logPxi(zi, i) for f in self.feats)
                zi[k] = int(np.random.uniform(0, 1) < expit(lp1 - lp0))
            self.m += zi - prev

            # Metropolis-Hastings step described in Meeds et al
            netk = poisson.rvs(self.alpha / self.N) - len(singletons)

            # Calculate the loglikelihoods
            lpold = sum(f.logPxi(zi, i) for f in self.feats)
            lpnew = sum(f.logPxi(zi, i, noise=netk * f.var_a) for f in self.feats)
            lpaccept = min(0.0, lpnew - lpold)
            lpreject = np.log(max(1.0 - np.exp(lpaccept), 1e-100))

            if np.random.uniform(0, 1) < expit(lpaccept - lpreject):
                if netk > 0:
                    self.Z = np.hstack((self.Z, np.zeros((self.N, netk))))
                    self.Z[i, self.K:] = 1
                    self.m = np.hstack((self.m, np.ones(netk)))

                    for feat in self.feats:
                        zero = np.zeros((self.K, netk))
                        feat.infoA = np.block([[feat.infoA, zero], [zero.T, np.eye(netk) / feat.var_a]])
                        feat.hA = np.vstack((feat.hA, np.zeros((netk, feat.D))))
                elif netk < 0:
                    dead = singletons[:-netk]
                    self.Z = np.delete(self.Z, dead, axis=1)
                    self.m = np.delete(self.m, dead)
                    for feat in self.feats:
                        feat.infoA = np.delete(feat.infoA, dead, axis=0)
                        feat.infoA = np.delete(feat.infoA, dead, axis=1)
                        feat.hA = np.delete(feat.hA, dead, axis=0)
                self.K += netk

            for feat in self.feats:
                feat.update(self.Z[i], feat.X[i])

    def logIBP(self):
        logp = self.K * np.log(self.alpha)
        logp -= lnfact(np.unique(self.Z, axis=1, return_counts=True)[1]).sum()
        logp -= (self.alpha / np.arange(1, self.N + 1)).sum()
        facts = lnfact(self.N - self.m) + self.lnfact(self.m - 1) - self.lnfact(self.N)
        return logp + facts.sum()

    def predict(self, Xs):
        best = (0, None, None)
        for Z in self.Zs:
            m = Z.sum(axis=0) - Z
            lps = np.sum(np.log(np.where(Z == 1, m, self.N - m) / self.N), axis=1)
            for f, X in zip(self.feats[:-1], Xs[:-1]):
                X_bar = Z @ f.weights(Z)
                diffs = np.sum((X - X_bar) ** 2, axis=1) / (2 * f.var_x)
                lps -= diffs + np.log(2 * np.pi) / 2 + np.log(f.var_x) * f.D / 2
            i = np.argmax(lps)
            if lps[i] > best[0]:
                best = lps[i], Z, i
        best_lp, best_Z, best_i = best
        return self.feats[-1].weights(best_Z) @ best_Z[best_i]

    def run_sampler(self, iters=5000, burn_in=3000, thin=10):
        self.Zs = []
        for i in tqdm_notebook(range(iters)):
            self.fullSample()
            print(self.K)
            if i > burn_in and i % thin == 0:
                self.Zs.append(np.copy(self.Z))
        return self.Zs
