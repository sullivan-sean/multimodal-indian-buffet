import numpy as np
from tqdm import tqdm_notebook

class IndianBuffetModel:
    def __init__(self, Xs, sigma_x, sigma_w, alpha=20):
        self.sigma_x = sigma_x
        self.sigma_w = sigma_w
        self.Xs = Xs
        self.XTXs = [X.T @ X for X in Xs]

        self.init_z(alpha)
        self.init_w()

    def init_z(self, alpha):
        N = len(self.Xs[0])
        init = np.random.poisson(alpha)
        self.Z = np.random.uniform(size=(N, init)) > 0.5
        self.Z[0] = 1

    def sample_ws(self, Z):
        # see http://mlg.eng.cam.ac.uk/pub/pdf/DosGha09.pdf eq 6
        ZTZ = Z.T @ Z
        I = np.eye(len(ZTZ))
        Ws = []

        for X, sig_x, sig_w in zip(self.Xs, self.sigma_x, self.sigma_w):
            ZTZI = ZTZ + I * (sig_x / sig_w)**2
            ZTZI_inv = np.linalg.inv(ZTZI)
            mean = ZTZI_inv @ Z.T @ X
            stddev = sig_x * np.sqrt(ZTZI_inv)
            Ws.append(np.random.normal(mean, stddev))

        return Ws

    def collapsed_nll(self, Z):
        # see http://mlg.eng.cam.ac.uk/pub/pdf/DosGha09.pdf eq 5
        N, K = Z.shape
        nll = 0
        ZTZ = Z.T @ Z
        I = np.eye(len(ZTZ))

        for X, XTX, sig_x, sig_w in zip(self.Xs, self.XTXs, self.sigma_x,
                                        self.sigma_w):
            sx = sig_x**2
            sig = sx / (sig_w**2)
            ZTZI = ZTZ + I * sig
            XZ = X.T @ Z
            log_denom = N * np.log(
                np.pi * sx) - K * np.log(sig) + np.linalg.norm(ZTZI)
            nll += (XTX - XZ @ np.linalg.inv(ZTZI) @ XZ.T) / (
                2 * sx) + log_denom * X.shape[1] / 2
        return nll

    def uncollapsed_nll(self, Z):
        N, K = Z.shape
        N, D = X.shape
        nll = 0
        for X, W, sig_x in zip(self.Xs, self.Ws, self.sigma_x):
            sx = 2 * sig_x**2
            nll += np.linalg.norm(X - Z @ W)**2 / sx + N * np.log(
                np.pi * sx) * X.shape[1] / 2
        return nll

    def likelihood(self, Z, collapsed=False):
        nll = self.collapsed_nll(Z) if collapsed else self.uncollapsed_nll(Z)
        return np.exp(-nll)

    def run_sampler(self, iters=5000, burn_in=3000, thin=10, **kwargs):
        Zs = [self.Z]
        for _ in tqdm_notebook(range(iters)):
            self.gibbs_sample(**kwargs)
            Zs.append(self.Z)
        return Zs[burn_in + 1::thin]

    def sample_z(self, alpha):
        Z = self.Z
        N = self.N
        for i in range(len(Z)):
            mi = Z.sum(axis=0) - Z[i]
            nonsingle = np.nonzero(mi > 0)[0]
            pz = np.stack([mi / N, 1 - mi / N])
            pz1 = mi / N
            pz0 = 1 - pz1
            if False:
                for j in nonsingle:
                    Z[i, j] = 0
                    pz[1, j] *= self.likelihood(Z)
                    Z[i, j] = 1
                    pz[0, j] *= self.likelihood(Z)
            pz /= pz.sum(axis=0)
            Z[i, mi > 0] = np.random.uniform(
                size=len(nonsingle)) < pz[0, mi > 0]
            Z = np.delete(Z, np.nonzero(mi <= 0)[0], axis=1)

            new_Z = np.zeros((N, int(np.round(np.random.poisson(alpha / N)))))
            new_Z[i] = 1
            Z = np.concatenate((Z, new_Z), axis=1)
        self.Z = Z
