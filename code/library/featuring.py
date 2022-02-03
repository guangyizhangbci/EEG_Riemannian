import numpy as np
from scipy.linalg import logm
from sklearn.base import TransformerMixin
from pyriemann.tangentspace import TangentSpace


class Riemann(TransformerMixin):
    def __init__(self, n_fb=9, metric='riemann'):
        self.n_fb = n_fb
        self.ts = [TangentSpace(metric=metric) for fb in range(n_fb)] # Tangent Space Learning

    def fit(self, X, y):
        for fb in range(self.n_fb):
            self.ts[fb].fit(X[:, fb, :, :])
        return self

    def transform(self, X):
        n_sub, n_fb, p, _ = X.shape
        Xout = np.empty((n_sub, n_fb, p*(p+1)//2))
        for fb in range(n_fb):
            Xout[:, fb, :] = self.ts[fb].transform(X[:, fb, :, :])
        return Xout.reshape(n_sub, -1)  # (sub, fb * c*(c+1)/2)


class Diag(TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y):
        return self

    def transform(self, X):
        n_sub, n_fb, n_compo, _ = X.shape
        Xout = np.empty((n_sub, n_fb, n_compo))
        for sub in range(n_sub):
            for fb in range(n_fb):
                Xout[sub, fb] = np.diag(X[sub, fb])
        return Xout.reshape(n_sub, -1)  # (sub, fb * n_compo)


class LogDiag(TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y):
        return self

    def transform(self, X):
        n_sub, n_fb, n_compo, _ = X.shape
        Xout = np.empty((n_sub, n_fb, n_compo))
        for sub in range(n_sub):
            for fb in range(n_fb):
                Xout[sub, fb] = np.log10(np.diag(X[sub, fb]))
        return Xout.reshape(n_sub, -1)  # (sub, fb * n_compo)


class NaiveVec(TransformerMixin):
    def __init__(self, method):
        self.method = method
        return None

    def fit(self, X, y):
        return self

    def transform(self, X):
        n_sub, n_fb, n_compo, _ = X.shape
        q = int(n_compo * (n_compo+1) / 2)
        Xout = np.empty((n_sub, n_fb, q))
        for sub in range(n_sub):
            for fb in range(n_fb):
                if self.method == 'upper':
                    Xout[sub, fb] = X[sub, fb][np.triu_indices(n_compo)]
                elif self.method == 'upperlog':
                    logmat = logm(X[sub, fb])
                    Xout[sub, fb] = logmat[np.triu_indices(n_compo)]
                elif self.method == 'logdiag+upper':
                    logdiag = np.log10(np.diag(X[sub, fb]))
                    upper = X[sub, fb][np.triu_indices(n_compo, k=1)]
                    Xout[sub, fb] = np.concatenate((logdiag, upper), axis=None)
        return Xout.reshape(n_sub, -1)  # (sub, fb * c*(c+1)/2)



def to_quotient(C, rank):
    d, U = np.linalg.eigh(C)
    U = U[:, -rank:]
    d = d[-rank:]
    Y = U * np.sqrt(d)
    return Y


def distance2(S1, S2, rank=None):
    Sq = sqrtm(S1, rank)
    P = sqrtm(np.dot(Sq, np.dot(S2, Sq)), rank)
    return np.trace(S1) + np.trace(S2) - 2 * np.trace(P)


def mean_covs(covmats, rank, tol=10e-4, maxiter=50, init=None,
              sample_weight=None):
    Nt, Ne, Ne = covmats.shape
    if sample_weight is None:
        sample_weight = np.ones(Nt)
    if init is None:
        C = np.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    K = sqrtm(C, rank)
    crit = np.finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter):
        k = k + 1
        J = np.zeros((Ne, Ne))
        for index, Ci in enumerate(covmats):
            tmp = np.dot(np.dot(K, Ci), K)
            J += sample_weight[index] * sqrtm(tmp)
        Knew = sqrtm(J, rank)
        crit = np.linalg.norm(Knew - K, ord='fro')
        K = Knew
    if k == maxiter:
        print('Max iter reach')
    C = np.dot(K, K)
    return C


def sqrtm(C, rank=None):
    if rank is None:
        rank = C.shape[0]
    d, U = np.linalg.eigh(C)
    U = U[:, -rank:]
    d = d[-rank:]
    return np.dot(U, np.sqrt(np.abs(d))[:, None] * U.T)


def logarithm_(Y, Y_ref):
    prod = np.dot(Y_ref.T, Y)
    U, D, V = np.linalg.svd(prod, full_matrices=False)
    Q = np.dot(U, V).T
    return np.dot(Y, Q) - Y_ref
