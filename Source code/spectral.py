import numpy as np


def spec(train, test):
    # Construct the weighted adjacency matrix W.
    W = test.Xu @ test.Xu.T
    # Set all negative values to zero.
    W[W < 0] = 0
    # Set the self - similarity to zero.
    W -= np.diag(np.diag(W))

    # Construct the symmetric Laplacian matrix Lsym.
    Dnsqrt = np.diag(1 / np.sqrt(W.sum(1)))
    I = np.identity(len(W))
    Lsym = I - Dnsqrt @ W @ Dnsqrt
    del W, I

    # Perform the eigen decomposition.
    val, vec = np.linalg.eigh(Lsym)
    # Pick up the second smallest eigenvector.
    v1 = Dnsqrt @ vec[-2]
    # v1 /= np.sqrt((v1 ** 2).sum())
    return v1 > 0


spec.__name__ = 'SpectralClustering'
