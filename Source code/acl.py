import numpy as np
from itertools import chain, count, starmap
from scipy import stats
from scipy.spatial import distance
from typing import Callable, Iterable


def harmonic_mean(x, y, beta=1):
    beta *= beta
    return (beta + 1) * x * y / np.array(beta * x + y)


def remove_diagonal(matrix: np.ndarray) -> np.ndarray:
    'Remove the diagonal elements from an square matrix (left-aligned).'
    return np.array([
        [element for i, element in enumerate(row) if i != j]
        for j, row in enumerate(matrix)
    ])


def correlation_kernel(reducer: Callable, mapper: Callable):
    'Calculate different types of correlation coefficients.'
    return reducer([
        mapper(lambda *args: getattr(stats, method)(*args)[0])
        for method in ['pearsonr', 'spearmanr', 'kendalltau']
    ])


def correlation_average(x, y):
    'Calculate the correlation coefficient between two metrics.'
    return correlation_kernel(np.mean, lambda f: f(x, y))


def correlation_cutoff(metrics) -> Iterable[float]:
    'Calculate the correlation cutoff for each metric in the original dataset.'
    pairwise_correlations = np.array(correlation_kernel(
        tuple, lambda f:
        remove_diagonal(distance.squareform(distance.pdist(metrics, f)))
    ))
    average_sums = np.array([
        getattr(np, method)(pairwise_correlations, axis=2).sum(axis=0)
        for method in ['mean', 'median']
    ])
    corrH = np.fromiter(starmap(harmonic_mean, average_sums.T), float)
    for i in corrH:
        yield i / (1.5 + i)


def partitions(metrics) -> Iterable[np.ndarray]:
    'Split metrics into subsets by their pairwise correlation coefficient.'
    for i, m, cutoff in zip(count(), metrics, correlation_cutoff(metrics)):
        yield np.array([
            metric for j, metric in enumerate(metrics)
            if i == j or correlation_average(m, metric) < cutoff
        ]).T


def acl_kernel(X: np.ndarray) -> np.ndarray:
    'Average clustering and labeling'
    # Algorithm 1
    HAM = np.mean(X, axis=0) / 2  # half average of metrics
    MVM = X > HAM  # metrics violation matrix
    MIVS = np.sum(MVM, axis=1)  # metrics of instances violation scores

    # Algorithm 2: Labeling Cutoff
    instances, metrics = X.shape  # rows and columns in the metric set
    PD = np.count_nonzero(MIVS > metrics / 2)  # possible defect
    PDr = PD / instances  # possible rate of defect

    AMIVS = np.mean(MIVS)
    AMIVSp = np.mean(np.unique(MIVS))
    MMIVS = np.median(MIVS)

    HMIVS = harmonic_mean(AMIVSp, MMIVS)  # Eq. 3
    cutoff = (
        HMIVS * PDr + (metrics - AMIVS) * (1 - PDr) if PDr >= 0.5  # Eq. 1
        else harmonic_mean(metrics * (1 - PDr), HMIVS)  # Eq.2
    )
    return MIVS, cutoff


def acl(train, test):
    'The wrapper for ACL algorithm'
    MIVS, cutoff = acl_kernel(test.Xu)
    return MIVS > cutoff


def cel(train, test):
    'Cluster ensemble and labeling'
    scoreH, scoreL = np.zeros((2, len(test.Xu)))
    for i, partition in enumerate(chain(partitions(test.Xu.T), [test.Xu])):
        weight = len(partition.T) / len(test.Xu.T)
        MIVS, cutoff = acl_kernel(partition)
        for j, MIVSj in enumerate(MIVS):
            scoreX = scoreH if MIVSj > cutoff else scoreL
            scoreX[j] += weight * abs(MIVSj - cutoff)
    return scoreH > scoreL


acl.__name__ = 'AverageClustering'
cel.__name__ = 'ClusterEnsemble'
acl.labeled = cel.labeled = True
