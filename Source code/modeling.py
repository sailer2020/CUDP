import numpy as np
import os
import time
from collections import OrderedDict
from importlib import import_module
from multiprocessing import RLock
from pandas import concat, DataFrame, Series
from sklearn.metrics import auc, roc_auc_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from typing import Callable, Dict, Iterable, Tuple
from warnings import filterwarnings

from acl import harmonic_mean

file_lock = RLock()
filterwarnings('ignore')


class Dataset(OrderedDict):

    __getattr__ = OrderedDict.__getitem__
    __setattr__ = OrderedDict.__setitem__

    def __init__(self, XYLu: DataFrame):
        Xu, Yu, Lu = map(
            np.array,
            (XYLu.iloc[:, part] for part in [slice(-2), -2, -1])
        )
        self.update(vars())


def threshold_index(loc: Iterable[int], percent: float) -> int:
    threshold = sum(loc) * percent
    for i, x in enumerate(loc):
        threshold -= x
        if threshold < 0:
            return i + 1


def modeling(train: Dataset, test: Dataset, source: str, c: Callable) -> None:
    try:
        elapsed_time = -time.monotonic()
        Yp = np.array(c(train, test))
        elapsed_time += time.monotonic()
    except Exception as e:
        print(c.__name__)
        print(e.__class__.__name__, e)
        return save(dict(source=source, algorithm=c.__name__))
    proba = None
    if len(Yp.shape) > 1:
        Yp, proba = Yp
    already_labeled = getattr(c, 'labeled', False)
    test.XYLu['pred'] = (
        Yp if already_labeled
        else Series(Yp).map(binary_labeling(Yp, test.Xu))
    )
    metrics = evaluate(test.Yu, Yp, source, c.__name__)
    metrics.update(AUC=0.5 if proba is None else roc_auc_score(test.Yu, proba))
    metrics.update(elapsed_time=elapsed_time)
    XYLuYp_sorted = test.XYLu.sort_values('loc')
    metrics.update(effort_aware(XYLuYp_sorted))
    save(metrics)


def split_dataset(df: DataFrame, values=[True, False], key='bug') -> tuple:
    'Partition the dataset into two symmetric divides.'
    positives, negatives = (df[df[key] == v] for v in values)
    (p_train, p_test), (n_train, n_test) = map(
        lambda dataset: train_test_split(dataset, test_size=0.5),
        (positives, negatives),
    )
    return (p_train.append(n_train), p_test.append(n_test))


def binary_labeling(Yp, Xu) -> Dict[int, bool]:
    'Label the instances after unsupervised clustering.'
    clusters = np.unique(Yp)
    K = (scale(Xu) > 0).sum(axis=1)
    Kc = np.array([
        K[Yp == c].mean() for c in clusters
    ])
    binarized = Kc > Kc.mean()
    return dict(zip(clusters, binarized))


def sklearn_classifier(module, prefix='sklearn.', *args, **kwargs) -> Callable:
    package, _, function = module.rpartition('.')
    cls = getattr(import_module(prefix + package), function)

    def c(train: Dataset, test: Dataset):
        model = cls(*args, **kwargs).fit(train.Xu, train.Yu)
        return model.predict(test.Xu)

    c.__name__ = cls.__name__
    c.labeled = True
    return c


def sklearn_clusterer(Clusterer, *args, **kwargs) -> Callable:
    c = lambda train, test: Clusterer(*args, **kwargs).fit(test.Xu).labels_
    c.__name__ = Clusterer.__name__
    return c


def pyclustering_clusterer(module: str, *args, **kwargs) -> Callable:
    package, _, function = module.rpartition('.')
    f = getattr(import_module('pyclustering.cluster.' + package), function)

    def c(train: Dataset, test: Dataset):
        model = f(test.Xu, *args, **kwargs)
        model.process()
        return flatten(model.get_clusters(), len(test.Xu))

    c.__name__ = module
    return c


def flatten(clusters: Iterable[Iterable], n_samples: int) -> np.ndarray:
    'Return a vector to represent the clustering results.'
    labels = [None] * n_samples
    label = -1  # in case the first loop is skipped
    for label, cluster in enumerate(clusters):
        for sample in cluster:
            labels[sample] = label
    for sample in range(len(labels)):
        if labels[sample] is None:
            label += 1
            labels[sample] = label
    return np.array(labels)


def evaluate(Yu, Yp, source, algorithm) -> dict:
    'Calculate the performance indicators.'
    TN, FP, FN, TP = np.fromiter((sum(
        bool(j >> 1) == bool(Yu[i]) and
        bool(j & 1) == bool(Yp[i])
        for i in range(len(Yu))
    ) for j in range(4)), float)

    RI = Accuracy = (TN + TP) / (TN + FP + FN + TP)
    Precision = TP / (TP + FP)
    Pd = Recall = TP / (TP + FN)
    Pf = FP / (FP + TN)
    F1 = harmonic_mean(Precision, Recall)
    F_negative = harmonic_mean(TN / (TN + FN), TN / (TN + FP))
    F2 = harmonic_mean(Precision, Recall, 2)
    G1 = 2 * Pd * (1 - Pf) / (Pd + (1 - Pf))
    g_mean = np.sqrt(Recall * TN / (TN + FP))
    Bal = 1 - np.sqrt(Pf**2 + (1 - Pd)**2) / np.sqrt(2)
    MCC = np.array([TP + FN, TP + FP, FN + TN, FP + TN]).prod()
    MCC = (TP * TN - FN * FP) / np.sqrt(MCC)
    NMI = normalized_mutual_info_score(Yu, Yp)
    Yp = vars()
    return {k: Yp[k] for k in reversed(list(Yp)) if k not in ['Yu', 'Yp']}


def save(metrics: dict, path='../results.csv') -> None:
    'Save the results to a comma-separated file.'
    print_header = not os.path.exists(path)
    with file_lock, open(path, 'a') as f:
        if print_header:
            for k in metrics:
                print('"%s"' % k, end=',', file=f)
            print(file=f)

        for v in metrics.values():
            print(v, end=',', file=f)
        print(file=f)


def positive_first(df: DataFrame) -> DataFrame:
    'Move the positive instances to the front of the dataset.'
    if sum(df.pred == df.bug) * 2 < len(df):
        df.pred = (df.pred == False)

    return concat([df[df.pred == True], df[df.pred == False]])


def norm_opt(*args: Tuple[DataFrame]) -> float:
    'Calculate the Alberg-diagram-based effort-aware indicator.'
    predict, optimal, worst = map(alberg_auc, args)
    return 1 - (optimal - predict) / (optimal - worst)


def alberg_auc(df: DataFrame) -> float:
    'Calculate the area under curve in Alberg diagrams.'
    points = df[['loc', 'bug']].values.cumsum(axis=0)
    points = np.insert(points, 0, [0, 0], axis=0) / points[-1]
    return auc(*points.T)


def effort_aware(df: DataFrame) -> Dict[str, float]:
    'Calculate the effort-aware performance indicators.'
    EAPredict = positive_first(df)
    EAOptimal = concat([df[df.bug == True], df[df.bug == False]])
    EAWorst = EAOptimal.iloc[::-1]

    M = len(df)
    N = sum(df.bug)
    m = threshold_index(EAPredict['loc'], 0.2)
    n = sum(EAPredict.bug[:m])
    for k, y in enumerate(EAPredict.bug):
        if y:
            break

    y = set(vars().keys())
    EA_Precision = n / m
    EA_Recall = n / N
    EA_F1 = harmonic_mean(EA_Precision, EA_Recall)
    EA_F2 = 5 * EA_Precision * EA_Recall / np.array(4 * EA_Precision + EA_Recall)
    PCI = m / M
    IFA = k
    P_opt = norm_opt(EAPredict, EAOptimal, EAWorst)
    M = vars()
    return {k: M[k] for k in reversed(list(M)) if k not in y}
