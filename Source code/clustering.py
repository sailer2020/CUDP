#!/usr/bin/env python3

import numpy as np
import scipy
from itertools import starmap
from sklearn.cluster import *
from sklearn.cluster.k_medoids_ import KMedoids
from skcmeans.algorithms import Hard

from rpy2.robjects import r
from rpy2.robjects.packages import importr

from acl import acl, cel
from modeling import flatten, Dataset
from modeling import sklearn_clusterer, pyclustering_clusterer
from spectral import spec
from weka_wrapper import weka_classifier, weka_clusterer, clami


def classifiers():
    yield from map(weka_classifier, [
        'bayes.NaiveBayes',
        'functions.Logistic',
        'lazy.IBk',
        'trees.J48',
        'rules.JRip',
        'trees.RandomForest',
    ])
    yield from [acl, cel, spec, clami(cla=True), clami()]
    yield skcmeans(Hard, 2)
    yield from starmap(sklearn_clusterer, [
        [AffinityPropagation],
        [AgglomerativeClustering, 2],
        [Birch, 0.5, 50, 2],
        [KMeans, 2],
        [KMedoids, 2],
        [MeanShift],
        [MiniBatchKMeans, 2],
    ])
    yield from starmap(pyclustering_clusterer, [
        ['bsas.bsas', 2, 1.0],
        ['cure.cure', 2],
        ['dbscan.dbscan', 0.5, 3],
        ['mbsas.mbsas', 2, 1.0],
        ['optics.optics', 0.5, 3],
        ['rock.rock', 1.0, 2],
        ['somsc.somsc', 2],
        ['syncsom.syncsom', 4, 4, 1.0],
    ])
    yield from starmap(weka_clusterer, [
        ['Canopy', '-N 2'],
        ['CascadeSimpleKMeans', '-min 2 -max 2'],
        ['Cobweb', '-A 0.5'],
        ['EM', '-N 2'],
        ['FarthestFirst', '-N 2'],
        ['LVQ', '-C 2'],
        ['SelfOrganizingMap'],
        ['XMeans', '-L 2 -H 2'],
    ])
    for i in ['cmeans', 'cshell']:
        yield r_clusterer('e1071', i, centers=2)
    for i in ['hardcl', 'neuralgas']:
        yield r_clusterer('cclust', 'cclust', method=i, centers=2)
    for i in ['pam', 'clara']:
        yield r_clusterer('cluster', i, 'clustering', k=2)
    yield r_clusterer('cluster', 'diana', None)
    yield r_clusterer('klaR', 'kmodes', modes=2)
    yield r_clusterer('subspace', 'CLIQUE')
    yield r_clusterer('factoextra', 'hkmeans', k=2)
    yield r_clusterer('FactoMineR', 'HCPC', 'data.clust$clust', 2, graph=False)


def skcmeans(Classifier, *args, **kwargs):
    def c(train: Dataset, test: Dataset):
        cls = Classifier(*args, **kwargs)
        cls.fit(test.Xu)
        return [pair[0] for pair in cls.memberships]
    c.__name__ = 'CMeans' + Classifier.__name__
    return c


def r_clusterer(package: str, function: str, index='cluster', *args, **kwargs):
    def normal(train: Dataset, test: Dataset):
        r = getattr(importr(package), function)(test.Xu, *args, **kwargs)
        i = index if isinstance(index, int) else list(r.names).index(index)
        return np.array(r[i]).astype(int) - 1

    def hierarchical(train: Dataset, test: Dataset):
        s = importr('stats')
        i = s.dist(test.Xu) if index is False else test.Xu  # hclust
        r = getattr(importr(package), function)(i, *args, **kwargs)
        return np.array(s.cutree(r, k=2)) - 1

    def hcpc(train: Dataset, test: Dataset):
        df = r('data.frame')(test.Xu)
        x = getattr(importr(package), function)(df, *args, **kwargs)
        for i in index.split('$'):
            x = x[list(x.names).index(i)]
        return np.array(x).astype(int) - 1

    def subspace(train: Dataset, test: Dataset):
        r = getattr(importr(package), function)(test.Xu, *args, **kwargs)
        clusters = [np.array(i[1]).astype(int) - 1 for i in r]
        centers = [np.mean(test.Xu[i], axis=0) for i in clusters]
        Yp = np.full_like(test.Yu, len(clusters))
        min = np.full((len(test.Yu), ), np.inf)
        for i, (cl, ce) in enumerate(zip(clusters, centers)):
            for j in cl:
                dist = scipy.spatial.distance.euclidean(test.Xu[j], ce)
                if min[j] > dist:
                    min[j] = dist
                    Yp[j] = i
        return Yp

    params = ''.join(repr(i) for i in [args, kwargs] if i)
    params = params.replace("'", '').replace(',', ';')
    if package == 'subspace':
        c = subspace
    elif function == 'HCPC':
        c = hcpc
    elif index:
        c = normal
    else:
        c = hierarchical
    c.__name__ = '{package}.{function}{params}'.format(**vars())
    return c
