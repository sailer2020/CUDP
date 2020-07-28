#!/usr/bin/env python3

import time
from math import ceil
from pandas import read_csv, DataFrame
from rpy2.robjects.numpy2ri import activate
from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import scale
from sys import argv
from tempfile import NamedTemporaryFile

from merge import new_path
from clustering import classifiers
from modeling import modeling, split_dataset
from weka_wrapper import JVM, WekaDataset

with NamedTemporaryFile(suffix='.csv', delete=False) as f:
    temp = f.name


def gamma(df) -> float:
    sigma = 1 / pairwise_distances(df).mean()
    return 1 / (2 * sigma * sigma)


def feature_selection(model, x, y):
    model.x_label, x_loc = x.pop('bug'), x.pop('loc')
    model.y_label, y_loc = y.pop('bug'), y.pop('loc')
    x = DataFrame(scale(x), columns=x.columns)
    y = DataFrame(scale(y), columns=y.columns)
    print(model)
    x = DataFrame(model.fit_transform(x)).join(model.x_label).join(x_loc)
    y = DataFrame(model.    transform(y)).join(model.y_label).join(y_loc)
    return model.__class__.__name__, x, y


def dispatch(path: str, times=100):
    df = read_csv(path)
    n_components = ceil(0.15 * len(df.columns))
    for i in range(times):
        x, y = split_dataset(df)
        for direction, pair in dict(L=(x, y), R=(y, x)).items():
            yield (i, direction, 'None', *pair)


def collect(path: str):
    for *variant, train, test in dispatch(path):
        train.to_csv(temp, index=None)
        train = WekaDataset(temp)
        test .to_csv(temp, index=None)
        test  = WekaDataset(temp)
        source = new_path(path, '-{:03d}{}-{}'.format(*variant))
        print(time.ctime(), source)
        for c in classifiers():
            if c.__name__ == 'subspace.CLIQUE' and '-ALL.csv' in path:
                continue
            print(time.ctime(), c.__name__)
            modeling(train, test, source, c)


if __name__ == '__main__':
    activate()
    with JVM(system_cp=True, packages=True) as weka:
        for path in argv[1:]:
            collect(path)
