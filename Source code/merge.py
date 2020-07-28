#!/usr/bin/env python3

import fnmatch
import os
import sys
from pandas import read_excel, DataFrame
from scipy.io import arff
from typing import Iterable, Tuple


def new_path(path: str, ext='') -> str:
    'Return the original path, but with a different suffix name.'
    path = list(os.path.splitext(path))
    path[1] = ext
    return ''.join(path)


def read_arff(path: str) -> DataFrame:
    return DataFrame(arff.loadarff(path)[0])


def feature_select(merged: DataFrame) -> Iterable[Tuple[str, DataFrame]]:
    groups = 'Clo dwReach'
    for group in groups.split():
        merged[group] = sum(
            merged[k] for k in merged.columns
            if k.endswith(group)
        )

    target_metrics = {
        'CK': 'loc2 wmc dit noc cbo rfc lcom',
        # ' ca ce npm lcom3 dam moa mfa cam ic cbm amc max_cc avg_cc'
        'NET': '*Ego Eff* Constraint Hierarchy Deg* Eigen* Between* ' + groups,
        'PROC': 'revision_num author_num lines* codechurn_*',
    }
    target_metrics['ALL'] = ' '.join(target_metrics.values())
    for name, metrics in target_metrics.items():
        target = DataFrame()
        metrics += ' bug loc'
        for metric_pattern in metrics.split():
            metric_group = fnmatch.filter(merged.columns, metric_pattern)
            if not metric_group:
                break
            for metric in metric_group:
                target[metric] = merged[metric]
        else:
            yield name, target


def main(arff_path: str):
    arff = read_arff(arff_path)
    arff['loc2'] = arff['loc']
    arff['bug'] = arff.pop('isBug').map(lambda i: int(i == b'YES'))
    for dataset in [arff]:
        for name, df in feature_select(dataset):
            csv_path = '%s-%s.csv' % (arff_path.partition('--')[0], name)
            df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        main(path)
