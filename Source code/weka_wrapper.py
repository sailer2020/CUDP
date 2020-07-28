import shlex
import weka.core
import weka.classifiers
import weka.clusterers
import weka.attribute_selection
from contextlib import contextmanager
from pandas import read_csv
from tempfile import NamedTemporaryFile
from typing import Callable

from modeling import Dataset


class WekaDataset(Dataset):

    def __init__(self, path: str):
        super().__init__(read_csv(path))
        weka_XYLu = weka.core.converters.load_any_file(path)
        weka_XYu = weka_filter('Remove', '-R last')(weka_XYLu)
        weka_XYu = weka_filter('NumericToNominal', '-R last')(weka_XYu)
        weka_XYu.class_is_last()
        weka_Xu = weka_filter('Remove', '-R last')(weka_XYu)
        self.update(vars())


class WekaAttrSel(weka.attribute_selection.AttributeSelection):

    def __init__(self, evaluator, search):
        super().__init__()
        search_cls, *search_options = shlex.split(search)
        self.search(weka.attribute_selection.ASSearch(
            classname='weka.attributeSelection.' + search_cls,
            options=search_options,
        ))
        evaluator_cls, *evaluator_options = shlex.split(evaluator)
        self.evaluator(weka.attribute_selection.ASEvaluation(
            classname='weka.attributeSelection.' + evaluator_cls,
            options=evaluator_options,
        ))
        self.__class__.__name__ = evaluator_cls

    def fit(self, train):
        with NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
            train.join(self.x_label.astype(bool)).to_csv(f, index=None)
        self.select_attributes(weka.core.converters.load_any_file(f.name))
        print(self.results_string)

    def transform(self, test):
        return test.iloc[:, self.selected_attributes[:-1]]

    def fit_transform(self, train):
        self.fit(train)
        return self.transform(train)


def weka_filter(module_name: str, options: str='') -> Callable:
    def c(X):
        name = 'weka.filters.unsupervised.attribute.' + module_name
        cls = weka.filters.Filter(name, options=shlex.split(options))
        cls.inputformat(X)
        return cls.filter(X)
    c.__name__ = module_name
    return c


def weka_classifier(module_name: str, options: str='') -> Callable:
    name = 'weka.classifiers.' + module_name
    def c(train: WekaDataset, test: WekaDataset):
        cls = weka.classifiers.Classifier(name, options=shlex.split(options))
        cls.build_classifier(train.weka_XYu)
        # distribution_for_instance(i)
        return [cls.classify_instance(i) for i in test.weka_XYu]
    c.__name__ = name
    c.labeled = True
    return c


def weka_clusterer(module_name: str, options: str='') -> Callable:
    name = 'weka.clusterers.' + module_name
    def c(train: WekaDataset, test: WekaDataset):
        cls = weka.clusterers.Clusterer(name, options=shlex.split(options))
        cls.build_clusterer(test.weka_Xu)
        return [cls.cluster_instance(i) for i in test.weka_Xu]
    c.__name__ = name
    return c


def clami(cla=False) -> Callable:
    Utils = weka.core.classes.JClassWrapper('net.lifove.clami.util.Utils')
    def c(train: WekaDataset, test: WekaDataset):
        return weka.core.classes.javabridge.get_env().get_int_array_elements(
            Utils.getCLAMI(test.weka_XYu.jobject, cla).o
        )
    c.__name__ = 'CLA' if cla else 'CLAMI'
    c.labeled = True
    return c


@contextmanager
def JVM(*args, **kwargs) -> None:
    try:
        weka.core.jvm.start(*args, **kwargs)
        yield weka
    finally:
        weka.core.jvm.stop()
