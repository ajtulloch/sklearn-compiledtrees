from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from sklearn import ensemble, tree
from compiledtrees.compiled import CompiledRegressionPredictor
from sklearn.utils.testing import \
    assert_array_almost_equal, assert_raises, assert_equal
import numpy as np
import unittest
import tempfile
import pickle
import cPickle

REGRESSORS = {
    ensemble.GradientBoostingRegressor,
    ensemble.RandomForestRegressor,
    tree.DecisionTreeRegressor,
}

CLASSIFIERS = {
    ensemble.GradientBoostingClassifier,
    ensemble.RandomForestClassifier,
    tree.DecisionTreeClassifier,
}


def pairwise(iterable):
    import itertools
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def assert_equal_predictions(cls, X, y):
    clf = cls()
    clf.fit(X, y)
    compiled = CompiledRegressionPredictor(clf)

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        pickle.dump(compiled, tf)
    depickled = pickle.load(open(tf.name))

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        pickle.dump(depickled, tf)
    dedepickled = pickle.load(open(tf.name))

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        cPickle.dump(compiled, tf)
    decpickled = cPickle.load(open(tf.name))

    predictors = [clf, compiled, depickled, decpickled, dedepickled]
    predictions = [p.predict(X) for p in predictors]
    for (p1, p2) in pairwise(predictions):
        assert_array_almost_equal(p1, p2)


class TestCompiledTrees(unittest.TestCase):
    def test_rejects_unfitted_regressors_as_compilable(self):
        for cls in REGRESSORS:
            assert_equal(CompiledRegressionPredictor.compilable(cls()), False)
            assert_raises(ValueError, CompiledRegressionPredictor, cls())

    def test_rejects_classifiers_as_compilable(self):
        for cls in CLASSIFIERS:
            assert_equal(CompiledRegressionPredictor.compilable(cls()), False)
            assert_raises(ValueError, CompiledRegressionPredictor, cls())

    def test_correct_predictions(self):
        num_features = 20
        num_examples = 1000
        X = np.random.normal(size=(num_examples, num_features))
        X = X.astype(np.float32)
        y = np.random.normal(size=num_examples)
        for cls in REGRESSORS:
            assert_equal_predictions(cls, X, y)
        y = np.random.choice([-1, 1], size=num_examples)
        for cls in REGRESSORS:
            assert_equal_predictions(cls, X, y)

    def test_predictions_with_invalid_input(self):
        num_features = 100
        num_examples = 100
        X = np.random.normal(size=(num_examples, num_features))
        X = X.astype(np.float32)
        y = np.random.choice([-1, 1], size=num_examples)

        for cls in REGRESSORS:
            clf = cls()
            clf.fit(X, y)
            compiled = CompiledRegressionPredictor(clf)
            assert_raises(ValueError, compiled.predict, X.astype(np.float64))
            assert_raises(ValueError, compiled.predict,
                          np.resize(X, (1, num_features, num_features)))
