from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import six

from sklearn.base import RegressorMixin
from sklearn.tree.tree import DecisionTreeRegressor, DTYPE, DOUBLE
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble.forest import ForestRegressor

from compiledtrees import _compiled
from compiledtrees import code_gen as cg
import numpy as np

import platform

if platform.system() == 'Windows':
    delete_files = False
else:
    delete_files = True


class CompiledRegressionPredictor(RegressorMixin):
    """Class to construct a compiled predictor from a previously trained
    ensemble of decision trees.

    Parameters
    ----------

    clf:
      A fitted regression tree/ensemble.

    References
    ----------

    http://courses.cs.washington.edu/courses/cse501/10au/compile-machlearn.pdf
    http://crsouza.blogspot.com/2012/01/decision-trees-in-c.html
    """
    def __init__(self, clf, n_jobs=-1):
        self._n_features, self._evaluator, self._so_f_object = self._build(clf, n_jobs)
        self._so_f = self._so_f_object.name

    def __getstate__(self):
        return dict(n_features=self._n_features, so_f=open(self._so_f, 'rb').read())

    def __setstate__(self, state):
        import tempfile
        self._so_f_object = tempfile.NamedTemporaryFile(mode='w+b',
                                                        prefix='compiledtrees_',
                                                        suffix='.so',
                                                        delete=delete_files)
        if isinstance(state["so_f"], six.text_type):
            state["so_f"] = state["so_f"].encode('latin1')
        self._so_f_object.write(state["so_f"])
        self._so_f_object.flush()
        self._so_f = self._so_f_object.name
        if platform.system() == 'Windows':
            self._so_f_object.close()
        self._n_features = state["n_features"]
        self._evaluator = _compiled.CompiledPredictor(
            self._so_f_object.name.encode("ascii"),
            cg.EVALUATE_FN_NAME.encode("ascii"))

    @classmethod
    def _build(cls, clf, n_jobs=1):
        if not cls.compilable(clf):
            raise ValueError("Predictor {} cannot be compiled".format(
                clf.__class__.__name__))

        files = None
        n_features = None
        if isinstance(clf, DecisionTreeRegressor):
            n_features = clf.n_features_
            files = cg.code_gen_tree(tree=clf.tree_)

        if isinstance(clf, GradientBoostingRegressor):
            n_features = clf.n_features

            # hack to get the initial (prior) on the decision tree.
            initial_value = clf._init_decision_function(
                np.zeros(shape=(1, n_features))).item((0, 0))

            files = cg.code_gen_ensemble(
                trees=[e.tree_ for e in clf.estimators_.flat],
                individual_learner_weight=clf.learning_rate,
                initial_value=initial_value, n_jobs=n_jobs)

        if isinstance(clf, ForestRegressor):
            n_features = clf.n_features_
            files = cg.code_gen_ensemble(
                trees=[e.tree_ for e in clf.estimators_],
                individual_learner_weight=1.0 / clf.n_estimators,
                initial_value=0.0, n_jobs=n_jobs)

        assert n_features is not None
        assert files is not None

        so_f = cg.compile_code_to_object(files, n_jobs=n_jobs)
        evaluator = _compiled.CompiledPredictor(
            so_f.name.encode("ascii"),
            cg.EVALUATE_FN_NAME.encode("ascii"))
        return n_features, evaluator, so_f

    @classmethod
    def compilable(cls, clf):
        """
        Verifies that the given fitted model is eligible to be compiled.

        Returns True if the model is eligible, and False otherwise.

        Parameters
        ----------

        clf:
          A fitted regression tree/ensemble.


        """
        # TODO - is there an established way to check `is_fitted``?
        if isinstance(clf, DecisionTreeRegressor):
            return (hasattr(clf, 'n_outputs_') and
                    clf.n_outputs_ == 1 and
                    hasattr(clf, 'n_classes_') and
                    clf.n_classes_ == 1 and
                    clf.tree_ is not None)

        if isinstance(clf, (GradientBoostingRegressor, ForestRegressor)):
            return (hasattr(clf, 'estimators_') and
                    np.asarray(clf.estimators_).size and
                    all(cls.compilable(e)
                        for e in np.asarray(clf.estimators_).flat))
        return False

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y: array of shape = [n_samples]
            The predicted values.
        """
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)
        if X.dtype != DTYPE:
            X = X.astype(DTYPE)
        if X.ndim != 2:
            raise ValueError(
                "Input must be 2-dimensional (n_samples, n_features), "
                "not {}".format(X.shape))

        n_samples, n_features = X.shape
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is {} and "
                             " input n_features is {}".format(
                                 self._n_features, n_features))

        result = np.empty(n_samples, dtype=DOUBLE)
        return self._evaluator.predict(X, result)
