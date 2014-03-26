# Compiled Trees

[![Build Status](https://travis-ci.org/ajtulloch/sklearn-compiledtrees.png?branch=master)](https://travis-ci.org/ajtulloch/sklearn-compiledtrees)

## Installation

```bash
pip install sklearn-compiledtrees
```

## Rationale

In some use cases, predicting given a model is in the hot-path, so speeding up decision tree evaluation is very useful.

An effective way of speeding up evaluation of decision trees can be to generate code representing the evaluation of the tree, compile that to optimized object code, and dynamically load that file via dlopen/dlsym or equivalent.

See <https://courses.cs.washington.edu/courses/cse501/10au/compile-machlearn.pdf> for a detailed discussion, and <http://tullo.ch/articles/decision-tree-evaluation/> for a more pedagogical explanation and more benchmarks in C++.


This package implements compiled decision tree evaluation for the
simple case of a single-output regression tree or ensemble.

It has been tested to work on both OS X and Linux.  We do not
currently support Windows platforms for compiled evaluation, although
this should not be a signficant amount of work.


## Usage

```python
import compiledtrees
import sklearn.ensemble

X_train, y_train, X_test, y_test = ...

clf = ensemble.GradientBoostingRegressor()
clf.fit(X_train, y_train)

compiled_predictor = compiledtrees.CompiledRegressionPredictor(clf)
predictions = compiled_predictor.predict(X_test)
```

## Benchmarks

For random forests, we see 5x to 8x speedup in evaluation. For
gradient boosted ensembles, it's between a 1.5x and 3x speedup in
evaluation. This is due to the fact that gradient boosted trees
already have an optimized prediction implementation.

There is a benchmark script attached that allows us to examine the
performance of evaluation across a range of ensemble configurations
and datasets.

In the graphs attached, `GB` is Gradient Boosted, `RF` is Random
Forest, `D1`, etc correspond to setting `max-depth=1`, and `B10` corresponds to setting `max_leaf_nodes=10`.

## Graphs

```bash
for dataset in friedman1 friedman2 friedman3 uniform hastie; do 
    python ../benchmarks/bench_compiled_tree.py \
        --iterations=10 \
        --num_examples=1000 \
        --num_features=50 \
        --dataset=$dataset \
        --max_estimators=300 \
        --num_estimator_values=6
done
```

![timings3907426606273805268](https://f.cloud.github.com/assets/1121581/2453407/c70a64bc-aedd-11e3-94c7-519411ae6276.png)
![timings-1162001441413946416](https://f.cloud.github.com/assets/1121581/2453409/c70ad4ec-aedd-11e3-972d-07a49a6bc610.png)
![timings5617004024503483042](https://f.cloud.github.com/assets/1121581/2453410/c70b48dc-aedd-11e3-9c68-ec3f9d4672b8.png)
![timings2681645894201472305](https://f.cloud.github.com/assets/1121581/2453411/c70b4de6-aedd-11e3-86bd-d534b0ad0618.png)
![timings2070620222460516071](https://f.cloud.github.com/assets/1121581/2453408/c70aa594-aedd-11e3-8b14-1a26eb1f3eba.png)
