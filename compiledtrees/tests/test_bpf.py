from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from sklearn import tree
import compiledtrees.bpf as bpf

from sklearn.utils.testing import \
    assert_array_almost_equal, assert_raises, assert_equal
import numpy as np
import unittest

import pprint
import textwrap


REGRESSORS = {
    # ensemble.GradientBoostingRegressor,
    # ensemble.RandomForestRegressor,
    # tree.DecisionTreeRegressor,
}

CLASSIFIERS = {
    # ensemble.GradientBoostingClassifier,
    # ensemble.RandomForestClassifier,
    lambda: tree.DecisionTreeClassifier(max_depth=10, random_state=3),
    lambda: tree.DecisionTreeClassifier(max_depth=2, random_state=3),
}


def pairwise(iterable):
    import itertools
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def assert_equal_predictions(cls, X, y):
    clf = cls()
    clf.fit(X, y)
    compiled = bpf.BpfClassifierPredictor(clf)
    predictors = [clf, compiled]
    predictions = [p.predict(X) for p in predictors]
    for (p1, p2) in pairwise(predictions):
        assert_array_almost_equal(p1, p2, decimal=10)


class TestBpfUtils(unittest.TestCase):
    bpf = True

    def setUp(self):
        import collections
        T = collections.namedtuple(
            'T',
            ['children_left', 'children_right',
             'feature', 'threshold', 'value'])
        # Deliberately sparse
        # 0 -> {4, 5}
        self.t = T(
            children_left={0: 4, 4: -1, 5: -1},
            children_right={0: 5},
            feature={0: 10},
            threshold={0: 23.4},
            value={4: np.asarray([89, 880]), 5: np.asarray([605, 105])}
        )

    def test_construct_cfg(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        self.assertEqual(
            cfg.edges(data=True),
            [
                (0, -3, {'data': bpf.Direction.LEFT}),
                (0, -2, {'data': bpf.Direction.RIGHT}),
                (-2, -1, {}),
                (-3, -1, {})
            ])

        self.assertEqual(
            cfg.nodes(data=True),
            [
                (0, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(10, 23.4))}),
                (-2, {'ann': bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))}),
                (-1, {'ann': bpf.Node(ty=bpf.NodeTy.EXIT, args=())}),
                (-3, {'ann': bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,))})
            ])

    def test_construct_node_ret_tys(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        self.assertEquals(
            node_ret_tys,
            {
                -3: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,))]),
                -2: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))]),
                0: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,)),
                        bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))])
            })

    def test_collapse_cfg(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        ccfg = bpf.collapse_cfg(cfg, node_ret_tys)
        self.assertEqual(
            sorted(ccfg.nodes(data=True)), sorted(cfg.nodes(data=True)))
        self.assertEqual(ccfg.edges(data=True), cfg.edges(data=True))

    def test_construct_fragments(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        ccfg = bpf.collapse_cfg(cfg, node_ret_tys)
        fragments = bpf.construct_fragments(ccfg, node_ret_tys)
        self.assertEqual(
            fragments, {
                -3: [bpf.Ins(code=bpf.BPF_RET | bpf.BPF_K, jt=0, jf=0, k=0)],
                -2: [bpf.Ins(code=bpf.BPF_RET | bpf.BPF_K, jt=0, jf=0, k=1)],
                -1: [],
                0: [
                    bpf.Ins(
                        code=bpf.BPF_W | bpf.BPF_LD | bpf.BPF_ABS,
                        jt=0, jf=0, k=10),
                    bpf.Ins(
                        code=bpf.BPF_JMP | bpf.BPF_JGT | bpf.BPF_K,
                        jt=-2, jf=-3, k=23.4)
                ]
            })

    def test_dce(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        ccfg = bpf.collapse_cfg(cfg, node_ret_tys)
        fragments = bpf.construct_fragments(ccfg, node_ret_tys)
        dead_fragments = copy.deepcopy(fragments)
        dead_fragments[50] = [None]
        fragments[50] = []
        dce_fragments = bpf.dce(cfg, dead_fragments)
        self.assertEqual(fragments, dce_fragments)

    def test_linearize(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        ccfg = bpf.collapse_cfg(cfg, node_ret_tys)
        fragments = bpf.construct_fragments(ccfg, node_ret_tys)
        fragments = bpf.dce(cfg, fragments)
        inss, label_offsets = bpf.linearize(ccfg, fragments)
        self.assertEqual(
            inss,
            [
                bpf.Ins(code=32, jt=0, jf=0, k=10),
                bpf.Ins(code=37, jt=-2, jf=-3, k=23.4),
                bpf.Ins(code=6, jt=0, jf=0, k=0),
                bpf.Ins(code=6, jt=0, jf=0, k=1)
            ])
        self.assertEqual(label_offsets, {-3: 2, -2: 3, -1: 4, 0: 0})

    def test_assemble(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        ccfg = bpf.collapse_cfg(cfg, node_ret_tys)
        fragments = bpf.construct_fragments(ccfg, node_ret_tys)
        fragments = bpf.dce(cfg, fragments)
        inss, label_offsets = bpf.linearize(ccfg, fragments)
        inss = bpf.assemble(inss, label_offsets)
        self.assertEqual(
            inss,
            [
                bpf.Ins(code=32, jt=0, jf=0, k=10),
                bpf.Ins(code=37, jt=1, jf=0, k=23.4),
                bpf.Ins(code=6, jt=0, jf=0, k=0),
                bpf.Ins(code=6, jt=0, jf=0, k=1)
            ])

    def test_interpret(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        ccfg = bpf.collapse_cfg(cfg, node_ret_tys)
        fragments = bpf.construct_fragments(ccfg, node_ret_tys)
        fragments = bpf.dce(cfg, fragments)
        inss, label_offsets = bpf.linearize(ccfg, fragments)
        inss = bpf.assemble(inss, label_offsets)
        result = bpf.interpret(inss, {10: 25})
        self.assertEqual(result, 1)
        result = bpf.interpret(inss, {10: 20})
        self.assertEqual(result, 0)


class TestBpfDeepUtils(unittest.TestCase):
    bpf = True

    def setUp(self):
        self.maxDiff = 10000
        np.random.seed(0)
        clf = tree.DecisionTreeClassifier(max_depth=4, random_state=3)
        X = np.random.random_integers(low=-100, high=100, size=(1000, 10))
        y = np.random.random_integers(low=0, high=1, size=1000)
        clf.fit(X, y)
        self.clf = clf
        self.t = clf.tree_

    def test_construct_cfg(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        self.assertEqual(
            cfg.edges(data=True),
            [
                (0, 16, {'data': bpf.Direction.RIGHT}),
                (0, 1, {'data': bpf.Direction.LEFT}),
                (1, 9, {'data': bpf.Direction.RIGHT}),
                (1, 2, {'data': bpf.Direction.LEFT}),
                (2, 3, {'data': bpf.Direction.LEFT}),
                (2, 6, {'data': bpf.Direction.RIGHT}),
                (3, -2, {'data': bpf.Direction.RIGHT}),
                (6, -3, {'data': bpf.Direction.RIGHT}),
                (9, 10, {'data': bpf.Direction.LEFT}),
                (9, 13, {'data': bpf.Direction.RIGHT}),
                (10, -2, {'data': bpf.Direction.RIGHT}),
                (13, -3, {'data': bpf.Direction.LEFT}),
                (13, -2, {'data': bpf.Direction.RIGHT}),
                (16, 17, {'data': bpf.Direction.LEFT}),
                (16, -3, {'data': bpf.Direction.RIGHT}),
                (17, 18, {'data': bpf.Direction.LEFT}),
                (17, -2, {'data': bpf.Direction.RIGHT}),
                (18, -3, {'data': bpf.Direction.RIGHT}),
                (-3, -1, {}),
                (-2, -1, {})
            ])
        # Always right:
        # 0 -> 16 -> -3 == 0

        # Always left:
        # 0 -> 1 -> 2 -> 3 -> -2 == 1

        self.assertEqual(
            cfg.nodes(data=True),
            [
                (0, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(5, 94.0))}),
                (1, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(6, 10.5))}),
                (2, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(8, -75.5))}),
                (3, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(0, -41.5))}),
                (6, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(5, 80.5))}),
                (9, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(0, -48.5))}),
                (10, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(2, 55.5))}),
                (13, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(9, 33.5))}),
                (16, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(2, 45.5))}),
                (17, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(1, 77.0))}),
                (18, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(7, 55.0))}),
                (-1, {'ann': bpf.Node(ty=bpf.NodeTy.EXIT, args=())}),
                (-3, {'ann': bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,))}),
                (-2, {'ann': bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))})
            ])

    def test_construct_node_ret_tys(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        self.assertEquals(
            node_ret_tys,
            {-3: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,))]),
             -2: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))]),
             0: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,)),
                     bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))]),
             1: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,)),
                     bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))]),
             2: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,)),
                     bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))]),
             3: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))]),
             6: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,))]),
             9: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,)),
                     bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))]),
             10: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))]),
             13: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,)),
                      bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))]),
             16: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,)),
                      bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))]),
             17: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,)),
                      bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))]),
             18: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,))])})

    def test_collapse_cfg(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        ccfg = bpf.collapse_cfg(cfg, node_ret_tys)
        self.assertEqual(
            sorted(ccfg.nodes(data=True)),
            [
                (-3, {'ann': bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,))}),
                (-2, {'ann': bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))}),
                (-1, {'ann': bpf.Node(ty=bpf.NodeTy.EXIT, args=())}),
                (0, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(5, 94.0))}),
                (1, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(6, 10.5))}),
                (2, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(8, -75.5))}),
                (9, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(0, -48.5))}),
                (13, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(9, 33.5))}),
                (16, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(2, 45.5))}),
                (17, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(1, 77.0))})
            ])
        self.assertEqual(
            ccfg.edges(data=True),
            [
                (0, 16, {'data': bpf.Direction.RIGHT}),
                (0, 1, {'data': bpf.Direction.LEFT}),
                (1, 9, {'data': bpf.Direction.RIGHT}),
                (1, 2, {'data': bpf.Direction.LEFT}),
                (2, -2, {'data': bpf.Direction.LEFT}),
                (2, -3, {'data': bpf.Direction.RIGHT}),
                (9, 13, {'data': bpf.Direction.RIGHT}),
                (9, -2, {'data': bpf.Direction.LEFT}),
                (13, -3, {'data': bpf.Direction.LEFT}),
                (13, -2, {'data': bpf.Direction.RIGHT}),
                (16, 17, {'data': bpf.Direction.LEFT}),
                (16, -3, {'data': bpf.Direction.RIGHT}),
                (17, -3, {'data': bpf.Direction.LEFT}),
                (17, -2, {'data': bpf.Direction.RIGHT}),
                (-2, -1, {}),
                (-3, -1, {})
            ])

    def test_construct_fragments(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        cfg = bpf.collapse_cfg(cfg, node_ret_tys)
        fragments = bpf.construct_fragments(cfg, node_ret_tys)
        self.assertEqual(
            fragments,
            {
                -3: [bpf.Ins(code=6, jt=0, jf=0, k=0)],
                -2: [bpf.Ins(code=6, jt=0, jf=0, k=1)],
                -1: [],
                0: [bpf.Ins(code=32, jt=0, jf=0, k=5),
                    bpf.Ins(code=37, jt=16, jf=1, k=94.0)],
                1: [bpf.Ins(code=32, jt=0, jf=0, k=6),
                    bpf.Ins(code=37, jt=9, jf=2, k=10.5)],
                2: [bpf.Ins(code=32, jt=0, jf=0, k=8),
                    bpf.Ins(code=37, jt=-3, jf=-2, k=-75.5)],
                9: [bpf.Ins(code=32, jt=0, jf=0, k=0),
                    bpf.Ins(code=37, jt=13, jf=-2, k=-48.5)],
                13: [bpf.Ins(code=32, jt=0, jf=0, k=9),
                     bpf.Ins(code=37, jt=-2, jf=-3, k=33.5)],
                16: [bpf.Ins(code=32, jt=0, jf=0, k=2),
                     bpf.Ins(code=37, jt=-3, jf=17, k=45.5)],
                17: [bpf.Ins(code=32, jt=0, jf=0, k=1),
                     bpf.Ins(code=37, jt=-2, jf=-3, k=77.0)],
            })

    def test_dce(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        cfg = bpf.collapse_cfg(cfg, node_ret_tys)
        fragments = bpf.construct_fragments(cfg, node_ret_tys)
        dead_fragments = copy.deepcopy(fragments)
        dead_fragments[50] = [None]
        fragments[50] = []
        dce_fragments = bpf.dce(cfg, dead_fragments)
        self.assertEqual(fragments, dce_fragments)

    def test_linearize(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        cfg = bpf.collapse_cfg(cfg, node_ret_tys)
        fragments = bpf.construct_fragments(cfg, node_ret_tys)
        fragments = bpf.dce(cfg, fragments)
        inss, label_offsets = bpf.linearize(cfg, fragments)

        self.assertEqual(
            inss,
            [
                bpf.Ins(code=32, jt=0, jf=0, k=5),
                bpf.Ins(code=37, jt=16, jf=1, k=94.0),
                bpf.Ins(code=32, jt=0, jf=0, k=2),
                bpf.Ins(code=37, jt=-3, jf=17, k=45.5),
                bpf.Ins(code=32, jt=0, jf=0, k=1),
                bpf.Ins(code=37, jt=-2, jf=-3, k=77.0),
                bpf.Ins(code=32, jt=0, jf=0, k=6),
                bpf.Ins(code=37, jt=9, jf=2, k=10.5),
                bpf.Ins(code=32, jt=0, jf=0, k=8),
                bpf.Ins(code=37, jt=-3, jf=-2, k=-75.5),
                bpf.Ins(code=32, jt=0, jf=0, k=0),
                bpf.Ins(code=37, jt=13, jf=-2, k=-48.5),
                bpf.Ins(code=32, jt=0, jf=0, k=9),
                bpf.Ins(code=37, jt=-2, jf=-3, k=33.5),
                bpf.Ins(code=6, jt=0, jf=0, k=0),
                bpf.Ins(code=6, jt=0, jf=0, k=1),
            ])

        self.assertEqual(
            label_offsets,
            {-3: 14, -2: 15, -1: 16, 0: 0,
             1: 6, 2: 8, 9: 10, 13: 12, 16: 2, 17: 4})

    def test_assemble(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        cfg = bpf.collapse_cfg(cfg, node_ret_tys)
        fragments = bpf.construct_fragments(cfg, node_ret_tys)
        fragments = bpf.dce(cfg, fragments)
        inss, label_offsets = bpf.linearize(cfg, fragments)
        inss = bpf.assemble(inss, label_offsets)
        self.assertEqual(
            inss,
            [
                bpf.Ins(code=32, jt=0, jf=0, k=5),
                bpf.Ins(code=37, jt=0, jf=4, k=94.0),
                bpf.Ins(code=32, jt=0, jf=0, k=2),
                bpf.Ins(code=37, jt=10, jf=0, k=45.5),
                bpf.Ins(code=32, jt=0, jf=0, k=1),
                bpf.Ins(code=37, jt=9, jf=8, k=77.0),
                bpf.Ins(code=32, jt=0, jf=0, k=6),
                bpf.Ins(code=37, jt=2, jf=0, k=10.5),
                bpf.Ins(code=32, jt=0, jf=0, k=8),
                bpf.Ins(code=37, jt=4, jf=5, k=-75.5),
                bpf.Ins(code=32, jt=0, jf=0, k=0),
                bpf.Ins(code=37, jt=0, jf=3, k=-48.5),
                bpf.Ins(code=32, jt=0, jf=0, k=9),
                bpf.Ins(code=37, jt=1, jf=0, k=33.5),
                bpf.Ins(code=6, jt=0, jf=0, k=0),
                bpf.Ins(code=6, jt=0, jf=0, k=1)
            ])

    def test_interpret(self):
        cfg = bpf.construct_cfg(self.t, 0.5)
        node_ret_tys = bpf.construct_node_ret_tys(cfg)
        cfg = bpf.collapse_cfg(cfg, node_ret_tys)
        fragments = bpf.construct_fragments(cfg, node_ret_tys)
        fragments = bpf.dce(cfg, fragments)
        inss, label_offsets = bpf.linearize(cfg, fragments)
        inss = bpf.assemble(inss, label_offsets)

        X = np.asarray([10000 for i in range(10)]).reshape(1, -1)
        result = bpf.interpret(inss, X[0])
        self.assertEqual(result, 0)
        self.assertEqual(result, 1 - self.clf.predict(X)[0])

        X = np.asarray([-10000 for i in range(10)]).reshape(1, -1)
        result = bpf.interpret(inss, X[0])
        self.assertEqual(result, 1)
        self.assertEqual(result, 1 - self.clf.predict(X)[0])


class TestBpfClassifier(unittest.TestCase):
    bpf = True

    def test_rejects_unfitted_classifiers_as_compilable(self):
        for cls in CLASSIFIERS:
            assert_equal(bpf.BpfClassifierPredictor.compilable(cls()), False)
            assert_raises(ValueError, bpf.BpfClassifierPredictor, cls())

    def test_rejects_regressors_as_compilable(self):
        for cls in REGRESSORS:
            assert_equal(bpf.BpfClassifierPredictor.compilable(cls()), False)
            assert_raises(ValueError, bpf.BpfClassifierPredictor, cls())

    def test_correct_predictions(self):
        num_features = 20
        num_examples = 1000
        X = np.random.random_integers(
            low=-100, high=100, size=(num_examples, num_features))
        y = np.random.choice([0, 1], size=num_examples)
        for cls in CLASSIFIERS:
            assert_equal_predictions(cls, X, y)


class TestBpfEnsembleUtils(unittest.TestCase):
    bpf = True

    def setUp(self):
        self.maxDiff = 10000
        import collections
        T = collections.namedtuple(
            'T',
            ['children_left', 'children_right',
             'feature', 'threshold', 'value'])
        # Deliberately sparse
        # 0 -> {4, 5}
        self.ts = [
            T(
                children_left={0: 4, 4: -1, 5: -1},
                children_right={0: 5},
                feature={0: 10},
                threshold={0: 23.4},
                value={4: np.asarray([89, 880]), 5: np.asarray([605, 105])}
            ),
            T(
                children_left={0: 4, 4: -1, 5: -1},
                children_right={0: 5},
                feature={0: 10},
                threshold={0: 23.4},
                value={4: np.asarray([89, 880]), 5: np.asarray([605, 105])}
            ),
        ]

    def test_construct_cfg(self):
        cfgs = [bpf.construct_cfg(t, 0.5) for t in self.ts]
        for cfg in cfgs:
            self.assertEqual(
                cfg.edges(data=True),
                [
                    (0, -3, {'data': bpf.Direction.LEFT}),
                    (0, -2, {'data': bpf.Direction.RIGHT}),
                    (-2, -1, {}),
                    (-3, -1, {})
                ])

        self.assertEqual(
            cfg.nodes(data=True),
            [
                (0, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(10, 23.4))}),
                (-2, {'ann': bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))}),
                (-1, {'ann': bpf.Node(ty=bpf.NodeTy.EXIT, args=())}),
                (-3, {'ann': bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,))})
            ])

    def test_construct_node_ret_tys(self):
        for t in self.ts:
            cfg = bpf.construct_cfg(t, 0.5)
            node_ret_tys = bpf.construct_node_ret_tys(cfg)
            self.assertEquals(
                node_ret_tys,
                {
                    -3: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,))]),
                    -2: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))]),
                    0: set([bpf.Node(ty=bpf.NodeTy.LEAF, args=(0,)),
                            bpf.Node(ty=bpf.NodeTy.LEAF, args=(1,))])
                })

    def test_collapse_cfg(self):
        for t in self.ts:
            cfg = bpf.construct_cfg(t, 0.5)
            node_ret_tys = bpf.construct_node_ret_tys(cfg)
            ccfg = bpf.collapse_cfg(cfg, node_ret_tys)
            self.assertEqual(
                sorted(ccfg.nodes(data=True)), sorted(cfg.nodes(data=True)))
            self.assertEqual(ccfg.edges(data=True), cfg.edges(data=True))

    def test_construct_fragments(self):
        for t in self.ts:
            cfg = bpf.construct_cfg(t, 0.5)
            node_ret_tys = bpf.construct_node_ret_tys(cfg)
            ccfg = bpf.collapse_cfg(cfg, node_ret_tys)
            fragments = bpf.construct_fragments(ccfg, node_ret_tys)
            self.assertEqual(
                fragments, {
                    -3: [bpf.Ins(code=bpf.BPF_RET | bpf.BPF_K,
                                 jt=0, jf=0, k=0)],
                    -2: [bpf.Ins(code=bpf.BPF_RET | bpf.BPF_K,
                                 jt=0, jf=0, k=1)],
                    -1: [],
                    0: [
                        bpf.Ins(
                            code=bpf.BPF_W | bpf.BPF_LD | bpf.BPF_ABS,
                            jt=0, jf=0, k=10),
                        bpf.Ins(
                            code=bpf.BPF_JMP | bpf.BPF_JGT | bpf.BPF_K,
                            jt=-2, jf=-3, k=23.4)
                    ]
                })

    def test_dce(self):
        for t in self.ts:
            cfg = bpf.construct_cfg(t, 0.5)
            node_ret_tys = bpf.construct_node_ret_tys(cfg)
            ccfg = bpf.collapse_cfg(cfg, node_ret_tys)
            fragments = bpf.construct_fragments(ccfg, node_ret_tys)
            dead_fragments = copy.deepcopy(fragments)
            dead_fragments[50] = [None]
            fragments[50] = []
            dce_fragments = bpf.dce(cfg, dead_fragments)
            self.assertEqual(fragments, dce_fragments)

    def test_merge_cfg(self):
        def fold_cfg(t):
            cfg = bpf.construct_cfg(t, 0.5)
            node_ret_tys = bpf.construct_node_ret_tys(cfg)
            return bpf.collapse_cfg(cfg, node_ret_tys)
        cfgs = [fold_cfg(t) for t in self.ts]
        mcfg = bpf.merge_cfgs(cfgs)
        self.assertEqual(
            sorted(mcfg.edges(data=True)),
            [
                (-3, -1, {}),
                (-2, -1, {}),
                (-1, 4, {}),
                (0, -3, {'data': bpf.Direction.LEFT}),
                (0, -2, {'data': bpf.Direction.RIGHT}),
                (1, 3, {}),
                (2, 3, {}),
                (3, 5, {}),
                (4, 1, {'data': bpf.Direction.LEFT}),
                (4, 2, {'data': bpf.Direction.RIGHT}),
                (5, 6, {'data': bpf.Direction.LEFT}),
                (5, 7, {'data': bpf.Direction.RIGHT})
            ])

        self.assertEqual(
            sorted(mcfg.nodes(data=True)),
            [
                (-3, {'ann': bpf.Node(ty=bpf.NodeTy.ENSEMBLE_LEAF, args=(0,))}),
                (-2, {'ann': bpf.Node(ty=bpf.NodeTy.ENSEMBLE_LEAF, args=(1,))}),
                (-1, {'ann': bpf.Node(ty=bpf.NodeTy.EXIT, args=())}),
                (0, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(10, 23.4))}),
                (1, {'ann': bpf.Node(ty=bpf.NodeTy.ENSEMBLE_LEAF, args=(0,))}),
                (2, {'ann': bpf.Node(ty=bpf.NodeTy.ENSEMBLE_LEAF, args=(1,))}),
                (3, {'ann': bpf.Node(ty=bpf.NodeTy.EXIT, args=())}),
                (4, {'ann': bpf.Node(ty=bpf.NodeTy.BRANCH, args=(10, 23.4))}),
                (5, {'ann': bpf.Node(ty=bpf.NodeTy.VOTE, args=())}),
                (6, {'ann': bpf.Node(bpf.NodeTy.LEAF, args=(1,))}),
                (7, {'ann': bpf.Node(bpf.NodeTy.LEAF, args=(0,))})
            ])

    def test_ensemble_fragments(self):
        def fold_cfg(t):
            cfg = bpf.construct_cfg(t, 0.5)
            node_ret_tys = bpf.construct_node_ret_tys(cfg)
            return bpf.collapse_cfg(cfg, node_ret_tys)
        cfgs = [fold_cfg(t) for t in self.ts]
        mcfg = bpf.merge_cfgs(cfgs)
        frags = bpf.construct_ensemble_fragments(mcfg)
        self.assertEqual(
            frags,
            {-3: [bpf.Ins(code=0, jt=0, jf=0, k=15),
                  bpf.Ins(code=20, jt=0, jf=0, k=1),
                  bpf.Ins(code=2, jt=0, jf=0, k=15),
                  bpf.Ins(code=5, jt=0, jf=0, k=-1)],
             -2: [bpf.Ins(code=0, jt=0, jf=0, k=15),
                  bpf.Ins(code=4, jt=0, jf=0, k=1),
                  bpf.Ins(code=2, jt=0, jf=0, k=15),
                  bpf.Ins(code=5, jt=0, jf=0, k=-1)],
             -1: [],
             0: [bpf.Ins(code=32, jt=0, jf=0, k=10),
                 bpf.Ins(code=37, jt=-2, jf=-3, k=23.4)],
             1: [bpf.Ins(code=0, jt=0, jf=0, k=15),
                 bpf.Ins(code=20, jt=0, jf=0, k=1),
                 bpf.Ins(code=2, jt=0, jf=0, k=15),
                 bpf.Ins(code=5, jt=0, jf=0, k=3)],
             2: [bpf.Ins(code=0, jt=0, jf=0, k=15),
                 bpf.Ins(code=4, jt=0, jf=0, k=1),
                 bpf.Ins(code=2, jt=0, jf=0, k=15),
                 bpf.Ins(code=5, jt=0, jf=0, k=3)],
             3: [],
             4: [bpf.Ins(code=32, jt=0, jf=0, k=10),
                 bpf.Ins(code=37, jt=2, jf=1, k=23.4)],
             5: [bpf.Ins(code=0, jt=0, jf=0, k=15),
                 bpf.Ins(code=37, jt=6, jf=7, k=0)],
             6: [bpf.Ins(code=6, jt=0, jf=0, k=1)],
             7: [bpf.Ins(code=6, jt=0, jf=0, k=0)]
            })


    def test_linearize(self):
        def fold_cfg(t):
            cfg = bpf.construct_cfg(t, 0.5)
            node_ret_tys = bpf.construct_node_ret_tys(cfg)
            return bpf.collapse_cfg(cfg, node_ret_tys)
        cfgs = [fold_cfg(t) for t in self.ts]
        mcfg = bpf.merge_cfgs(cfgs)
        frags = bpf.construct_ensemble_fragments(mcfg)
        inss, label_offsets = bpf.linearize_ensemble(mcfg, frags)
        self.assertEqual(
            inss,
            [
                bpf.Ins(code=32, jt=0, jf=0, k=10),
                bpf.Ins(code=37, jt=-2, jf=-3, k=23.4),
                bpf.Ins(code=0, jt=0, jf=0, k=15),
                bpf.Ins(code=4, jt=0, jf=0, k=1),
                bpf.Ins(code=2, jt=0, jf=0, k=15),
                bpf.Ins(code=5, jt=0, jf=0, k=-1),
                bpf.Ins(code=0, jt=0, jf=0, k=15),
                bpf.Ins(code=20, jt=0, jf=0, k=1),
                bpf.Ins(code=2, jt=0, jf=0, k=15),
                bpf.Ins(code=5, jt=0, jf=0, k=-1),
                bpf.Ins(code=32, jt=0, jf=0, k=10),
                bpf.Ins(code=37, jt=2, jf=1, k=23.4),
                bpf.Ins(code=0, jt=0, jf=0, k=15),
                bpf.Ins(code=4, jt=0, jf=0, k=1),
                bpf.Ins(code=2, jt=0, jf=0, k=15),
                bpf.Ins(code=5, jt=0, jf=0, k=3),
                bpf.Ins(code=0, jt=0, jf=0, k=15),
                bpf.Ins(code=20, jt=0, jf=0, k=1),
                bpf.Ins(code=2, jt=0, jf=0, k=15),
                bpf.Ins(code=5, jt=0, jf=0, k=3),
                bpf.Ins(code=0, jt=0, jf=0, k=15),
                bpf.Ins(code=37, jt=6, jf=7, k=0),
                bpf.Ins(code=6, jt=0, jf=0, k=0),
                bpf.Ins(code=6, jt=0, jf=0, k=1)
            ])
        self.assertEqual(
            label_offsets,
            {0: 0, 1: 16, 2: 12, 3: 20, 4: 10, 5: 20, 6: 23,
             7: 22, -2: 2, -3: 6, -1: 10}
        )

    def test_assemble(self):
        def fold_cfg(t):
            cfg = bpf.construct_cfg(t, 0.5)
            node_ret_tys = bpf.construct_node_ret_tys(cfg)
            return bpf.collapse_cfg(cfg, node_ret_tys)
        cfgs = [fold_cfg(t) for t in self.ts]
        mcfg = bpf.merge_cfgs(cfgs)
        frags = bpf.construct_ensemble_fragments(mcfg)
        inss, label_offsets = bpf.linearize_ensemble(mcfg, frags)
        inss = bpf.assemble_ensemble(inss, label_offsets)
        self.assertEqual(
            inss,
            [
                bpf.Ins(code=32, jt=0, jf=0, k=10),
                bpf.Ins(code=37, jt=0, jf=4, k=23.4),
                bpf.Ins(code=0, jt=0, jf=0, k=15),
                bpf.Ins(code=4, jt=0, jf=0, k=1),
                bpf.Ins(code=2, jt=0, jf=0, k=15),
                bpf.Ins(code=5, jt=0, jf=0, k=-1),
                bpf.Ins(code=0, jt=0, jf=0, k=15),
                bpf.Ins(code=20, jt=0, jf=0, k=1),
                bpf.Ins(code=2, jt=0, jf=0, k=15),
                bpf.Ins(code=5, jt=0, jf=0, k=-1),
                bpf.Ins(code=32, jt=0, jf=0, k=10),
                bpf.Ins(code=37, jt=0, jf=4, k=23.4),
                bpf.Ins(code=0, jt=0, jf=0, k=15),
                bpf.Ins(code=4, jt=0, jf=0, k=1),
                bpf.Ins(code=2, jt=0, jf=0, k=15),
                bpf.Ins(code=5, jt=0, jf=0, k=3),
                bpf.Ins(code=0, jt=0, jf=0, k=15),
                bpf.Ins(code=20, jt=0, jf=0, k=1),
                bpf.Ins(code=2, jt=0, jf=0, k=15),
                bpf.Ins(code=5, jt=0, jf=0, k=3),
                bpf.Ins(code=0, jt=0, jf=0, k=15),
                bpf.Ins(code=37, jt=1, jf=0, k=0),
                bpf.Ins(code=6, jt=0, jf=0, k=0),
                bpf.Ins(code=6, jt=0, jf=0, k=1)
            ])
