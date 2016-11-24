from sklearn import tree
from sklearn.base import ClassifierMixin

import copy
import itertools
import numpy as np
import enum
import collections
import networkx as nx


class Direction(enum.Enum):
    LEFT = 0
    RIGHT = 1


class NodeTy(enum.Enum):
    BRANCH = 0
    LEAF = 1
    EXIT = 2
    VOTE = 3
    ENSEMBLE_LEAF = 4


Node = collections.namedtuple("Node", ["ty", "args"])
Ins = collections.namedtuple("Ins", ["code", "jt", "jf", "k"])

EXIT_NODE, ACCEPT_NODE, REJECT_NODE = -1, -2, -3
EXIT = Node(ty=NodeTy.EXIT, args=())
ACCEPT = Node(ty=NodeTy.LEAF, args=(1,))
REJECT = Node(ty=NodeTy.LEAF, args=(0,))

ENSEMBLE_VOTING_REGISTER = 15


def visit(tree, node, visit_leaf, visit_inner, join):
    if tree.children_left[node] == -1:
        return join(node, visit_leaf(node), None, None)
    cur = visit_inner(node)
    left = visit(
        tree, tree.children_left[node], visit_leaf, visit_inner, join)
    right = visit(
        tree, tree.children_right[node], visit_leaf, visit_inner, join)
    return join(node, cur, left, right)


def construct_cfg(tree, threshold):
    cfg = nx.DiGraph()

    cfg.add_node(EXIT_NODE, ann=EXIT)
    cfg.add_node(ACCEPT_NODE, ann=ACCEPT)
    cfg.add_node(REJECT_NODE, ann=REJECT)

    cfg.add_edge(ACCEPT_NODE, EXIT_NODE)
    cfg.add_edge(REJECT_NODE, EXIT_NODE)

    def visit_leaf(node):
        negatives, positives = tree.value[node].ravel()
        probability = float(positives) / (negatives + positives)
        return REJECT_NODE if probability > threshold else ACCEPT_NODE

    def visit_inner(node):
        feature = tree.feature[node]
        threshold = tree.threshold[node]
        return Node(ty=NodeTy.BRANCH, args=(feature, threshold))

    def join(node, cur, left, right):
        if left is None and right is None:
            return cur
        cfg.add_node(node, ann=cur)
        cfg.add_edge(node, left, data=Direction.LEFT)
        cfg.add_edge(node, right, data=Direction.RIGHT)
        return node

    visit(tree, 0, visit_leaf, visit_inner, join)
    return cfg


def construct_node_ret_tys(cfg):
    node_ret_tys = {}
    for n in nx.dfs_postorder_nodes(cfg, 0):
        ty = cfg.node[n]['ann'].ty
        if ty == NodeTy.EXIT:
            continue
        if ty == NodeTy.LEAF:
            node_ret_tys[n] = {cfg.node[n]['ann']}
            continue
        if ty == NodeTy.BRANCH:
            parents = [node_ret_tys[c] for c in cfg.successors(n)]
            result = set()
            for parent in parents:
                result |= parent
            node_ret_tys[n] = result
            continue
    return node_ret_tys


BPF_LD = 0x00
BPF_ST = 0x02
BPF_ALU = 0x04
BPF_JMP = 0x05
BPF_RET = 0x06

# ld/ldx fields
BPF_W = 0x00
BPF_H = 0x08
BPF_B = 0x10
BPF_ABS = 0x20

# alu/jmp fields
BPF_ADD = 0x00
BPF_SUB = 0x10
BPF_JA = 0x00
BPF_JEQ = 0x10
BPF_JGT = 0x20
BPF_K = 0x00


def bpf_jump(code, k, jt, jf):
    return Ins(code, jt, jf, k)


def bpf_stmt(code, k):
    return bpf_jump(code, k, 0, 0)


def extract_feature_bpf(feature):
    offset = feature
    # offset = OFFSETS[features[feature]]
    return [
        bpf_stmt(BPF_W | BPF_LD | BPF_ABS, offset),
    ]


def collapse_cfg(cfg, node_ret_tys):
    import copy
    cfg = copy.deepcopy(cfg)
    for n in list(nx.dfs_postorder_nodes(cfg, 0)):
        if n not in node_ret_tys or len(node_ret_tys[n]) is not 1:
            continue
        if n in (REJECT_NODE, ACCEPT_NODE, EXIT_NODE):
            continue

        (leaf,) = node_ret_tys[n]
        assert leaf in (ACCEPT, REJECT)
        leaf_node = ACCEPT_NODE if leaf == ACCEPT else REJECT_NODE
        # Add source nodes
        for (src, dst, direction) in cfg.edges(data='data'):
            if dst != n:
                continue
            cfg.add_edge(src, leaf_node, data=direction)
        cfg.remove_node(n)
    return cfg


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def merge_cfgs(cfgs):
    # Algorithm:
    # a) Renumber all nodes/edges and insert into the graph.
    # b) Insert edge from the `exit` node of the cfg
    # to the `input` node of the next cfg.

    G = collections.namedtuple('G', ['input_node', 'exit_node'])
    gs = []
    mcfg = nx.DiGraph()

    def f(cfg):
        min_current_node = min(cfg.nodes())
        if not mcfg.nodes():
            max_existing_node = min_current_node - 1
        else:
            max_existing_node = max(mcfg.nodes())
        offset = max_existing_node - min_current_node + 1

        for (node, data) in cfg.nodes(data=True):
            if 'ann' in data and data['ann'].ty == NodeTy.LEAF:
                data['ann'] = data['ann']._replace(ty=NodeTy.ENSEMBLE_LEAF)
            mcfg.add_node(node + offset, **data)
        for (src, dst, data) in cfg.edges(data=True):
            mcfg.add_edge(src + offset, dst + offset, **data)
        (exit_node,) = [n for (n, data) in cfg.nodes(data=True)
                        if data['ann'].ty == NodeTy.EXIT]
        return G(input_node=0 + offset, exit_node=exit_node + offset)
    gs = [f(cfg) for cfg in cfgs]
    for (pred, succ) in pairwise(gs):
        mcfg.add_edge(pred.exit_node, succ.input_node)

    ensemble_vote_node = max(mcfg.nodes()) + 1
    ensemble_accept_node = max(mcfg.nodes()) + 2
    ensemble_reject_node = max(mcfg.nodes()) + 3
    mcfg.add_edge(gs[-1].exit_node, ensemble_vote_node)
    mcfg.add_node(ensemble_vote_node, ann=Node(ty=NodeTy.VOTE, args=()))
    mcfg.add_node(ensemble_accept_node, ann=ACCEPT)
    mcfg.add_node(ensemble_reject_node, ann=REJECT)
    mcfg.add_edge(ensemble_vote_node, ensemble_accept_node,
                  data=Direction.LEFT)
    mcfg.add_edge(ensemble_vote_node, ensemble_reject_node,
                  data=Direction.RIGHT)
    return mcfg


def construct_fragments(cfg, node_ret_tys=None):
    fragments = {}
    for n in nx.dfs_postorder_nodes(cfg, 0):
        ty = cfg.node[n]['ann'].ty
        if ty == NodeTy.EXIT:
            fragments[n] = []
            continue
        if ty == NodeTy.LEAF:
            if node_ret_tys:
                assert n in node_ret_tys and len(node_ret_tys[n]) == 1
            (decision,) = cfg.node[n]['ann'].args
            fragments[n] = [bpf_stmt(BPF_RET | BPF_K, decision)]
            continue
        if ty == NodeTy.BRANCH:
            if node_ret_tys:
                assert n in node_ret_tys and len(node_ret_tys[n]) > 1
            (feature, threshold) = cfg.node[n]['ann'].args
            threshold = threshold

            left, right = None, None
            for (src, dst, direction) in cfg.edges(data='data'):
                if src != n:
                    continue
                if direction == Direction.LEFT:
                    left = dst
                    continue
                if direction == Direction.RIGHT:
                    right = dst
            assert left
            assert right
            # If true, jump to the right
            # Note: tests X[f] > threshold, so logic is inverted.
            # Thus, in the true branch, we jump the entire left subtree,
            # and on failure we just continue.
            compare = bpf_jump(
                BPF_JMP | BPF_JGT | BPF_K, threshold, right, left)
            frag = extract_feature_bpf(feature) + [compare]
            fragments[n] = frag
            continue
    return fragments


def construct_ensemble_fragments(cfg, node_ret_tys=None):
    fragments = {}
    for n in nx.dfs_postorder_nodes(cfg, 0):
        ty = cfg.node[n]['ann'].ty
        if ty == NodeTy.EXIT:
            fragments[n] = []
            continue
        elif ty == NodeTy.LEAF:
            if node_ret_tys:
                assert n in node_ret_tys and len(node_ret_tys[n]) == 1
            (decision,) = cfg.node[n]['ann'].args
            fragments[n] = [bpf_stmt(BPF_RET | BPF_K, decision)]
            continue
        elif ty == NodeTy.ENSEMBLE_LEAF:
            if node_ret_tys:
                assert n in node_ret_tys and len(node_ret_tys[n]) == 1
            (decision,) = cfg.node[n]['ann'].args
            (exit_node,) = cfg.successors(n)
            # XXX: DIRECTION!
            # Decision == 1 -> accept
            # Decision == 0 -> reject
            fragments[n] = [
                # Load current count from register.
                bpf_stmt(BPF_LD | BPF_K, ENSEMBLE_VOTING_REGISTER),
                # Increment/decrement count of accumulator
                bpf_stmt((BPF_ADD if decision else BPF_SUB) | BPF_ALU, 1),
                # Store new value in register
                bpf_stmt(BPF_ST | BPF_K, ENSEMBLE_VOTING_REGISTER),
                # Jump to the exit node.
                bpf_stmt(BPF_JMP | BPF_JA, exit_node),
            ]
            continue
        elif ty == NodeTy.BRANCH:
            if node_ret_tys:
                assert n in node_ret_tys and len(node_ret_tys[n]) > 1
            (feature, threshold) = cfg.node[n]['ann'].args
            threshold = threshold

            left, right = None, None
            for (src, dst, direction) in cfg.edges(data='data'):
                if src != n:
                    continue
                if direction == Direction.LEFT:
                    left = dst
                    continue
                if direction == Direction.RIGHT:
                    right = dst
            assert left
            assert right
            # If true, jump to the right
            # Note: tests X[f] > threshold, so logic is inverted.
            # Thus, in the true branch, we jump the entire left subtree,
            # and on failure we just continue.
            compare = bpf_jump(
                BPF_JMP | BPF_JGT | BPF_K, threshold, right, left)
            frag = extract_feature_bpf(feature) + [compare]
            fragments[n] = frag
            continue
        elif ty == NodeTy.VOTE:
            left, right = None, None
            for (src, dst, direction) in cfg.edges(data='data'):
                if src != n:
                    continue
                if direction == Direction.LEFT:
                    left = dst
                    continue
                if direction == Direction.RIGHT:
                    right = dst
            assert left
            assert right

            # If true, jump to the right
            # Note: tests X[f] > threshold, so logic is inverted.
            fragments[n] = [
                # Load current count from register.
                bpf_stmt(BPF_LD | BPF_K, ENSEMBLE_VOTING_REGISTER),
                # If K > 0, reject, otherwise accept.
                # XXX: DIRECTION!
                bpf_jump(BPF_JMP | BPF_JGT | BPF_K, 0, left, right),
            ]
            continue
        else:
            raise Exception("Unhandled node type, {}, {}".format(n, cfg.node[n]))
    return fragments


def dce(cfg, fragments):
    fragments = copy.deepcopy(fragments)
    visited = {0}
    for n in nx.dfs_preorder_nodes(cfg, 0):
        ty = cfg.node[n]['ann'].ty
        if ty == NodeTy.BRANCH:
            jmp = fragments[n][-1]
            if jmp.code == BPF_JMP | BPF_JGT | BPF_K:
                visited |= {jmp.jt, jmp.jf}

    for dst in fragments.keys():
        if dst not in visited:
            fragments[dst] = []
    return fragments


def linearize_ensemble(cfg, fragments):
    inss = []
    label_offsets = {}
    for n in reversed(list(nx.dfs_postorder_nodes(cfg, 0))):
        label_offsets[n] = len(inss)
        inss += fragments[n]
    return (inss, label_offsets)


def linearize(cfg, fragments):
    inss = []
    label_offsets = {}
    # TODO: Fix traversal order
    for n in nx.dfs_postorder_nodes(cfg.reverse(copy=True), -1):
        label_offsets[n] = len(inss)
        inss += fragments[n]
    return (inss, label_offsets)


def assemble_ensemble(inss, label_offsets):
    inss = copy.deepcopy(inss)
    for i, ins in enumerate(inss):
        if ins.code == BPF_JMP | BPF_JGT | BPF_K:
            # Special case
            print(ins)
            jt_abs = label_offsets[ins.jt]
            jf_abs = label_offsets[ins.jf]
            jt_rel = jt_abs - i - 1
            jf_rel = jf_abs - i - 1
            assert jt_rel >= 0
            assert jf_rel >= 0
            inss[i] = inss[i]._replace(jt=jt_rel, jf=jf_rel)
    return inss


def assemble(inss, label_offsets):
    inss = copy.deepcopy(inss)
    for i, ins in enumerate(inss):
        if ins.code == BPF_JMP | BPF_JGT | BPF_K:
            jt_abs = label_offsets[ins.jt]
            jf_abs = label_offsets[ins.jf]
            jt_rel = jt_abs - i - 1
            jf_rel = jf_abs - i - 1
            assert jt_rel >= 0
            assert jf_rel >= 0
            inss[i] = inss[i]._replace(jt=jt_rel, jf=jf_rel)
    return inss


def interpret(inss, features):
    ip = -1
    acc = None
    while True:
        ip += 1
        ins = inss[ip]
        if ins.code == BPF_W | BPF_LD | BPF_ABS:
            # Load from features
            acc = features[ins.k]
            assert inss[ip+1].code == BPF_JMP | BPF_JGT | BPF_K
            continue
        elif ins.code == BPF_JMP | BPF_JGT | BPF_K:
            if acc > ins.k:
                ip += ins.jt
            else:
                ip += ins.jf
            assert inss[ip+1].code in (BPF_W | BPF_LD | BPF_ABS,
                                       BPF_RET | BPF_K)
            continue
        elif ins.code == BPF_RET | BPF_K:
            return ins.k
        else:
            raise Exception("Unknown opcode: {}".format(ins))


class BpfClassifierPredictor(ClassifierMixin):
    def __init__(self, clf, decision_threshold=0.5):
        if not self.compilable(clf):
            raise ValueError("Invalid classifier: {}".format(clf))
        cfg = construct_cfg(clf.tree_, decision_threshold)
        node_ret_tys = construct_node_ret_tys(cfg)
        cfg = collapse_cfg(cfg, node_ret_tys)
        fragments = construct_fragments(cfg, node_ret_tys)
        fragments = dce(cfg, fragments)
        inss, label_offsets = linearize(cfg, fragments)
        assembled_inss = assemble(inss, label_offsets)
        self.assembled_ins = assembled_inss

    def predict(self, X):
        (n_samples, n_features) = X.shape
        Y_pred = np.zeros((n_samples,))
        for i in range(n_samples):
            Y_pred[i] = interpret(self.assembled_ins, X[i])
        # In BPF, we invert the signals (drop is 0, pass is 1).
        return 1 - Y_pred

    @classmethod
    def compilable(cls, clf):
        if isinstance(clf, tree.DecisionTreeClassifier) \
           and clf.tree_ is not None:
            return True
        return False
