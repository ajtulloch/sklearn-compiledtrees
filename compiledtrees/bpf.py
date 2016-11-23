from sklearn import tree
from sklearn.base import ClassifierMixin

import copy
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

Node = collections.namedtuple("Node", ["ty", "args"])
Ins = collections.namedtuple("Ins", ["code", "jt", "jf", "k"])

exit_node, accept_node, reject_node = -1, -2, -3
exit = Node(ty=NodeTy.EXIT, args=())
accept = Node(ty=NodeTy.LEAF, args=(1,))
reject = Node(ty=NodeTy.LEAF, args=(0,))


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

    cfg.add_node(exit_node, ann=exit)
    cfg.add_node(accept_node, ann=accept)
    cfg.add_node(reject_node, ann=reject)

    cfg.add_edge(accept_node, exit_node)
    cfg.add_edge(reject_node, exit_node)

    def visit_leaf(node):
        negatives, positives = tree.value[node].ravel()
        probability = float(positives) / (negatives + positives)
        return reject_node if probability > threshold else accept_node

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
BPF_JMP = 0x05
BPF_RET = 0x06

# ld/ldx fields
BPF_W = 0x00
BPF_H = 0x08
BPF_B = 0x10
BPF_ABS = 0x20

# alu/jmp fields
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
        if n in (reject_node, accept_node, exit_node):
            continue

        (leaf,) = node_ret_tys[n]
        assert leaf in (accept, reject)
        leaf_node = accept_node if leaf == accept else reject_node
        # Add source nodes
        for (src, dst, direction) in cfg.edges(data='data'):
            if dst != n:
                continue
            cfg.add_edge(src, leaf_node, data=direction)
        cfg.remove_node(n)
        print("Removing: ", n)
        for (src, dst, direction) in cfg.edges(data='data'):
            if src == n or dst == n:
                raise Exception("WTF")
    return cfg


def construct_fragments(cfg, node_ret_tys):
    fragments = {}
    for n in nx.dfs_postorder_nodes(cfg, 0):
        ty = cfg.node[n]['ann'].ty
        if ty == NodeTy.EXIT:
            fragments[n] = []
            continue
        if ty == NodeTy.LEAF:
            assert n in node_ret_tys and len(node_ret_tys[n]) == 1
            (retval,) = node_ret_tys[n]
            (decision,) = retval.args
            fragments[n] = [bpf_stmt(BPF_RET | BPF_K, decision)]
            continue
        if ty == NodeTy.BRANCH:
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


def linearize(cfg, fragments):
    inss = []
    label_offsets = {}
    for n in nx.dfs_postorder_nodes(cfg.reverse(copy=True), -1):
        label_offsets[n] = len(inss)
        if n not in fragments:
            raise Exception("Error: {}, {}".format(cfg, fragments))
        inss += fragments[n]
    return (inss, label_offsets)


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
            assert inss[ip+1].code in (BPF_W | BPF_LD | BPF_ABS, BPF_RET | BPF_K)
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
