from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from distutils import sysconfig

import contextlib
import os
import subprocess
import tempfile
from joblib import Parallel, delayed
from compiledtrees.utils import convert_to_quasi_float

import platform

if platform.system() == 'Windows':
    CXX_COMPILER = os.environ['CXX'] if 'CXX' in os.environ else None
    delete_files = False
else:
    CXX_COMPILER = sysconfig.get_config_var('CXX')
    delete_files = True

EVALUATE_FN_NAME = "evaluate"
ALWAYS_INLINE = "__attribute__((__always_inline__))"


class CodeGenerator(object):
    def __init__(self):
        self._file = tempfile.NamedTemporaryFile(mode='w+b',
                                                 prefix='compiledtrees_',
                                                 suffix='.cpp',
                                                 delete=delete_files)
        self._indent = 0

    @property
    def file(self):
        self._file.flush()
        return self._file

    def write(self, line):
        self._file.write(("  " * self._indent + line + "\n").encode("ascii"))

    @contextlib.contextmanager
    def bracketed(self, preamble, postamble):
        assert self._indent >= 0
        self.write(preamble)
        self._indent += 1
        yield
        self._indent -= 1
        self.write(postamble)


def code_gen_regressor_tree(tree, evaluate_fn=EVALUATE_FN_NAME, gen=None):
    """
    Generates C code representing the evaluation of a tree.

    Writes code similar to:
    ```
        extern "C" {
          __attribute__((__always_inline__)) double evaluate(float* f) {
            if (f[9] <= 0.175931170583) {
              return 0.0;
            }
            else {
              return 1.0;
            }
          }
        }
    ```

    to the given CodeGenerator object.
    """
    if gen is None:
        gen = CodeGenerator()

    def recur(node):
        if tree.children_left[node] == -1:
            assert tree.value[node].size == 1
            gen.write("return {0};".format(tree.value[node].item()))
            return

        branch = "if (f[{feature}] <= {threshold}f) {{".format(
            feature=tree.feature[node],
            threshold=tree.threshold[node])
        with gen.bracketed(branch, "}"):
            recur(tree.children_left[node])

        with gen.bracketed("else {", "}"):
            recur(tree.children_right[node])

    with gen.bracketed('extern "C" {', "}"):
        fn_decl = "{inline} double {name}(float* f) {{".format(
            inline=ALWAYS_INLINE,
            name=evaluate_fn)
        with gen.bracketed(fn_decl, "}"):
            recur(0)
    return gen.file


def _gen_regressor_tree(i, tree):
    """
    Generates cpp code for i'th tree.
    Moved out of code_gen_ensemble_regressor scope for parallelization.
    """
    name = "{name}_{index}".format(name=EVALUATE_FN_NAME, index=i)
    gen_tree = CodeGenerator()
    return code_gen_regressor_tree(tree, name, gen_tree)


def code_gen_ensemble_regressor(trees, individual_learner_weight, initial_value,
                                gen=None, n_jobs=1):
    """
    Writes code similar to:

    ```
    extern "C" {
      __attribute__((__always_inline__)) double evaluate_partial_0(float* f) {
        if (f[4] <= 0.662200987339) {
          return 1.0;
        }
        else {
          if (f[8] <= 0.804652512074) {
            return 0.0;
          }
          else {
            return 1.0;
          }
        }
      }
    }
    extern "C" {
      __attribute__((__always_inline__)) double evaluate_partial_1(float* f) {
        if (f[4] <= 0.694428026676) {
          return 1.0;
        }
        else {
          if (f[7] <= 0.4402526021) {
            return 1.0;
          }
          else {
            return 0.0;
          }
        }
      }
    }

    extern "C" {
      double evaluate(float* f) {
        double result = 0.0;
        result += evaluate_partial_0(f) * 0.1;
        result += evaluate_partial_1(f) * 0.1;
        return result;
      }
    }
    ```

    to the given CodeGenerator object.
    """

    if gen is None:
        gen = CodeGenerator()

    tree_files = [_gen_regressor_tree(i, tree) for i, tree in enumerate(trees)]

    with gen.bracketed('extern "C" {', "}"):
        # add dummy definitions if you will compile in parallel
        for i, tree in enumerate(trees):
            name = "{name}_{index}".format(name=EVALUATE_FN_NAME, index=i)
            gen.write("double {name}(float* f);".format(name=name))

        fn_decl = "double {name}(float* f) {{".format(name=EVALUATE_FN_NAME)
        with gen.bracketed(fn_decl, "}"):
            gen.write("double result = {0};".format(initial_value))
            for i, _ in enumerate(trees):
                increment = "result += {name}_{index}(f) * {weight};".format(
                    name=EVALUATE_FN_NAME,
                    index=i,
                    weight=individual_learner_weight)
                gen.write(increment)
            gen.write("return result;")
    return tree_files + [gen.file]


# classifier code goes below
def code_gen_classifier_tree(tree, evaluate_fn=EVALUATE_FN_NAME, gen=None, weight=1.):
    """
    Generates C code representing the evaluation of a tree.

    Writes code similar to:
    ```
        extern "C" {
          __attribute__((__always_inline__)) double evaluate(float* f, double* o) {
            if (f[9] <= 0.175931170583) {
              o[0] = 0;
              o[1] = 0.7;
            }
            else {
              o[0] = 0.3;
              o[1] = 0;
            }
          }
        }
    ```

    to the given CodeGenerator object.
    """
    if gen is None:
        gen = CodeGenerator()

    def recur(node):
        if tree.children_left[node] == -1:
            assert tree.value[node].shape[0] == 1
            n_leaf_samples = tree.value[node].sum()
            assert n_leaf_samples > 0
            for i, val in enumerate(tree.value[node][0]):
                gen.write("o[{i}] += {val};".format(i=i, val=float(val)*weight/n_leaf_samples))
            return

        branch = "if (f[{feature}] <= {threshold}f) {{".format(
            feature=tree.feature[node],
            threshold=tree.threshold[node])
        with gen.bracketed(branch, "}"):
            recur(tree.children_left[node])

        with gen.bracketed("else {", "}"):
            recur(tree.children_right[node])

    with gen.bracketed('extern "C" {', "}"):
        fn_decl = "{inline} void {name}(float* f, double* o) {{".format(
            inline=ALWAYS_INLINE,
            name=evaluate_fn)
        with gen.bracketed(fn_decl, "}"):
            recur(0)
    return gen.file


def _gen_classifier_tree(i, tree, weight):
    """
    Generates cpp code for i'th tree.
    Moved out of code_gen_ensemble_regressor scope for parallelization.
    """
    name = "{name}_{index}".format(name=EVALUATE_FN_NAME, index=i)
    gen_tree = CodeGenerator()
    return code_gen_classifier_tree(tree, name, gen_tree, weight)


def code_gen_ensemble_classifier(trees, individual_learner_weight, initial_value,
                                gen=None, n_jobs=1):
    """
    Writes code similar to:

    ```
    extern "C" {
      void evaluate(float* f, double* probas) {
        evaluate_partial_0(f, probas[0], 0.1);
        votes[(int) evaluate_partial_1(f)] += 0.1;
      }
    }
    ```

    to the given CodeGenerator object.
    """
    if gen is None:
        gen = CodeGenerator()

    tree_files = [_gen_classifier_tree(i, tree, individual_learner_weight) for i, tree in enumerate(trees)]

    with gen.bracketed('extern "C" {', "}"):
        # add dummy definitions if you will compile in parallel
        for i, tree in enumerate(trees):
            name = "{name}_{index}".format(name=EVALUATE_FN_NAME, index=i)
            gen.write("void {name}(float* f, double* probas);".format(name=name)) #FIXME: can this be int?

        fn_decl = "void {name}(float* f, double* probas) {{".format(name=EVALUATE_FN_NAME)
        with gen.bracketed(fn_decl, "}"):
            for i, _ in enumerate(trees):
                increment = "{name}_{index}(f, probas);".format(
                    name=EVALUATE_FN_NAME,
                    index=i)
                gen.write(increment)
    return tree_files + [gen.file]


# Quasi-float classifier code goes below

def code_gen_ensemble_classifier_quasi_float(trees, individual_learner_weight, generator=None):
    if generator is None:
        generator = CodeGenerator()

    with generator.bracketed('extern "C" {', "}"):
        for i, tree in enumerate(trees):
            function_name = "{name}_{index}".format(name=EVALUATE_FN_NAME, index=i)
            generator.write("void {name}(int* f, int* probas);".format(name=function_name))

        function_declare = "void {name}(int* f, int* probas) {{".format(name=EVALUATE_FN_NAME)
        with generator.bracketed(function_declare, "}"):
            for i, _ in enumerate(trees):
                increment = "{name}_{index}(f, probas);".format(
                    name=EVALUATE_FN_NAME,
                    index=i
                )
                generator.write(increment)

    tree_files = [
        gen_classifier_tree_quasi_float(i, tree, individual_learner_weight)
        for i, tree in enumerate(trees)
    ]
    tree_files.append(generator.file)

    return tree_files


def gen_classifier_tree_quasi_float(i, tree, weight):
    name = "{name}_{index}".format(name=EVALUATE_FN_NAME, index=i)
    gen_tree = CodeGenerator()
    return code_gen_classifier_tree_quasi_float(tree, name, gen_tree, weight)


def code_gen_classifier_tree_quasi_float(tree, evaluate_fn=EVALUATE_FN_NAME,
                                         generator=None, weight=1.):
    if generator is None:
        generator = CodeGenerator()

    def recur(node):
        if tree.children_left[node] == -1:
            assert tree.value[node].shape[0] == 1
            n_leaf_samples = tree.value[node].sum()
            assert n_leaf_samples > 0
            for i, val in enumerate(tree.value[node][0]):
                quasi_val = convert_to_quasi_float(float(val) * weight / n_leaf_samples)
                generator.write("o[{i}] += {val};".format(i=i, val=quasi_val))
            return

        branch = "if (f[{feature}] <= {threshold}) {{".format(
            feature=tree.feature[node],
            threshold=convert_to_quasi_float(tree.threshold[node])
        )
        with generator.bracketed(branch, "}"):
            recur(tree.children_left[node])

        with generator.bracketed("else {", "}"):
            recur(tree.children_right[node])

    with generator.bracketed('extern "C" {', "}"):
        fn_decl = "{inline} void {name}(int* f, int* o) {{".format(
            inline=ALWAYS_INLINE,
            name=evaluate_fn
        )
        with generator.bracketed(fn_decl, "}"):
            recur(0)

    return generator.file


def _compile(cpp_f):
    if CXX_COMPILER is None:
        raise Exception("CXX compiler was not found. You should set CXX "
                        "environmental variable")
    o_f = tempfile.NamedTemporaryFile(mode='w+b',
                                      prefix='compiledtrees_',
                                      suffix='.o',
                                      delete=delete_files)
    if platform.system() == 'Windows':
        o_f.close()
    _call([CXX_COMPILER, cpp_f, "-c", "-fPIC", "-o", o_f.name, "-O3", "-pipe"])
    return o_f


def _call(args):
    DEVNULL = open(os.devnull, 'w')
    subprocess.check_call(" ".join(args),
                          shell=True, stdout=DEVNULL, stderr=DEVNULL)


def compile_code_to_object(files, n_jobs=1):
    # if ther is a single file then create single element list
    # unicode for filename; name attribute for file-like objects
    if isinstance(files, str) or hasattr(files, 'name'):
        files = [files]

    # Close files on Windows to avoid permission errors
    if platform.system() == 'Windows':
        for f in files:
            f.close()

    o_files = (Parallel(n_jobs=n_jobs, backend='threading')
               (delayed(_compile)(f.name) for f in files))

    so_f = tempfile.NamedTemporaryFile(mode='w+b',
                                       prefix='compiledtrees_',
                                       suffix='.so',
                                       delete=delete_files)
    # Close files on Windows to avoid permission errors
    if platform.system() == 'Windows':
        so_f.close()

    # link trees
    if platform.system() == 'Windows':
        # a hack to overcome large RFs on windows and CMD 9182 chaacters limit
        list_ofiles = tempfile.NamedTemporaryFile(mode='w+b',
                                                  prefix='list_ofiles_',
                                                  delete=delete_files)
        for f in o_files:
            list_ofiles.write((f.name.replace('\\', '\\\\') +
                               "\r").encode('latin1'))
        list_ofiles.close()
        _call([CXX_COMPILER, "-shared", "@%s" % list_ofiles.name, "-fPIC",
               "-flto", "-o", so_f.name, "-O3", "-pipe"])

        # cleanup files
        for f in o_files:
            os.unlink(f.name)
        for f in files:
            os.unlink(f.name)
        os.unlink(list_ofiles.name)
    else:
        _call([CXX_COMPILER, "-shared"] +
              [f.name for f in o_files] +
              ["-fPIC", "-flto", "-o", so_f.name, "-O3", "-pipe"])

    return so_f
