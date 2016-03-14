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

CXX_COMPILER = sysconfig.get_config_var('CXX')

EVALUATE_FN_NAME = "evaluate"
ALWAYS_INLINE = "__attribute__((__always_inline__))"


class CodeGenerator(object):
    def __init__(self):
        self._file = tempfile.NamedTemporaryFile(prefix='compiledtrees_', suffix='.cpp', delete=True)
        self._indent = 0

    @property
    def file(self):
        self._file.flush()
        return self._file

    def write(self, line):
        self._file.write("  " * self._indent + line + "\n")

    @contextlib.contextmanager
    def bracketed(self, preamble, postamble):
        assert self._indent >= 0
        self.write(preamble)
        self._indent += 1
        yield
        self._indent -= 1
        self.write(postamble)


def code_gen_tree(tree, evaluate_fn=EVALUATE_FN_NAME, gen=None):
    """
    Generates C code representing the evaluation of a tree.

    Writes code similar to:
    ```
        extern "C" {
          __attribute__((__always_inline__)) float evaluate(float* f) {
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
            gen.write("return {0}f;".format(tree.value[node].item()))
            return

        branch = "if (f[{feature}] <= {threshold}f) {{".format(
            feature=tree.feature[node],
            threshold=tree.threshold[node])
        with gen.bracketed(branch, "}"):
            recur(tree.children_left[node])

        with gen.bracketed("else {", "}"):
            recur(tree.children_right[node])

    with gen.bracketed('extern "C" {', "}"):
        fn_decl = "{inline} float {name}(float* f) {{".format(
            inline=ALWAYS_INLINE,
            name=evaluate_fn)
        with gen.bracketed(fn_decl, "}"):
            recur(0)
    return gen.file

def _gen_tree(i, tree):
    """
    Generates cpp code for i'th tree.
    Moved out of code_gen_ensemble scope for parallelization.
    """
    name = "{name}_{index}".format(name=EVALUATE_FN_NAME, index=i)
    gen_tree = CodeGenerator()
    return code_gen_tree(tree, name, gen_tree)

def code_gen_ensemble(trees, individual_learner_weight, initial_value,
                      gen=None, n_jobs=1):
    """
    Writes code similar to:

    ```
    extern "C" {
      __attribute__((__always_inline__)) float evaluate_partial_0(float* f) {
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
      __attribute__((__always_inline__)) float evaluate_partial_1(float* f) {
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
      float evaluate(float* f) {
        float result = 0.0;
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

    tree_files =[_gen_tree(i, tree) for i, tree in enumerate(trees)]

    with gen.bracketed('extern "C" {', "}"):
        # add dummy definitions if you will compile in parallel
        for i, tree in enumerate(trees):
            name = "{name}_{index}".format(name=EVALUATE_FN_NAME, index=i)
            gen.write("float {name}(float* f);".format(name=name))

        fn_decl = "float {name}(float* f) {{".format(name=EVALUATE_FN_NAME)
        with gen.bracketed(fn_decl, "}"):
            gen.write("float result = {0}f;".format(initial_value))
            for i, _ in enumerate(trees):
                increment = "result += {name}_{index}(f) * {weight}f;".format(
                    name=EVALUATE_FN_NAME,
                    index=i,
                    weight=individual_learner_weight)
                gen.write(increment)
            gen.write("return result;")
    return tree_files + [gen.file]

def _compile(cpp_f):
    o_f = tempfile.NamedTemporaryFile(prefix='compiledtrees_', suffix='.o', delete=True)
    _call([CXX_COMPILER, cpp_f, "-c", "-fPIC", "-o", o_f.name, "-O3"])
    return o_f

def _call(args):
    DEVNULL = open(os.devnull, 'w')
    subprocess.check_call(" ".join(args),
                          shell=True, stdout=DEVNULL, stderr=DEVNULL)

def compile_code_to_object(files, n_jobs=1):
    # if ther is a single file then create single element list
    # unicode for filename; name attribute for file-like objects
    if type(files) is unicode or hasattr(files, 'name'):
        files = [files]

    so_f = tempfile.NamedTemporaryFile(prefix='compiledtrees_', suffix='.so', delete=True)
    o_files = Parallel(n_jobs=n_jobs, backend='threading')(delayed(_compile)(f.name) for f in files)
    # link trees
    _call([CXX_COMPILER, "-shared"] + [f.name for f in o_files] + ["-fPIC",
        "-flto", "-o", so_f.name, "-O3"])
    return so_f
