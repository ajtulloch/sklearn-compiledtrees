cimport cython
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "dlfcn.h":
  void* dlopen(const char*, long)
  void* dlsym(void*, const char* )
  char* dlerror()
  void dlclose(void* handle)

cdef extern from "dlfcn.h":
  cdef long RTLD_NOW


cdef class BaseCompiledPredictor:
    def __cinit__(self, const char* filename, const char* symbol):
        cdef void* handle = dlopen(filename, RTLD_NOW)
        if handle == NULL:
            raise ValueError("Could not find compiled evaluation file")
        self.handle = handle
        cdef void* func = dlsym(self.handle, symbol)
        if func == NULL:
            raise ValueError("Could not find compiled evaluation function in file")
        self.func = func

    def __dealloc__(self):
        dlclose(self.handle)


cdef class CompiledPredictor(BaseCompiledPredictor):
    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict(self,
                np.ndarray[DTYPE_t, ndim=2, mode='c'] X,
                np.ndarray[DOUBLE_t, ndim=1, mode='c'] output):
        cdef Py_ssize_t num_samples = X.shape[0]
        for i in range(num_samples):
            output[i] = ((<DOUBLE_t (*)(DTYPE_t*)> self.func))(&X[i, 0])
        return output


cdef class CompiledClassifier(BaseCompiledPredictor):
    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict_proba(self,
                      np.ndarray[DTYPE_t, ndim=2, mode='c'] X,
                      np.ndarray[DOUBLE_t, ndim=2, mode='c'] output):
        cdef Py_ssize_t num_samples = X.shape[0]
        for i in range(num_samples):
            (<void (*)(DTYPE_t*, DOUBLE_t*)> self.func)(&X[i, 0], &output[i, 0])
        return output


cdef class CompiledClassifierQuasiFloat(BaseCompiledPredictor):
    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict_proba(self,
                      np.ndarray[INT_t, ndim=2, mode='c'] X,
                      np.ndarray[INT_t, ndim=2, mode='c'] output):
        cdef Py_ssize_t num_samples = X.shape[0]
        for i in range(num_samples):
            (<void (*)(INT_t*, INT_t*)> self.func)(&X[i, 0], &output[i, 0])
        return output
