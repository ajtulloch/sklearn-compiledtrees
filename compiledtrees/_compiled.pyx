cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
np.import_array()

cdef extern from "dlfcn.h":
  void* dlopen(const char*, long)
  void* dlsym(void*, const char* )
  char* dlerror()
  void dlclose(void* handle)
  cdef long RTLD_NOW

cdef class CompiledPredictor:
    def __cinit__(self, const char* filename, const char* symbol):
        cdef void* handle = dlopen(filename, RTLD_NOW)
        if handle == NULL:
            raise ValueError("Could not find compiled evaluation file")
        self.handle = handle
        cdef void* func = <DOUBLE_t (*)(float*, int) nogil> dlsym(self.handle, symbol)
        if func == NULL:
            raise ValueError("Could not find compiled evaluation function in file")
        self.func = func

    def __dealloc__(self):
        dlclose(self.handle)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict(self,
                float[:, :] X,
                double[:] output,
                int n_jobs):
        func = <double (*)(float*) nogil> self.func
        cdef Py_ssize_t num_samples = X.shape[0]
        cdef int i
        cdef int n_jobs_samples = 1

        if num_samples > 1:
            n_jobs_samples = n_jobs

        for i in prange(num_samples,
                        num_threads=n_jobs_samples,
                        nogil=True,
                        schedule="static"):
            output[i] = func(&X[i, 0])

        return output
