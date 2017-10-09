from __future__ import print_function
import numpy as np
import cffi
import psutil
import threading
import os
import sys

'''
    Helper class to execute TSNE in separate thread.
'''


class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)


class MulticoreTSNE:
    '''
        Only
            - n_components
            - perplexity
            - angle
            - n_iter
        parameters are used.
        Other are left for compatibility with sklearn TSNE.
    '''

    def __init__(self,
                 n_components=2,
                 perplexity=30.0,
                 early_exaggeration=4.0,
                 learning_rate=1000.0,
                 n_iter=1000,
                 n_iter_without_progress=30,
                 min_grad_norm=1e-07,
                 metric='euclidean',
                 init='random',
                 verbose=0,
                 random_state=None,
                 method='barnes_hut',
                 angle=0.5,
                 n_jobs=1):
        self.n_components = n_components
        self.angle = angle
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = -1 if random_state is None else random_state

        assert n_components == 2, 'n_components should be 2'

        self.ffi = cffi.FFI()
        self.ffi.cdef(
            "void tsne_run_double(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, int _num_threads, int max_iter, int random_state);")

        path = os.path.dirname(os.path.realpath(__file__))
        self.C = self.ffi.dlopen(path + "/libtsne_multicore.so")

    def fit_transform(self, X):

        assert X.ndim == 2, 'X should be 2D array.'
        assert X.dtype == np.float64, 'Only double arrays are supported for now. Use .astype(np.float64) to convert.'
        
        if self.n_jobs == -1:
            self.n_jobs = psutil.cpu_count()

        assert self.n_jobs > 0, 'Wrong n_jobs parameter.'
        
        if (X.flags['C_CONTIGUOUS'] is False):
        	print('Converting input to contiguous array...')
        	X = np.ascontiguousarray(X)

        N, D = X.shape
        Y = np.zeros((N, self.n_components))

        cffi_X = self.ffi.cast('double*', X.ctypes.data)
        cffi_Y = self.ffi.cast('double*', Y.ctypes.data)

        t = FuncThread(self.C.tsne_run_double,
                       cffi_X, N, D,
                       cffi_Y, self.n_components,
                       self.perplexity, self.angle, self.n_jobs, self.n_iter, self.random_state)
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()

        return Y
