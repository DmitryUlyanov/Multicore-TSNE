from __future__ import print_function
from glob import glob
import threading
import os
import sys

import numpy as np
import cffi

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
        self.init = init
        self.embedding_ = None
        self.verbose = int(verbose)

        assert n_components == 2, 'n_components should be 2'

        assert isinstance(init, np.ndarray) or init == 'random', "init must be 'random' or array"
        if isinstance(init, np.ndarray):
            assert init.ndim == 2, "init array must be 2D"
            assert init.shape[1] == n_components, "init array must be of shape (n_instances, n_components)"
            self.init = np.ascontiguousarray(init, float)

        self.ffi = cffi.FFI()
        self.ffi.cdef(
            "void tsne_run_double(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, int num_threads, int max_iter, int random_state, bool init_from_Y, int verbose);")

        path = os.path.dirname(os.path.realpath(__file__))
        try:
            sofile = (glob(os.path.join(path, 'libtsne*.so')) +
                      glob(os.path.join(path, 'libtsne*.dll')))[0]
            self.C = self.ffi.dlopen(os.path.join(path, sofile))
        except (IndexError, OSError):
            raise RuntimeError('Cannot find/open tsne_multicore shared library')

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, _y=None):

        assert X.ndim == 2, 'X should be 2D array.'

        # X may be modified, make a copy
        X = np.array(X, dtype=float, order='C', copy=True)

        N, D = X.shape
        init_from_Y = isinstance(self.init, np.ndarray)
        if init_from_Y:
            Y = self.init.copy('C')
            assert X.shape[0] == Y.shape[0], "n_instances in init array and X must match"
        else:
            Y = np.zeros((N, self.n_components))

        cffi_X = self.ffi.cast('double*', X.ctypes.data)
        cffi_Y = self.ffi.cast('double*', Y.ctypes.data)

        t = FuncThread(self.C.tsne_run_double,
                       cffi_X, N, D,
                       cffi_Y, self.n_components,
                       self.perplexity, self.angle, self.n_jobs, self.n_iter, self.random_state,
                       init_from_Y, self.verbose)
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()

        self.embedding_ = Y

        return Y
