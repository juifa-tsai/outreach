## Common analysis packages
import numpy as NP
import scipy.linalg
from scipy import optimize as OPT
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


class ordinarykriging(object):

    def __init__(self, varigoram, debug=False):
        ## Local parameters ##
        self.variogram = varigoram
        self.debug = debug

    def fit(self, X, y):
        """
        [DESCRIPTION]
           Store the data with KDTree or original data structure, and fit the variogram model with set parameters.
        INPUT]
           X    : array-like, input data with features, same data size with y.
           y    : array-like, input data with interesting value, same data size with X
        OUTPUT]
           Null
        """
        X = NP.atleast_1d(X)
        y = NP.atleast_1d(y)
        self.shape = X.shape
        self.ndim = X.ndim
        self.y = y
        self.X = cKDTree(X, copy_data=True)


    def predict(self, X, n_neighbor=None, radius=NP.inf, use_nugget=False):
        '''
        [DESCRIPTION]
           Calculate and predict interesting value with looping for all data, the method take long time but save memory
           Obtain the linear argibra terms
            | V_ij -1 || w_i | = | V_k |
            |  1   0 ||  u  |   |  1  |
                a    *   w    =    b
              w = a^-1 * b
              y = w_i * Y
            V_ij : semi-variance matrix within n neighors
            V_k  : semi-variance vector between interesting point and n neighbors
            w_i  : weights for linear combination
            u    : lagrainge multiplier
            Y    : true value of neighbors
            y    : predicted value of interesting point
        [INPUT]
            X          : array-like, input data with same number of fearture in training data
            n_neighbor : int,        number of neighbor w.r.t input data, while distance < searching radius (5)
            radius     : float,      searching radius w.r.t input data (inf)
            use_nugget : bool,       if use nugget to be diagonal of kriging matrix for prediction calculation (False)
        [OUTPUT]
            1D/2D array(float)
            prediction : float, prdicted value via Kriging system y
            error      : float, error of predicted value (only if get_error = True)
            lambda     : float, lagrange multiplier u
        '''
        ## Make input to numpy.array
        X = NP.atleast_1d(X)
        if self.ndim == 1:
            if X.ndim < 2: X = X[:, NP.newaxis]
        else:
            X = NP.atleast_2d(X)

        ## Find the neighbors with K-tree object
        if n_neighbor is None:
            n_neighbor = self.shape[0]
        neighbor_dst, neighbor_idx = self.X.query(X, k=n_neighbor, p=2)

        ## Calculate prediction
        idxes = range(X.shape[0])
        out_lambda = NP.zeros(len(X))
        out_predict = NP.zeros(len(X))
        out_error = NP.zeros(len(X))
        for nd, ni, i in zip(neighbor_dst, neighbor_idx, idxes):
            ## select in searching radius
            ni = ni[nd < radius] # neighbors' index, while the distance < search radius
            nd = nd[nd < radius] # neighbors' distance, while the distance < search radius

            if len(ni) == 0:
                continue
            else:
                n = len(ni)

            ## Initialization
            a = NP.zeros((n+1,n+1))
            b = NP.ones((n+1,))

            ## Fill matrix a
            a[:n, :n] = self.variogram.predict(cdist(self.X.data[ni], self.X.data[ni], metric='euclidean'))
            a[:n, n] = 1
            a[n, :n] = 1

            ## Fill vector b
            b[:n] = self.variogram.predict(nd)

            ## set self-varinace is zero if not using Nugget
            if not use_nugget:
                ## modify a
                NP.fill_diagonal(a, 0.)
                ## modify b
                zero_index = NP.where(NP.absolute(nd) == 0)
                if len(zero_index) > 0:
                    b[zero_index[0]] = 0.

            ## Get weights
            #w = scipy.linalg.solve(a, b) # no constraint solution
            w = OPT.nnls(a, b)[0] # non-negative solution

            ## Fill results and prediction
            out_lambda[i] = w[n]
            out_predict[i] = w[:n].dot(self.y[ni])
            out_error[i] = NP.sqrt(w[:n].dot(b[:n]))

        return out_predict, out_error, out_lambda
