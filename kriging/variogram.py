import os, sys
import pandas as PD
import numpy as NP
import matplotlib.pyplot as PLT
from tqdm import tqdm as TQDM
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares

class variogram(object):
    """
    Variogram estimator 
    """
    def __init__(self, lag, lag_min=0, lag_max=NP.inf, model=None):
        """
        Initialization model parameters
        [Input]
            lag     : float, width of bin of binned variogram PDF
            lag_min : float, minimum of bin edge of binned variogram PDF (default 0)
            lag_max : float, maximum of bin edge of binned variogram PDF (default 0)
            model   : callable, PDF of variogram for fitting
        """
        ## Local parameters ##
        self.model = model
        self.lag = lag
        self.lag_min = lag_min
        self.lag_max = lag_max

        ## Local variables ##
        # lag edges
        self.lags_edge = []
        # sum of distance in lag range
        self.lags_d = []
        # number of pair in lag range 
        self.lags_n = []
        # sum of semi-variance in lag range
        self.lags_v = []
        # measurements of semi-variance in lag range
        self.lags_e = []
        # number of fills in lag range
        self.nfills = []
        # fitted parameters of variogram model
        self.fitted_parms = None


    ## Predict function ##
    def predict(self, X):
        """
        Predict semi-variance with input pair distance
        [input] 
            X: array-like, pair distance
        [output] 
            array-like, semi-varaince
        """
        return self.model(NP.atleast_1d(X), self.fitted_parms)


    ## Fitting functions ##
    def fit(self, X, y, x0=None, bounds=None, fit_range=None, loss='soft_l1', batch_size=100, metric='euclidean'):
        """
        Fit binned semi-variogram with input model (overwrite binned semi-variogram and fitting results)
        [Input]
            X          : array-like, input features
            y          : array-like, target/interested values
            x0         : array-like (3,), initial parameters (sill, range, nugget). 
                                If None, the default will be provided by statistical estimation. (default None)
            bounds     : array-like (2,3), boundary of fitting parameter ((lower_bound), (upper_bound)). 
                                Each bound includes (sill, range, nugget). If None, the default will be provided 
                                by statistical estimation. (default None)
            fit_range  : array-like (2,), minmum and maximum lag for fitting (default None)
            loss       : str, loss function for fitting (defaut 'soft_l1')
            batch_size : int, number of data in each batch (default 100)
            metric     : str, distance algorithm for measuring pair. See options in scipy.spatial.distance.cdist (default 'euclidean')
        """
        self.lags_edge = []
        self.lags_d = []
        self.lags_n = []
        self.lags_v = []
        self.lags_e = []
        self.nfills = []
        self.fitted_parms = None
        self.partial_fit(X, y, x0, bounds, fit_range, loss, batch_size, metric)

    def partial_fit(self, X, y, x0=None, bounds=None, fit_range=None, loss='soft_l1', batch_size=100, metric='euclidean'):
        """
        Fit binned semi-variogram with input model (overwrite fitting results, but append binned semi-variogram from input)
        [Input]
            X          : array-like, input features
            y          : array-like, target/interested values
            x0         : array-like (3,), initial parameters (sill, range, nugget). 
                                If None, the default will be provided by statistical estimation. (default None)
            bounds     : array-like (2,3), boundary of fitting parameter ((lower_bound), (upper_bound)). 
                                Each bound includes (sill, range, nugget). If None, the default will be provided 
                                by statistical estimation. (default None)
            fit_range  : array-like (2,), minmum and maximum lag for fitting (default None)
            loss       : str, loss function for fitting (defaut 'soft_l1')
            batch_size : int, number of data in each batch (default 100)
            metric     : str, distance algorithm for measuring pair. See options in scipy.spatial.distance.cdist (default 'euclidean')
        """
        X = NP.atleast_1d(X)
        y = NP.atleast_1d(y)
        ## calculate binned semi-variogram
        self.calculate(X, y, batch_size, metric)
        ## fit the binned semi-variogram
        self.update_fit(x0, bounds, fit_range, loss)


    def calculate(self, X, y, batch_size=100, metric='euclidean'):
        """
        Calculate semi-variogram and make binned lags
        [Input]
            X          : array-like, input features
            y          : array-like, target/interested values
            batch_size : int, number of data in each batch (default 100)
            metric     : str, distance algorithm for measuring pair. See options in scipy.spatial.distance.cdist (default euclidean)
        """
        X = NP.atleast_1d(X)
        y = NP.atleast_1d(y)
        n = y.shape[0]

        ## Make X, y two 2-D ##
        if X.ndim < 2: X = X[:, NP.newaxis]
        if y.ndim < 2: y = y[:, NP.newaxis]

        ## create batch index ##
        n_batch = int(n/batch_size)
        n_batch = n_batch + 1 if n_batch < n/batch_size else n_batch

        ## calculate binned semi-variogram ##
        for i in range(n_batch):
            i = i*batch_size
            j = i + batch_size

            ## calucalte distance (lag) and semivariance of pair
            d = NP.triu(cdist(X[i:j, :], X[i:, :], metric=metric))
            v = NP.triu(cdist(y[i:j, :], y[i:, :], metric='sqeuclidean'))

            ## select distance in lag's range
            selection = (d > self.lag_min) & (d <= self.lag_max)
            d = d[selection]
            v = v[selection]

            ## distribute semivariance and lags to lag's range (bins) 
            nlags = int(max(d)/self.lag)
            nlags = nlags + 1 if nlags != 1 else nlags
            for b in range(nlags):
                edge = b*self.lag
                msk = (d >= edge) & (d < edge + self.lag)
                if edge in self.lags_edge:
                    self.nfills[b] += 1
                    self.lags_d[b] += sum(d[msk])
                    self.lags_n[b] += len(d[msk])
                    self.lags_v[b] += sum(v[msk])
                    if len(d[msk]) > 0:
                        self.lags_e[b].extend([sum(v[msk])/len(d[msk])])
                else:
                    # first fill
                    self.nfills.append(1)
                    self.lags_edge.append(edge)
                    self.lags_d.append(sum(d[msk]))
                    self.lags_n.append(len(d[msk]))
                    self.lags_v.append(sum(v[msk]))
                    self.lags_e.append([sum(v[msk])/len(d[msk])] if len(d[msk]) > 0 else [])


    def update_fit(self, x0=None, bounds=None, fit_range=None, loss='soft_l1'):
        """
        Fit binned semi-variogram with input model (overwrite fitting results)
        [Input]
            x0         : array-like (3,), initial parameters (sill, range, nugget). 
                                If None, the default will be provided by statistical estimation. (default None)
            bounds     : array-like (2,3), boundary of fitting parameter ((lower_bound), (upper_bound)). 
                                Each bound includes (sill, range, nugget). If None, the default will be provided 
                                by statistical estimation. (default None)
            fit_range  : array-like (2,), minmum and maximum lag for fitting (default None)
            loss       : str, loss function for fitting (defaut 'soft_l1')
        """
        if self.model is None:
            print("Error : no model found")
            return self

        ## Fit with least square ##
        self.fit_range = fit_range
        ## parameter initialization
        self.initial_fitparams(x0, bounds)
        ## fit with least squares method
        self.results = least_squares(fun=self._cost, x0=self.x0, bounds=self.bounds, loss=loss)
        ## obtain fitted results
        self.fitted_parms = self.results.x
        self.fitted_semivariograms = self.model(NP.atleast_1d(self.lags), self.fitted_parms)

        ## fitting metric ##
        self.chi2 = sum(((self.semivariograms[self.msk] - self.fitted_semivariograms[self.msk])/self.error[self.msk])**2)
        self.chi2ndf = self.chi2/(len(self.semivariograms[self.msk]) - 3)
        self.r2 = 1 - sum((self.fitted_semivariograms[self.msk] - self.semivariograms[self.msk])**2)/sum((self.semivariograms[self.msk] - self.semivariograms[self.msk].mean())**2)


    def initial_fitparams(self, x0=None, bounds=None):
        """
        Initializae fitting parameters for variogram models
        [Input]
            x0     : array-like (3,), initial parameters (sill, range, nugget). 
                     If None, the default will be provided by statistical estimation. (default None)
            bounds : array-like (2,3), boundary of fitting parameter ((lower_bound), (upper_bound)). 
                     Each bound includes (sill, range, nugget). If None, the default will be provided 
                     by statistical estimation. (default None)
        """
        if x0 is None:
            self.x0 = [NP.amax(self.semivariograms)-NP.amin(self.semivariograms),
                       0.25*self.lags[self.semivariograms == NP.amax(self.semivariograms)][0],
                       NP.amin(self.semivariograms)]
        else:
            self.x0 = x0

        if bounds is None:
            self.bounds = ([0., 0., 0.],
                           [10.*NP.amax(self.semivariograms), NP.amax(self.lags), NP.amax(self.semivariograms)])
        else:
            self.bounds = bounds


    def _cost(self, params):
        """
        Cost function for least square method
        [Input]
            params : array-like, input fitted parameters from least_square function
        """
        ## make fitting range data mask ## 
        if self.fit_range is not None:
            self.msk = (NP.atleast_1d(self.lags_edge) >= self.fit_range[0]) & (NP.atleast_1d(self.lags_edge) <= self.fit_range[1])
        else:
            self.msk = NP.full(NP.atleast_1d(self.lags_edge).shape, True)
        ## Get residual ##
        cost = self.model(NP.atleast_1d(self.lags)[self.msk], params) - NP.atleast_1d(self.semivariograms)[self.msk]
        return cost


    ## Display functions ##
    def summary(self):
        """
        Summary fitted results
        """
        print("Sill %f"%(self.sill))
        print("Range %f"%(self.range))
        print("Nugget %f"%(self.nugget))
        if self.fitted_parms is not None:
            print("Fit chi2/ndf %f"%(self.chi2ndf))
            print("Fit r2 %f"%(self.r2))

    def plot(self, xmin=None, xmax=None, error=False, errorplot=False, to=None, title='', transparent=True, show=True):
        """
        Displays variogram model with the actual binned data.
        [Input]
            xmin        : float, minimum of x-axis (lags) (default None)
            xmin        : float, maximum of x-axis (lags) (default None)
            error       : bool, show data error bar (default False)
            errorplot   : bool, show relative error ratio plot (default False)
            to          : str, path for saving plot to .png file (default None)
            title       : str, title of plot (default '')
            transparent : bool, background transparent (default True)
            show        : bool, show on screen (default True)
        """
        if errorplot:
            nplot = 3
        else:
            nplot = 2
        fig, axes = PLT.subplots(nplot, 1, figsize=PLT.gcf().get_size_inches())
        if xmin is None: xmin = min(self.lags_edge)
        if xmax is None: xmax = max(self.lags_edge)
        
        ## data point ##
        axes[0].set_xlim([xmin, xmax])
        axes[0].set_ylim([0, max(self.semivariograms)*1.1])
        axes[0].plot(self.lags, self.semivariograms, 'k.', markersize=10)
        if error:
            axes[0].set_ylim([0, max(self.semivariograms+self.error)*1.1])
            axes[0].errorbar(self.lags, self.semivariograms, yerr=self.error, color='k', fmt='none')
        ## fitting line ##
        if self.fitted_parms is not None:
            axes[0].plot(self.lags, self.model(self.lags, self.fitted_parms), 'r-')
            if self.fit_range is not None:
                axes[0].plot([self.fit_range[0], self.fit_range[0]], axes[0].get_ylim(), 'r--' )
                axes[0].plot([self.fit_range[1], self.fit_range[1]], axes[0].get_ylim(), 'r--' )

        ## statistics plot ##
        axes[1].set_xlim([xmin, xmax])
        axes[1].set_ylim([0, max(self.lags_n)*1.1])
        axes[1].plot(self.lags, self.lags_n, 'k-')

        ## relative error ratio plot ##
        if errorplot:
            axes[2].set_xlim([xmin, xmax])
            axes[2].set_ylim([0, 1])
            axes[2].plot(self.lags, self.error/self.semivariograms, 'k-')
        
        ## Minor adjustion ##
        PLT.title(title)

        ## show on screen ##
        if show:
            PLT.show()

        ## save to .png ##
        if to is not None:
            print('>> [INFO] Saving plot to %s'% to)
            PLT.savefig(to, transparent=transparent)


    ## Callable variables ##
    @property
    def sill(self):
        if self.fitted_parms is None:
            return
        else:
            return self.fitted_parms[0] + self.fitted_parms[2]

    @property
    def range(self):
        if self.fitted_parms is None:
            return
        else:
            return self.fitted_parms[1]

    @property
    def nugget(self):
        if self.fitted_parms is None:
            return
        else:
            return self.fitted_parms[2]

    @property
    def semivariograms(self):
        return NP.divide(self.lags_v, self.lags_n, out=NP.zeros_like(self.lags_v), where=NP.array(self.lags_n)!=0)

    @property
    def lags(self):
        lags = NP.divide(self.lags_d, self.lags_n, out=NP.zeros_like(self.lags_d), where=NP.array(self.lags_n)!=0)
        msk = (lags == 0)
        lags[msk] = NP.array(self.lags_edge)[msk]
        return lags

    @property
    def error(self):
        lags_e = []
        for b in range(len(self.lags_edge)):
            if len(self.lags_e[b]) > 0:
                lags_e.append(NP.std(self.lags_e[b]))
            else:
                lags_e.append(0)
        return NP.array(lags_e)

