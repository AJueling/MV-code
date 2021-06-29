import dask
import numpy as np
import xesmf as xe
import mtspec
import xarray as xr

import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.weightstats import DescrStatsW

from grid import generate_lats_lons
from regions import boolean_mask
from filters import lowpass
from xr_DataArrays import dll_coords_names, xr_AREA


class AnalyzeDataArray(object):
    """ functions that work on 1D time series as well as on 3D fields
    > trends
    """
    
    def __init__(self):
        return
    
    
    def lintrend(self, x):
        """ linear trend timeseries of a timeseries """
        pf = np.polynomial.polynomial.polyfit(x.time, x, 1)
        lf = pf[1]*x.time + pf[0]
        return lf
    
    
    def quadtrend(self, x):
        """ quadratic trend timeseries of a timeseries """
        pf = np.polynomial.polynomial.polyfit(x.time, x, 2)
        lf = pf[2]*x.time**2 + pf[1]*x.time + pf[0]
        return lf
    
    
    def standard_deviation(self, x):
        """ calculated standard deviation
        can deal with NaNs in time dimension, e.g. in HadISST
        """    
        xstd = x.std(axis=0, skipna=True)
        xstd.name = 'standard_deviation'
        return xstd
    
    
    def autocorrelation(self, x):
        """ autocorrelation 
        can deal with NaNs in time dimension, e.g. in HadISST
        """    
        x, y = x[1:,:,:], x.shift(time=1)[1:,:,:]
        y = xr.where(np.isnan(x), np.nan, y)
        x = xr.where(np.isnan(y), np.nan, x)

        n     = xr.where(np.isnan(x), 0, 1).sum(axis=0)
        xmean = x.mean(axis=0, skipna=True)
        ymean = y.mean(axis=0, skipna=True)
        xstd  = x.std( axis=0, skipna=True)
        ystd  = y.std( axis=0, skipna=True)

        x -= xmean
        y -= ymean

        cov      = np.divide(np.nansum(np.multiply(x,y), axis=0), n)
        cor      = cov/(xstd*ystd)
        cor.name = 'autocorrelation'
        return cor
    
    
    def lag_linregress(self, x, y, dof_corr=1, lagx=0, lagy=0,\
                       autocorrelation=None, filterperiod=None, standardize=False):
        """  (lagged) linear regression of y on x (e.g. y=SST on x=index)

        adapted from: https://hrishichandanpurkar.blogspot.com/2017/09/vectorized-functions-for-correlation.html
        
        input:
        x               .. 1D time series
        y               .. time series (first dim must be time),
                           can be three dimensions (time,lat,lon)
        dof_corr        .. (0,1] correction factor for reduced degrees of freedom
        lagx, lagy      .. lags for x or y
        autocorrelation .. 2D map
        filterperiod    .. if time filter is applied to either time series or spatial data
        standardize     .. normalize time series to standard deviation 1
        
        output:
        ds .. xr Dataset containing covariance, 
                                    correlation, 
                                    regression slope and intercept (for y with respect to x),
                                    p-value, and
                                    standard error on regression 
                                    between the two datasets along their aligned time dimension.
        """
        np.warnings.filterwarnings('ignore')  # silencing numpy warning for NaNs
        assert dof_corr<=1 and dof_corr>0
        assert 'time' in x.coords and 'time' in y.coords
        if not np.all(x.time.values==y.time.values):
            x = x.assign_coords(time=y.time.values)
        
        # aligning data on time axis
        # x,y = xr.align(x,y)  # assert this earlier

        # lags
        if lagx!=0:  x = x.shift(time = -lagx).dropna(dim='time')
        if lagy!=0:  y = y.shift(time = -lagy).dropna(dim='time')
        if lagx!=0 or lagy!=0:  x,y = xr.align(x,y)
        
        # standardize
        if standardize==True:
            x -= x.mean(axis=0, skipna=True)
            x /= x.std(axis=0, skipna=True)

        # statistics
        n     = x.shape[0]                                   # data length
        xmean = x.mean(axis=0, skipna=True)                  # mean
        ymean = y.mean(axis=0, skipna=True)
        xstd  = x.std(axis=0, skipna=True)                   # standard deviation
        ystd  = y.std(axis=0, skipna=True)
        cov   = np.sum((x - xmean)*(y - ymean), axis=0)/(n)  # covariance
        cor   = cov/(xstd*ystd)                              # correlation  
        slope = cov/(xstd**2)                                # regression slope
        intercept = ymean - xmean*slope                      # intercept

        # statistics for significance test
        if autocorrelation is not None:
            dof_corr = autocorrelation.copy()
            dof_corr_filter = autocorrelation.copy()
            if filterperiod is None:
                dof_corr_filter[:,:] = 1/13
            else:
                assert type(filterperiod)==int
                dof_corr_filter[:,:] = 1/filterperiod
            dof_corr_auto = (1-autocorrelation)/(1+autocorrelation)
            dof_corr[:,:] = np.maximum(dof_corr_filter.values, dof_corr_auto.values)      

        # t-statistics
        tstats = cor*np.sqrt(dof_corr*n-2)/np.sqrt(1-cor**2)
        stderr = slope/tstats
        pval   = stats.t.sf(tstats, n*dof_corr-2)
        pval   = xr.DataArray(pval, dims=cor.dims, coords=cor.coords)

        # create xr Dataset
        cov.name       = 'cov'
        cor.name       = 'cor'
        slope.name     = 'slope'
        intercept.name = 'intercept'
        pval.name      = 'pval'
        stderr.name    = 'stderr'
        ds = xr.merge([cov, cor, slope, intercept, pval, stderr])
        ds.attrs['first_year'] = int(y.time[0]/365)
        ds.attrs['last_year']  = int(y.time[-1]/365)
        ds.attrs['lagx'] = lagx
        ds.attrs['lagy'] = lagy
        ds.attrs['standardized'] = int(standardize)

        return ds