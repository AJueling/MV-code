# inspiration from https://gist.github.com/rabernat/bc4c6990eb20942246ce967e6c9c3dbe

import os
import dask
import numpy as np
import scipy as sp
import xarray as xr
import cftime
import datetime


# from numba import jit
from paths import path_samoc, path_results, CESM_filename, file_ex_ocn_ctrl
from regions import boolean_mask
from constants import imt, jmt, km


def datetime_to_float(x):
    """ converts datetime values to float for regression """
    time_ = x.time.values
    if type(x.time.values[0])==np.datetime64:
        x.time.values = np.array(x.time.values - np.datetime64('0001-01-01'), dtype=np.float)
    elif type(x.time.values[0])==cftime._cftime.Datetime360Day:
        x.time.values = cftime.date2num(x.time.values, units='months since 0001-01-01', calendar='360_day')
    return x, time_


def xr_lintrend(x):
    """ linear trend timeseries of a timeseries """
    y = x
#     print(x.time)
    time_to_float  = False
    if type(x.time.values[0]) in [np.datetime64, cftime._cftime.Datetime360Day]:
        x, time_ = datetime_to_float(x)
        time_to_float = True
    pf = np.polynomial.polynomial.polyfit(x.time.values, x.values, 1)
    lt = np.outer(x.time, pf[1]) + np.row_stack(([pf[0]]*len(x.time)))
    if np.ndim(x)==1:  lt = lt.flatten()  # remove dimension of length 1
    if time_to_float:  x.time.values = time_
    da = x.copy(data=lt)
#     return da.where(x.notnull(), np.array(len(x.time)*[np.nan])
    return da


def xr_quadtrend(x):
    """ quadratic trend timeseries of (a/an array of) time series
    returns xr.DataArray with the same dimensions as input """
    assert 'time' in list(x.coords) or 'time' in x.dims
    stacked, chunked = False, False
    for dim in x.dims:  # assign coords to dimensions initially without coords
        if dim not in x.coords:
            x = x.assign_coords(**{dim:x[dim].values})
    
    if len(x.dims)>2:  # stacking coordinates
        print('before stacking')
        c = list(x.dims)
        c.remove('time')
        X = x.stack(stacked_coord=c).copy()
        stacked = True
#         print('after stacking')
    else: X = x.copy()
    X = X.fillna(0)  # temporarily replaces nan's with 0's, but those are masked in the end again
    if stacked==True and len(X['stacked_coord'])%3600==0:
        print('before chunking')
        X = X.chunk(chunks=(len(X['time']),3600))
        chunked = True
#         print('after chunking')
#     if chunked:
#         def f(xx, yy):
#             return np.polynomial.polynomial.polyfit(xx, yy, 2)
#         pf = dask.array.apply_along_axis(f, 1, X.time, X).compute()        
#     else:
    pf = np.polynomial.polynomial.polyfit(X.time, X, 2)
    qt = np.outer(X.time**2,pf[2]) + np.outer(X.time,pf[1]) + np.row_stack(([pf[0]]*len(X.time)))  # np.ndarray
    if np.ndim(x)==1:  qt = qt.flatten()  # remove dimension of length 1
    if stacked:  # unstacking
        X.data = qt
        da = X.unstack()
    else:
        da = x.copy(data=qt)
    
    return da.where(x.notnull(), np.nan)


def xr_linear_trend(x):
    """ function to compute a linear trend coefficient of a timeseries """
    pf = np.polynomial.polynomial.polyfit(x.time, x, 1)
    return xr.DataArray(pf[1])


# ! slow, use xr_2D_trends instead
def xr_linear_trends_2D(da, dim_names, with_nans=False):
    """ calculate linear trend of 2D field in time

    ! slow, use xr_2D_trends instead
    
    input:
    da        .. 3D xr DataArray with (dim_names) dimensions
    dim_names .. tuple of 2 strings: e.g. lat, lon dimension names
    
    output:
    da_trend  .. slope of linear regression
    """
    if type(da.time.values[0]) in [np.datetime64, cftime._cftime.Datetime360Day]:
        x, time_ = datetime_to_float(da)
#         time_to_float = True
        
    def xr_linear_trend_with_nans(x):
        """ function to compute a linear trend coefficient of a timeseries """
        if np.isnan(x).any():
            x = x.dropna(dim='time')
            if x.size>1:
                pf = np.polynomial.polynomial.polyfit(x.time, x, 1)
            else:
                pf = np.array([np.nan, np.nan])
        else:
            pf = np.polynomial.polynomial.polyfit(x.time, x, 1)
        return xr.DataArray(pf[1])
    
    (dim1, dim2) = dim_names
    # stack lat and lon into a single dimension called allpoints
    stacked = da.stack(allpoints=[dim1, dim2])
    # apply the function over allpoints to calculate the trend at each point
    if with_nans==False:
        trend = stacked.groupby('allpoints').apply(xr_linear_trend)
        # unstack back to lat lon coordinates
        da_trend = trend.unstack('allpoints')
    if with_nans==True:
        trend = stacked.groupby('allpoints').apply(xr_linear_trend_with_nans)
        # unstack back to lat lon coordinates
        da_trend = trend.unstack('allpoints')
#     if time_to_float:  da_trend.time.values = time_
#     print(da_trend)
    if 'allpoints_level_0' in da_trend.coords.keys():
        da_trend = da_trend.rename({'allpoints_level_0':dim1, 'allpoints_level_1':dim2})
    return da_trend


def xr_2D_trends(xa):
    """ calculates linear slope of 2D xr.DataArray `xa` """
    (jm, im) = xa[0,:,:].shape
    xa_slope  = xa[0,:,:].copy()
    xa_interc = xa[0,:,:].copy()
    Nt = xa.values.shape[0]
    A = xa.values.reshape((Nt, im*jm))

    xa_lin = np.polyfit(xa.time, A, 1)  
    xa_slope.values = xa_lin[0,:].reshape((jm,im)) 
    return xa_slope



def ocn_field_regression(xa, run):
    """ calculates the trends of ocean fields
    
    input:
    xa      .. xr DataArray
    
    output:
    da_trend .. 2D xr DataArray with linear trends
    
    (takes about 40 seconds)
    """
    print(f'started at\n{datetime.datetime.now()}')
    
    assert type(xa)==xr.core.dataarray.DataArray
    assert len(xa.values.shape)==3
    # assert xa.values.shape[1:]==(jmt,imt)
    
    if run in ['ctrl', 'rcp']:
        MASK = boolean_mask('ocn'     , 0)
    elif run in ['lpi', 'lpd', 'lc1', 'lr1', 'lr2', 'ld']:
        MASK = boolean_mask('ocn_low' , 0)
    (jm, im) = MASK.shape
    xa   = xa.where(MASK>0).fillna(-9999)
    
    xa_slope  = xa[0,:,:].copy()
    xa_interc = xa[0,:,:].copy()
    Nt = xa.values.shape[0]
    A = xa.values.reshape((Nt, im*jm))
    
    xa_lin = np.polyfit(xa.time, A, 1)  
    xa_slope.values = xa_lin[0,:].reshape((jm,im))  # slope; [xa unit/time]
    xa_slope = xa_slope.where(MASK>0)
    
    xa_interc.values = xa_lin[1,:].reshape((jm,im))  # intercept; [xa unit]
    xa_interc = xa_interc.where(MASK>0)
    
    return xa_slope, xa_interc


def xr_regression_with_stats(xa, fn=None):
    """ linear regression wrt. time
    using scipy.stats.linregress to obtain statistics like p-values
    inspired by https://stackoverflow.com/questions/52094320
    42 sec for 100 years of yearly ocn_low SSTs
    50 min for ocn SSTs
    input:
    xa  (xr.Dataarray) with time dimension
    
    output:
    ds  (xs.Dataset)  contains linregress output
    """
    def new_linregress(x, y):
        # Wrapper around scipy linregress to use in apply_ufunc
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y)
        return np.array([slope, intercept, r_value, p_value, std_err])

    try:
        assert os.path.exists(fn)
        ds = xr.open_dataset(fn)
        if fn is not None:  print(f'file exists and is loaded:\n  {fn}')
    except:
        print(f'file does not yet exist; calculating it now:\n  {fn}')
        stats = xr.apply_ufunc(new_linregress, xa.time, xa,
                        input_core_dims=[['time'], ['time']],
                        output_core_dims=[["parameter"]],
                        vectorize=True,
                        dask="parallelized",
                        output_dtypes=['float64'],
                        output_sizes={"parameter": 5},
                        )
        ds = xr.Dataset()
        ds['slope'] = stats.isel(parameter=0)
        ds['intercept'] = stats.isel(parameter=1)
        ds['r_value'] = stats.isel(parameter=2)
        ds['p_value'] = stats.isel(parameter=3)
        ds['std_err'] = stats.isel(parameter=4)
        if fn is not None:  ds.to_netcdf(fn)
    return ds


def zonal_trend(ds):
    """ calculated tends for lat_bin field
    
    input:
    ds    .. xr Dataset of OHC fields
             ds.OHC_zonal [J m^-1]
    
    output:
    trend .. 1D xr DataArray in [J m^-1 year^-1]
    """    
    assert 'OHC_zonal' in ds
    assert 'time' in ds
    (min_lat, max_lat) = find_valid_domain(ds.OHC_zonal)
    trend = xr_linear_trend(ds.OHC_zonal[:,min_lat:max_lat+1])*365
    trend = trend.rename({'dim_0': 'TLAT_bins'})
    trend = trend.assign_coords(TLAT_bins=ds.TLAT_bins[min_lat:max_lat+1])
    return trend


def zonal_levels_trend(ds):
    """ calculated trend for depth-lat_bin field
    
    input:
    ds .. xr Dataset
          ds.OHC_zonal_levels [J m^-2]
    
    output:
    trend .. xr DataArray containing trends of OHC [J m^-2 year^-1]
    """
    
    assert 'OHC_zonal_levels' in ds
    
    dn = ('TLAT_bins', 'z_t')
    (min_lat, max_lat, max_depth) = find_valid_domain(ds.OHC_zonal_levels)
    print(min_lat, max_lat)
    trend = xr_linear_trends_2D(ds.OHC_zonal_levels[:,:max_depth+1,min_lat:max_lat+1], dim_names=dn)*365  #  to [yr^-1]
    return trend


def find_valid_domain(da):
    """ finds first and last index of non-NaN values for zonal integrals 
    
    input:
    da                       .. xr DataArray
    
    output:
    min/max_lat (/max_depth) .. indices of first/last non-NaN entry in da 
    """
        
    if da.ndim==2:  # lat_bin field
        min_lat = 0 
        while np.isnan(da[0,min_lat]).item():
            min_lat += 1

        max_lat = len(da[0,:])-1
        while np.isnan(da[0,max_lat]).item():
            max_lat -= 1
        print(min_lat, max_lat)
        return (min_lat, max_lat)
    
    if da.ndim==3:  # depth-lat_bin field
        min_lat = 0 
        while np.isnan(da[0,0,min_lat]).item():
            min_lat += 1

        max_lat = len(da[0,0,:])-1
        while np.isnan(da[0,0,max_lat]).item():
            max_lat -= 1
        
        
        max_depth = 41
        while np.isnan(da[0,max_depth, min_lat]).item():
            max_depth -= 1
        return (min_lat, max_lat, max_depth)    
    
    
def lag_linregress_3D(x, y, dof_corr=1, lagx=0, lagy=0):
    """
    adapted from: https://hrishichandanpurkar.blogspot.com/2017/09/vectorized-functions-for-correlation.html
    Input: Two xr.Datarrays of any dimensions with the first dim being time. 
    Thus the input data could be a 1D time series, or for example, have three dimensions (time,lat,lon). 
    Datasets can be provied in any order, but note that the regression slope and intercept will be calculated
    for y with respect to x.
    Output: xr Dataset containing covariance, correlation, regression slope and intercept, p-value, and
    standard error on regression between the two datasets along their aligned time dimension.  
    Lag values can be assigned to either of the data, with lagx shifting x, and lagy shifting y, with the specified lag amount.
    dof_corr .. (0,1] correction factor for reduced degrees of freedom
    """ 
    assert dof_corr<=1 and dof_corr>0
    #1. Ensure that the data are properly aligned to each other.
    x,y = xr.align(x,y)
    
    #2. Add lag information if any, and shift the data accordingly
    if lagx!=0:
        #If x lags y by 1, x must be shifted 1 step backwards. 
        #But as the 'zero-th' value is nonexistant, xr assigns it as invalid (nan). Hence it needs to be dropped
        x   = x.shift(time = -lagx).dropna(dim='time')
        #Next important step is to re-align the two datasets so that y adjusts to the changed coordinates of x
        x,y = xr.align(x,y)

    if lagy!=0:
        y   = y.shift(time = -lagy).dropna(dim='time')
        x,y = xr.align(x,y)
 
    #3. Compute data length, mean and standard deviation along time axis for further use: 
    n     = x.shape[0]
    xmean = x.mean(axis=0)
    ymean = y.mean(axis=0)
    xstd  = x.std(axis=0)
    ystd  = y.std(axis=0)
    
    #4. Compute covariance along time axis
    cov   =  np.sum((x - xmean)*(y - ymean), axis=0)/(n)
    
    #5. Compute correlation along time axis
    cor   = cov/(xstd*ystd)
    
    #6. Compute regression slope and intercept:
    slope     = cov/(xstd**2)
    intercept = ymean - xmean*slope  
    
    #7. Compute P-value and standard error
    #Compute t-statistics
    tstats = cor*np.sqrt(n*dof_corr-2)/np.sqrt(1-cor**2)
    stderr = slope/tstats
    
    pval   = sp.stats.t.sf(tstats, n-2)  # *2 for t-tailed test
    pval   = xr.DataArray(pval, dims=cor.dims, coords=cor.coords)

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
    
    return ds