""" ____ Regression Pattern Pipeline ____

input:
    run    .. simulations 'ctrl' & 'lpd', or observations 'had'
    idx    .. index 'AMO', 'SOM', 'TPI', 'PMV'

procedure:
    1. creating monthly and yearly SST files
    2. deseasonalizing & detrending of SST field
        - model data: pointwise quadratic fit
        - observations: two-factor detrending
    3. derive raw SST indices
    4. autocorrelation maps
    5. regression patterns

detrending methods:
    'had'          :  two-factor => anthro & natural CMIP5 MMM
    'ctrl' or 'lpd':  quadratic pointwise
"""

import os
import sys
import dask
import datetime
import numpy as np

from paths import path_prace
from ab_derivation_SST import DeriveSST as DS
from bc_analysis_fields import AnalyzeField as AF
from bd_analysis_indices import AnalyzeIndex as AI

def time_print():  return f'\n   {datetime.datetime.now().time()}'

def trex(fn, fct, kwargs={}):
    """  try: existence / except: create
    checks whether file exists already and if not, creates it
    input:
    fn      .. file name of file that function is supposed to create
    fct     .. function that will create data stored as `fn`
    kwargs  .. keyword arguments for function `fct`
    """
    assert type(fn)==str 
    assert hasattr(fct, '__call__')
    assert type(kwargs)==dict
    try:
        assert os.path.exists(fn), f'   now creating file:  {fn[46:]}'
        print(f'   file exists:  {fn[46:]}')
    except:
        fct(**kwargs)
    return

# 10x 149 year time segments and one 250 year segment
times_ctrl = list(zip(np.append(np.arange( 51, 150, 10), [ 51]),
                      np.append(np.arange(200, 299, 10), [301])))
# times_lpd  = list(zip(np.append(np.arange(154, 253, 10), [154]),
#                       np.append(np.arange(303, 402, 10), [404])))

# to compare lc1 and lpd
# times_lpd  = [(1050,1200)]#,(500,650)]
times_lpd  = [(154,1220)]#,(500,650)]
times_lc1  = [(11,261)]# [(50,200)]

times_had  = [None]

if __name__=='__main__':
    run = str(sys.argv[1]).lower()
    idx = str(sys.argv[2]).upper()
    assert run in ['ctrl', 'lpd', 'had', 'lc1']
    assert idx in ['AMO', 'SOM', 'TPI', 'PMV', 'SMV']

    
    # ==============================================================================================
    print(f'\nRegression Pattern Pipeline for {run} {idx}', time_print(), '\n')
    # ==============================================================================================

    if run=='ctrl':  times = times_ctrl
    elif run=='lpd':  times = times_lpd
    elif run=='lc1':  times = times_lc1
    elif run=='had':  times = times_had


    print('1. create monthly and yearly SST files', time_print())
    # ==============================================================================================
    # ==============================================================================================
    # also need to make sure that these files contain all the relevant years
    
    kwargs = dict(run=run)

    # yearly data must be present for all index calculations
    fn = f'{path_prace}/SST/SST_yrly_{run}.nc'
    fct = DS().generate_yrly_SST_files
    trex(fn=fn, fct=fct, kwargs=kwargs)

    # ocn rect of ctrl run
    if run=='ctrl':
        fn = f'{path_prace}/SST/SST_yrly_rect_ctrl.nc'
        trex(fn=fn, fct=fct, kwargs=kwargs)
        
    # monthly data
    fn = f'{path_prace}/SST/SST_monthly_{run}.nc'
    fct = DS().generate_monthly_SST_files
    trex(fn=fn, fct=fct, kwargs=kwargs)


    # ==============================================================================================
    print('\n2. deseasonalize and detrended SST field', time_print())
    # ==============================================================================================

    for time in times:
        # yrly data
        ts = AI().time_string(time)
        if run=='had':  # two-factor detrending
            dt = 'tfdt'
            fct = DS().SST_remove_forced_signal
            kwargs = dict(run='had', tavg='yrly', detrend_signal='GMST', time=None)
        elif run in ['ctrl', 'lpd','lc1']:  # quadratic detrending
            dt = 'pwdt'
            fct = DS().SST_pointwise_detrending
            kwargs = dict(run=run, tavg='yrly', degree=2, time=time)
        fn = f'{path_prace}/SST/SST_yrly_{dt}_{run}{ts}.nc'
        trex(fn=fn, fct=fct, kwargs=kwargs)

        # monthly data
        # deseasonalize
        fct = DS().deseasonalize_monthly_data
        fn = f'{path_prace}/SST/SST_monthly_ds_{run}{ts}.nc'
        kwargs = dict(run=run)
        if run in ['ctrl', 'lpd', 'lc1']: kwargs['time'] = time
        trex(fn=fn, fct=fct, kwargs=kwargs)

        # detrend
        fn = f'{path_prace}/SST/SST_monthly_ds_dt_{run}{ts}.nc'
        if run=='had':
            fct = DS().detrend_monthly_obs_two_factor
            kwargs = {}
        elif run in ['ctrl', 'lpd', 'lc1']:
            fct = DS().detrend_monthly_data_pointwise
            kwargs = dict(run=run, time=time)
        trex(fn=fn, fct=fct, kwargs=kwargs)

        # subselect Pacific data for subsequent EOF analysis
        if idx=='PMV':
            fct = DS().isolate_Pacific_SSTs
            for extent in ['38S', 'Eq', '20N']:
                kwargs = dict(run=run, extent=extent, time=time)
                fn = f'{path_prace}/SST/SST_monthly_ds_dt_{extent}_{run}{ts}.nc'
                trex(fn=fn, fct=fct, kwargs=kwargs)    

            
    # ==============================================================================================
    print('\n3. raw SST indices', time_print())
    # ==============================================================================================

    for time in times:
        ts = AI().time_string(time)
        
        # area average SST indices
        if idx in ['AMO', 'SOM', 'TPI', 'SMV']:
            fct = AI().derive_SST_avg_index
            kwargs = dict(run=run, index=idx, time=time)
            fn = f'{path_prace}/SST/{idx}_ds_dt_raw_{run}{ts}.nc'
            trex(fn=fn, fct=fct, kwargs=kwargs) 

        # EOF analysis
        if idx=='PMV':
            fct = AI().Pacific_EOF_analysis
            for extent in ['38S', 'Eq', '20N']:
                kwargs = dict(run=run, extent=extent, time=time)
                fn = f'{path_prace}/SST/PMV_EOF_{extent}_{run}{ts}.nc'
                trex(fn=fn, fct=fct, kwargs=kwargs)


    # ==============================================================================================
    print('\n4. autocorrelation fields', time_print())
    # ==============================================================================================

    fct = AI().derive_autocorrelation_maps

#     if idx in ['AMO', 'SOM']:  tavg = 'yrly'
#     if idx in ['TPI', 'PMV']:  
    tavg = 'monthly'

    for time in times:
        ts = AI().time_string(time)
        kwargs = dict(run=run, tavg=tavg, time=time)
        fn = f'{path_prace}/SST/SST_{tavg}_autocorrelation_{run}{ts}.nc'
        trex(fn=fn, fct=fct, kwargs=kwargs)


    # ==============================================================================================
    print('5. SST regression on indices', time_print())
    # ==============================================================================================

    fct = AI().make_regression_files

    for time in times:
        ts = AI().time_string(time)
        kwargs = dict(run=run, idx=idx, time=time)
        if idx=='PMV':
            for extent in ['38S', 'Eq', '20N']:
                fn = f'{path_prace}/SST/{idx}_{extent}_regr_{run}{ts}.nc'
                trex(fn=fn, fct=fct, kwargs=kwargs)
        else:
            fn = f'{path_prace}/SST/{idx}_regr_{run}{ts}.nc'
            trex(fn=fn, fct=fct, kwargs=kwargs)


    # ==============================================================================================
#     print('7. spectra', time_print())
    # ==============================================================================================




    # ==============================================================================================
    print('\nSuccess.', time_print(), '\n')
    # ==============================================================================================