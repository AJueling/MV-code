import os
import sys
import numpy as np
import xarray as xr
import datetime

from dask.distributed import Client, LocalCluster

from grid import create_dz_mean
from paths import path_samoc
from regions import boolean_mask, regions_dict
from constants import cp_sw, rho_sw, km
from timeseries import IterateOutputCESM, ncfile_list
from xr_integrate import xr_int_global, xr_int_global_level, xr_int_vertical,\
                         xr_int_zonal, xr_int_zonal_level
from xr_DataArrays import xr_DZ, xr_AREA, xr_HTN, xr_LATS, dll_from_arb_da
from xr_regression import xr_linear_trend


def OHC_parallel(run, mask_nr=0):
    """ ocean heat content calculation """
    print('***************************************************')
    print('* should be run with a dask scheduler             *')
    print('`from dask.distributed import Client, LocalCluster`')
    print('`cluster = LocalCluster(n_workers=2)`')
    print('`client = Client(cluster)`')
    print('***************************************************')
    
    
    print(f'\n{datetime.datetime.now()}  start OHC calculation: run={run} mask_nr={mask_nr}')
    assert run in ['ctrl', 'rcp', 'lpd', 'lc1', 'lpi']
    assert type(mask_nr)==int
    assert mask_nr>=0 and mask_nr<13
    
#     file_out = f'{path_samoc}/OHC/OHC_test.nc'
    file_out = f'{path_samoc}/OHC/OHC_integrals_{regions_dict[mask_nr]}_{run}.nc'
    
    if run in ['ctrl', 'rcp']:
        domain = 'ocn'
    elif run in ['lpd', 'lc1', 'lpi']:
        domain = 'ocn_low'
        
    MASK = boolean_mask(domain, mask_nr)
        
    # geometry
    DZT  = xr_DZ(domain)
    AREA = xr_AREA(domain)
    HTN  = xr_HTN(domain) 
    LATS = xr_LATS(domain)
    print(f'{datetime.datetime.now()}  done with geometry')
    
    # multi-file 
    file_list = ncfile_list(domain='ocn', run=run, tavg='yrly', name='TEMP_PD')
    OHC = xr.open_mfdataset(paths=file_list,
                            combine='nested',
                            concat_dim='time',
                            decode_times=False,
#                             compat='minimal',
                            parallel=True).drop(['ULAT','ULONG']).TEMP*cp_sw*rho_sw
    if mask_nr!=0:
        OHC = OHC.where(MASK)
    print(f'{datetime.datetime.now()}  done loading data')
    
    for i, ds in enumerate([OHC, HTN, LATS]):
        print(i)
        print(ds)
        if 'TLAT' in ds.coords:
            round_tlatlon(ds)
    OHC_DZT = OHC*DZT
    print(f'{datetime.datetime.now()}  done OHC_DZT')

    # xr DataArrays
    da_g  = xr_int_global(da=OHC, AREA=AREA, DZ=DZT)
    da_gl = xr_int_global_level(da=OHC, AREA=AREA, DZ=DZT)
    da_v  = OHC_DZT.sum(dim='z_t') #xr_int_vertical(da=OHC, DZ=DZT)
    da_va = OHC_DZT.isel(z_t=slice(0, 9)).sum(dim='z_t')  # above 100 m
    da_vb = OHC_DZT.isel(z_t=slice(9,42)).sum(dim='z_t')  # below 100 m
    da_z  = xr_int_zonal(da=OHC, HTN=HTN, LATS=LATS, AREA=AREA, DZ=DZT)
    da_zl = xr_int_zonal_level(da=OHC, HTN=HTN, LATS=LATS, AREA=AREA, DZ=DZT)
    print(f'{datetime.datetime.now()}  done calculations')

    # xr Datasets
    ds_g  = da_g .to_dataset(name='OHC_global'             )
    ds_gl = da_gl.to_dataset(name='OHC_global_levels'      )
    ds_v  = da_v .to_dataset(name='OHC_vertical'           )
    ds_va = da_va.to_dataset(name='OHC_vertical_above_100m')
    ds_vb = da_vb.to_dataset(name='OHC_vertical_below_100m')
    ds_z  = da_z .to_dataset(name='OHC_zonal'              )
    ds_zl = da_zl.to_dataset(name='OHC_zonal_levels'       )
    print(f'{datetime.datetime.now()}  done dataset')

    print(f'output: {file_out}')

    ds_new = xr.merge([ds_g, ds_gl, ds_z, ds_zl, ds_v, ds_va, ds_vb])
    ds_new.to_netcdf(path=file_out, mode='w')
#     ds_new.close()
    print(f'{datetime.datetime.now()}  done\n')
    
    return ds_new


def round_tlatlon(das):
    """ rounds TLAT and TLONG to 2 decimals
    some files' coordinates differ in their last digit
    rounding them avoids problems in concatonating
    """
    das['TLAT']   = das['TLAT'].round(decimals=2)
    das['TLONG']  = das['TLONG'].round(decimals=2)
    return das


def t2da(da, t):
    """adds time dimension to xr DataArray, then sets time value to t"""
    da = da.expand_dims('time')
    da = da.assign_coords(time=[t])
    return da


def t2ds(da, name, t):
    """ 
    adds time dimension to xr DataArray, then sets time value to t,
    and then returns as array in xr dataset
    """
    da = t2da(da, t)
    ds = da.to_dataset(name=name)
    
    return ds


def trend_global_levels(ds):
    """ trend of OHC per level as [J/m/y] """
    dz_mean = create_dz_mean(domain='ocn')
    
    return xr_linear_trend(ds.OHC_global_levels/dz_mean)*365


def OHC_detrend_levels(da, detrend='lin'):
    """ """
    assert detrend in ['lin', 'quad']
    assert 'time' in da.coords
    
    dz_mean = create_dz_mean(domain='ocn')
    n = len(da.time)
    times = np.arange(n)
    levels_trend = np.zeros((n, km))
    
    if detrend=='lin':
        lfit_par  = np.zeros((2, km))
        for k in range(km):
            lfit_par[:,k] = np.polyfit(times, da[:,k]/dz_mean[k], 1)

        lin_fit  = np.zeros((n, km))
        for t in range(n):
            lin_fit[t,:]  = lfit_par[0,:]*t + lfit_par[1,:]

        levels_trend = ((da[:,:]/dz_mean - lin_fit ))

    elif detrend=='quad':
        qfit_par = np.zeros((3, km))
        for k in range(km):
            qfit_par[:,k] = np.polyfit(times, da[:,k]/dz_mean[k], 2)
        
        quad_fit = np.zeros((n, km))
        for t in range(n):
            quad_fit[t,:] = qfit_par[0,:]*t**2 + qfit_par[1,:]*t + qfit_par[2,:]
        
        levels_trend = ((da[:,:]/dz_mean - quad_fit))

    return levels_trend


def OHC_vert_diff_mean_rm(ds, run):

    assert run in ['ctrl', 'rcp']
    
    for suffix in ['']:#, '_above_100m','_below_100m']:
        assert f'OHC_vertical{suffix}' in ds
    
        OHC_vert_diff = ds[f'OHC_vertical{suffix}']-ds[f'OHC_vertical{suffix}'].shift(time=1)

        OHC_vert_diff_mean = OHC_vert_diff.mean(dim='time')  # 1 min
        OHC_vert_diff_rm   = OHC_vert_diff.rolling({'time':10}, center=True).mean(dim='time')

        OHC_vert_diff_mean.to_netcdf(f'{path_samoc}/OHC/OHC_vert{suffix}_diff_mean_{run}.nc' )
        OHC_vert_diff_rm  .to_netcdf(f'{path_samoc}/OHC/OHC_vert{suffix}_diff_rm_{run}.nc'  )

    return


def replace_OHC_year(ds, y):
    """replaces a year's OHC data with the average of the preceding and following years """
    y_idx = int(y - ds.time[0]/365)
    for field in list(ds.variables.keys())[-7:-3]:
        ds[field][dict(time=y_idx)] = ( ds[field][dict(time=y_idx-1)] +
                                        ds[field][dict(time=y_idx+1)] )/2
    return ds


if __name__=="__main__":
    run     = sys.argv[1]
    mask_nr = int(sys.argv[2])

    assert run in ['ctrl', 'rcp', 'lpd', 'lc1', 'lpi']
    assert mask_nr>=0 and mask_nr<13
    
    # 15 min for one lpi run
  
    if run in ['lpd', 'lpi', 'lc1']:     # low res
    
        cluster = LocalCluster(n_workers=4)
        client = Client(cluster)
        OHC_parallel(run=run, mask_nr=mask_nr)
        
    elif run in ['ctrl', 'rcp']:  # high res
        OHC_integrals(run=run, mask_nr=mask_nr)
    
    print(f'\n\nfinished at\n{datetime.datetime.now()}')