import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import statsmodels.api as sm

from tqdm import tqdm

from OHC import t2ds
from paths import CESM_filename, path_prace, path_data, file_ex_ocn_ctrl, file_ex_ocn_lpd, file_HadISST
from regions import boolean_mask
from timeseries import IterateOutputCESM
from xr_DataArrays import depth_lat_lon_names, xr_DZ, xr_DXU, xr_AREA
from xr_regression import xr_lintrend, xr_quadtrend
from ba_analysis_dataarrays import AnalyzeDataArray as ADA




class DeriveSST(object):
    """ generate SST fields """
    def __init__(self):
        return

    
    def generate_yrly_SST_files(self, run):
        """ generate the SST data files from TEMP_PD yearly averaged files """
        # ca. 4:30 min for ctrl/rcp, 1:25 for lpi
        # stacking files into one xr DataArray object
        
        if run in ['ctrl', 'rcp', 'lpd', 'lc1', 'lpi', 'lr1']:
            for i, (y,m,s) in enumerate(IterateOutputCESM('ocn', run, 'yrly', name='TEMP_PD')):
                print(y)
                da = xr.open_dataset(s, decode_times=False).TEMP[0,:,:]
                da = da.drop(['z_t', 'ULONG', 'ULAT'])
                da_time = int(da.time.item())
                if run=='ctrl':
                    # years 5-50 have different TLAT/TLON grids
                    # somehow the non-computed boxes changed (in the continents)
                    if i==0:
                        TLAT = da['TLAT'].round(decimals=2)
                        TLONG = da['TLONG'].round(decimals=2)
                    da['TLAT'] = TLAT
                    da['TLONG'] = TLONG
                else:
                    da['TLAT' ] = da['TLAT' ].round(decimals=2)
                    da['TLONG'] = da['TLONG'].round(decimals=2)
                del da.encoding["contiguous"]
                ds = t2ds(da=da, name='SST', t=da_time)
                ds.to_netcdf(path=f'{path_prace}/SST/SST_yrly_{run}_{y:04}.nc', mode='w')
            if run=='lpd':
                combined = xr.open_mfdataset(f'{path_prace}/SST/SST_yrly_lpd_*.nc',
                             combine='nested', concat_dim='time', 
                             data_vars=['SST'],
                             coords='minimal', drop_variables=['TLAT','TLONG'])
            else:
                combined = xr.open_mfdataset(f'{path_prace}/SST/SST_yrly_{run}_*.nc',
                                             concat_dim='time', autoclose=True, coords='minimal')
            combined.to_netcdf(f'{path_prace}/SST/SST_yrly_{run}.nc')
            self.remove_superfluous_files(f'{path_prace}/SST/SST_yrly_{run}_*.nc')
            
            if run=='ctrl':  # create also ocn_rect file
                fn = f'{path_prace}/SST/SST_monthly_ctrl.nc'
                fn_out = f'{path_prace}/SST/SST_yrly_rect_ctrl.nc'
                monthly_ctrl = xr.open_dataarray(fn, decode_times=False)
                t_bins = np.arange(0,len(monthly_ctrl)+1,12)/12
                t_coords = np.array(monthly_ctrl.time[0::12].values, dtype=int)
                yrly_ctrl = monthly_ctrl.groupby_bins('time', t_bins, right=False).mean(dim='time')
                yrly_ctrl = yrly_ctrl.assign_coords(time_bins=t_coords).rename({'time_bins':'time'})
                yrly_ctrl.to_netcdf(fn_out)
            
        elif run=='had':
            ds2 = xr.open_dataset(file_HadISST)
            ds2 = ds2.where(ds2['sst'] != -1000.)
            ds2 = ds2.sst.where(np.isnan(ds2.sst)==False, -1.8)
            ds2 = ds2.groupby('time.year').mean('time')
            ds2 = ds2.rename({'year':'time'}).isel(time=slice(0,-1))
            ds2.coords['time'] = (ds2.coords['time']-1870)*365
            ds2.to_netcdf(f'{path_prace}/SST/SST_yrly_had.nc')
            
        return

    
    def generate_yrly_global_mean_SST(self, run):
        """ calcaultes the global mean sea surface temperature
        ca. 37 sec for ctrl """
        assert run in ['ctrl', 'lpd']
        
        da = xr.open_dataarray(f'{path_prace}/SST/SST_yrly_{run}.nc', decode_times=False)
        if run=='ctrl':
            AREA = xr_AREA(domain='ocn')
            REGION_MASK = xr.open_dataset(file_ex_ocn_ctrl, decode_times=False).REGION_MASK
        elif run=='lpd':
            AREA = xr_AREA(domain='ocn_low')
            REGION_MASK = xr.open_dataset(file_ex_ocn_lpd, decode_times=False).REGION_MASK
        AREA_total = AREA.where(REGION_MASK>0).sum(dim=['nlat', 'nlon'], skipna=True)
        print(AREA_total)
        da_new = (da*AREA).where(REGION_MASK>0).sum(dim=['nlat', 'nlon'], skipna=True)/AREA_total
        da_new.to_netcdf(f'{path_prace}/SST/GMSST_yrly_{run}.nc')
        return
            
            
    def generate_monthly_SST_files(self, run, time=None):
        """ concatonate monthly files, ocn_rect for high res runs"""
        # 8 mins for 200 years of ctrl
        if run in ['ctrl', 'rcp', 'hq']:                 domain = 'ocn_rect'
        elif run in ['lpd', 'lpi', 'lc1', 'lr1', 'ld']:  domain = 'ocn_low'
            
        for y,m,s in IterateOutputCESM(domain=domain, tavg='monthly', run=run):
            fn = f'{path_prace}/SST/SST_monthly_{run}_y{y:04}.nc'
            if os.path.exists(fn):  continue
            if time==None:
                pass
            elif y<time[0] or y>=time[1]:
                continue
            if m==1: print(y)
            if run in ['ctrl', 'rcp']:
                xa = xr.open_dataset(s, decode_times=False).TEMP[0,:,:]
            if run in ['lpd', 'lpi', 'lc1', 'lr1', 'ld']:
                xa = xr.open_dataset(s, decode_times=False).TEMP[0,0,:,:]
            if m==1:   xa_out = xa.copy()    
            else:      xa_out = xr.concat([xa_out, xa], dim='time')
            drops = ['ULAT','ULONG','TLAT','TLONG']
            if m==12:  
                if type(xa_out)==xr.core.dataarray.Dataset:
                    if 'z_t' in xa_out.variables:  drops.append('z_t')
                elif 'z_t' in xa_out.coords:  drops.append('z_t')
                xa_out.drop(drops).to_netcdf(fn)
            # this also means only full years are written out
                        
        combined = xr.open_mfdataset(f'{path_prace}/SST/SST_monthly_{run}_y*.nc',
                                     combine='nested', concat_dim='time', decode_times=False)
        if run=='ctrl':
            time_ctrl = np.arange(1+1/24, 301, 1/12)
            combined = combined.assign_coords(time=time_ctrl)
            # something is wrong in Febraury 99, so I average 
            n = np.where(np.isclose(time_ctrl, 99.125))[0][0]
            print(f'month 99/2 is averaged; n={n}')
            combined[n] = (combined[n-1]+combined[n+1])/2
        combined = combined.to_dataarray()
        
        if time==None:
            fn_out = f'{path_prace}/SST/SST_monthly_{run}.nc'
        else:
            fn_out = f'{path_prace}/SST/SST_monthly_{run}_{time[0]}_{time[1]}.nc'
        combined.to_netcdf(fn_out)
        combined.close()
        self.remove_superfluous_files(f'{path_prace}/SST/SST_monthly_{run}_y*.nc')
        return
    
    
    def deseasonalize_monthly_data(self, run, time=None):
        """ removes the average difference of a month to the yearly average"""
        assert run in ['ctrl', 'lpd', 'had', 'lc1', 'cobe', 'ersst']

        if time==None:
            monthly = xr.open_dataarray(f'{path_prace}/SST/SST_monthly_{run}.nc', decode_times=False)
        else:
            monthly = xr.open_dataarray(f'{path_prace}/SST/SST_monthly_{run}_{time[0]}_{time[1]}.nc', decode_times=False)
            
        if run=='ctrl':
            yrly = xr.open_dataarray(f'{path_prace}/SST/SST_yrly_rect_ctrl.nc', decode_times=False)
        else:
            yrly = xr.open_dataarray(f'{path_prace}/SST/SST_yrly_{run}.nc', decode_times=False)

        if time==None:
            fn_out = f'{path_prace}/SST/SST_monthly_ds_{run}.nc'
        else:
            fn_out  = f'{path_prace}/SST/SST_monthly_ds_{run}_{time[0]}_{time[1]}.nc'
            yrly    = self.select_time(data=yrly, time=time)
            monthly = self.select_time(data=monthly, time=time)
        
        assert len(monthly)/len(yrly) == 12.0,\
               f'monthly len: {len(monthly)}; yearly len: {len(yrly)}'
        
        print(monthly)
        print(yrly)

        temp = monthly.copy()
        for j in tqdm(range(12)):
            m = monthly.isel(time=slice(j,len(monthly)+1,12))
            if j==0:
                print(m)
            #     print(yrly.assign_coords(time=m.time))
            # temp[j::12] -= (m-yrly.assign_coords(time=m.time)).mean(dim='time')
            temp[j::12] -= (m-yrly.assign_coords(time=('time', m.time))).mean(dim='time')
        temp.to_netcdf(fn_out)
        return
    
    
    def detrend_monthly_data_pointwise(self, run, time):
        """ quadratically detrend monthly fields """
        assert run in ['ctrl', 'lpd', 'lc1']
        
        da = xr.open_dataarray(f'{path_prace}/SST/SST_monthly_ds_{run}_{time[0]}_{time[1]}.nc',
                               decode_times=False)
        da = self.select_time(data=da, time=time)
        fn_out = f'{path_prace}/SST/SST_monthly_ds_dt_{run}_{time[0]}_{time[1]}.nc'
        (da-xr_quadtrend(da)).to_netcdf(fn_out)
        return
    
        
    def detrend_monthly_obs_two_factor(self, run):
        """ remove linear combination of anthropogenic and natural forcing signal from CMIP5 MMM """
        assert run in ['had', 'ersst', 'cobe']
            
        monthly_ds = xr.open_dataarray(f'{path_prace}/SST/SST_monthly_ds_{run}.nc')
        if run in ['ersst','cobe']:
            monthly_ds = monthly_ds.rename({'lat':'latitude','lon':'longitude'})
        if run=='ersst':
            monthly_ds = monthly_ds.drop('lev')
            monthly_ds = monthly_ds.squeeze()#drop('lev')
            print(monthly_ds)
        SST_stacked = monthly_ds.stack(z=('latitude', 'longitude'))
        print(SST_stacked)
        # break
        MMM_natural = xr.open_dataarray(f'{path_prace}/GMST/CMIP5_natural.nc', decode_times=False)
        MMM_anthro  = xr.open_dataarray(f'{path_prace}/GMST/CMIP5_anthro.nc' , decode_times=False)
        monthly_MMM_natural = np.repeat(MMM_natural, 12)
        monthly_MMM_anthro  = np.repeat(MMM_anthro , 12)
        monthly_MMM_natural = monthly_MMM_natural.assign_coords(time=monthly_ds.time)
        monthly_MMM_anthro  = monthly_MMM_anthro .assign_coords(time=monthly_ds.time)
        forcings = monthly_MMM_natural.to_dataframe(name='natural').join(
                    monthly_MMM_anthro.to_dataframe(name='anthro'))

        ds_constant = SST_stacked[0,:].squeeze().copy()
        ds_anthro   = SST_stacked[0,:].squeeze().copy()
        ds_natural  = SST_stacked[0,:].squeeze().copy()

        # multiple linear regression
        X = sm.add_constant(forcings[['anthro', 'natural']])
        for i, coordinate in tqdm(enumerate(SST_stacked.z)):
            y = SST_stacked[:, i].values
            model = sm.OLS(y, X).fit()
            ds_constant[i] = model.params['const']
            ds_natural[i]  = model.params['natural']
            ds_anthro[i]   = model.params['anthro']

        constant     = ds_constant.unstack('z')
        beta_anthro  = ds_anthro  .unstack('z')
        beta_natural = ds_natural .unstack('z')

        ds = xr.merge([{'forcing_anthro' : monthly_MMM_anthro} ,{'beta_anthro' :beta_anthro} ,
                       {'forcing_natural': monthly_MMM_natural},{'beta_natural':beta_natural},
                       {'constant':constant}])
        ds.to_netcdf(f'{path_prace}/SST/SST_monthly_MMM_fit_{run}.nc')
        
        monthly_ds_dt = monthly_ds.assign_coords(time=monthly_MMM_anthro.time) \
                            - beta_anthro*monthly_MMM_anthro \
                            - beta_natural*monthly_MMM_natural \
                            - constant
        monthly_ds_dt.to_netcdf(f'{path_prace}/SST/SST_monthly_ds_dt_{run}.nc')
        
        return
        
    
    def isolate_Pacific_SSTs(self, run, extent, time):
        """"""
        print('isolate Pacific SSTs')
        if time is not None: print(time, time[0], time[1])
        assert run in ['ctrl', 'lpd', 'lc1', 'had']
        assert extent in ['38S', 'Eq', '20N']
        
        if run=='had':  fn = f'{path_prace}/SST/SST_monthly_ds_dt_had.nc'
        else:           fn = f'{path_prace}/SST/SST_monthly_ds_dt_{run}_{time[0]}_{time[1]}.nc'
            
        if run=='ctrl':              domain = 'ocn_rect'
        elif run in ['lpd', 'lc1']:  domain = 'ocn_low'
        elif run=='had':             domain = 'ocn_had'
            
        area = xr.open_dataarray(f'{path_prace}/geometry/AREA_{extent}_{domain}.nc')  # created in SST_PDO.ipynb
        da = xr.open_dataarray(fn)
        if run=='had':  da = self.shift_had(da)  # longitude: [-180,180] -> [0,360]
        da = da.where(np.isnan(area)==False)
        da = self.focus_data(da)
        # print(da)
        if run in ['ctrl', 'lpd', 'lc1']:
            fn_out = f'{path_prace}/SST/SST_monthly_ds_dt_{extent}_{run}_{time[0]}_{time[1]}.nc'
        else:
            fn_out = f'{path_prace}/SST/SST_monthly_ds_dt_{extent}_{run}.nc'
        da.where(np.isnan(area)==False).to_netcdf(fn_out)
            
        return


    def SST_remove_forced_signal(self, run, tavg='yrly', detrend_signal='GMST', time=None):
        """ detrending the SST field
        a) remove the scaled, forced MMEM GMST signal (method by Kajtar et al. (2019)) at each grid point
        b) remove MMEM SST index (Steinman et al. (2015))

        1. load raw SST data
        2. generate forced signal
            model:  fit to GMST
                linear
                quadratic
            observations:
                single-factor CMIP GMST MMEM
                two-factor CMIP all natural and CMIP anthropogenic (= all forcings - all natural)
        3. regression:
            single time series: forced signal onto SST data -> \beta
            two time series:
        4. use regression coefficient \beta to generate SST signal due to forcing
        5. remove that signal

        run            .. CESM simulation name
        tavg           .. time resolution
        detrend_signal .. either GMST (Kajtar et al. (2019))
                          or target region (Steinman et al. (2015))
        time           .. time range selected
        """
        assert run in ['ctrl', 'rcp', 'lpd', 'lpi', 'had']
        assert tavg in ['yrly', 'monthly']
        assert detrend_signal in ['GMST', 'AMO', 'SOM', 'TPI1', 'TPI2', 'TPI3']
        if detrend_signal in ['AMO', 'SOM', 'TPI1', 'TPI2', 'TPI3']:
            assert run=='had'
        if run=='had':
            assert time==None

        # file name and domain
        fn = f'{path_prace}/SST/SST_{tavg}_{run}.nc'
        if run in ['ctrl', 'rcp']:
            if tavg=='yrly':
                domain = 'ocn'
            elif tavg=='monthly':
                domain = 'ocn_rect'
        elif run in ['lpd', 'lpi']:  
            domain = 'ocn_low'
        elif run=='had':
            domain = 'ocn_had'

        print('load and subselect data')
        MASK = boolean_mask(domain=domain, mask_nr=0, rounded=True)
        SST = self.select_time(xr.open_dataarray(f'{path_prace}/SST/SST_{tavg}_{run}.nc',\
                                                  decode_times=False).where(MASK),
                                time)
        
        if time!=None:
            first_year, last_year = time
        
        if tavg=='monthly':  # deseasonalize
            for t in range(12):
                SST[t::12,:,:] -= SST[t::12,:,:].mean(dim='time')
        SST = SST - SST.mean(dim='time')

        print('calculate forced signal')
        forced_signal = self.forcing_signal(run=run, tavg=tavg, detrend_signal=detrend_signal, time=time)

        if detrend_signal=='GMST':
            print('Kajtar et al. (2019) scaled MMM GMST detrending method')
            if time==None:
                fn = f'{path_prace}/SST/SST_beta_{tavg}_all_{run}.nc'
            else:
                fn = f'{path_prace}/SST/SST_beta_{tavg}_{detrend_signal}_{run}_{first_year}_{last_year}.nc'
                
            try:
                assert 1==0
                assert os.path.exists(fn)
                print('reusing previously calculated beta!')
                print(f'file exists: {fn}')
                beta = xr.open_dataset(fn).slope
            except:
                if run=='ctrl': SST = SST[40:,:,:]
                beta = ADA().lag_linregress(forced_signal, SST)['slope']
                if run=='had':
                    beta = xr.where(abs(beta)<5, beta, np.median(beta))
                ds = xr.merge([forced_signal, beta])
                ds.to_netcdf(fn)
                
            SST_dt = SST - beta * forced_signal
            SST_dt = SST_dt - SST_dt.mean(dim='time')
            print('test')

            # output name
            if run=='had':
                dt = 'sfdt'  # single factor detrending
            elif run in ['ctrl', 'lpd', 'rcp']:
                dt = 'sqdt'  # scaled quadratic detrending
            else:
                dt = 'sldt'  # scaled linear detrending
            
        elif detrend_signal in ['AMO', 'SOM', 'TPI1', 'TPI2', 'TPI3']:
            print('Steinman et al. (2015) method')
            # these indices will be detrended afterwards
            SST_dt = SST - forced_signal
            ds = None
            dt = f'{detrend_signal}dt'

        print('writing output')
        if time==None:
            fn = f'{path_prace}/SST/SST_{tavg}_{dt}_{run}.nc'
        else:
            fn = f'{path_prace}/SST/SST_{tavg}_{dt}_{run}_{first_year}_{last_year}.nc'
        SST_dt.to_netcdf(fn)
        print(f'detrended {run} SST file written out to:\n{fn}')
        
        # additional two factor detrending for had
        if run=='had' and tavg=='yrly':
            self.two_factor_detrending(SST)
            
        return


    def forcing_signal(self, run, tavg, detrend_signal, time=None):
        """ GMST forced component
        run            .. dataset
        tavg           .. time resolution
        detrend_signal .. 
        time
        """
        print('creating forcing signal')
        assert run in ['ctrl', 'rcp', 'lpd', 'lpi', 'had']
        assert tavg in ['yrly', 'monthly']
        assert detrend_signal in ['GMST', 'AMO', 'SOM', 'TPI1', 'TPI2', 'TPI3']

        # simulations: linear/quadratic fit to GMST signal
        if run in ['ctrl', 'rcp', 'lpd', 'lpi']:
            assert detrend_signal=='GMST'
            if run=='rcp':  # need actual GMST time series
                forced_signal = xr.open_dataset(f'{path_prace}/GMST/GMST_{tavg}_{run}.nc', decode_times=False).GMST
                forced_signal = xr_quadtrend(forced_signal)
            elif run in ['ctrl', 'lpd']:  # use global mean SST as as proxy
                forced_signal = xr.open_dataarray(f'{path_prace}/SST/GMSST_{tavg}_{run}.nc', decode_times=False)
                if run=='ctrl':  # strong adjustment in the first 40 years
                    forced_signal = forced_signal[40:]
                forced_signal = xr_quadtrend(forced_signal)
            else:  # create mock linear trend 
                times = xr.open_dataarray(f'{path_prace}/SST/SST_{tavg}_{run}.nc', decode_times=False).time
                forced_signal = xr.DataArray(np.linspace(0,1,len(times)),
                                             coords={'time': np.sort(times.values)}, dims=('time'))
                forced_signal.name = 'GMST'
                forced_signal.attrs = {'Note':'This is the mock linear trend, simply going from 0 to 1'}
            #if tavg=='yrly':
                #times = forced_signal['time'] + 31 # time coordinates shifted by 31 days (SST saved end of January, GMST beginning)
                #if run=='ctrl':  # for this run, sometimes 31 days, sometimes 15/16 days offset
                    #times = xr.open_dataset(f'{path_prace}/SST/SST_yrly_ctrl.nc', decode_times=False).time
                #forced_signal = forced_signal.assign_coords(time=times)

        # observations: CMIP5 multi model ensemble mean of all forcings GMST
        elif run=='had':
            forced_signal = xr.open_dataarray(f'{path_data}/CMIP5/KNMI_CMIP5_{detrend_signal}_{tavg}.nc', decode_times=False)
            if tavg=='monthly':  # deseasonalize
                for t in range(12):
                    forced_signal[t::12] -= forced_signal[t::12].mean(dim='time')

            if tavg=='yrly':# select 1870-2018
                times = (forced_signal['time'].astype(int) - 9)*365
                forced_signal = forced_signal.assign_coords(time=times)  # days since 1861
                forced_signal = forced_signal[9:158]
            elif tavg=='monthly':
                # ...
                forced_signal = forced_signal[9*12:158*12-1]
                times = xr.open_dataarray(f'{path_prace}/SST/SST_monthly_had.nc', decode_times=False).time.values
                forced_signal = forced_signal.assign_coords(time=times)

        forced_signal = self.select_time(forced_signal, time)

        forced_signal -= forced_signal.mean()
        forced_signal.name = 'forcing'
        return forced_signal
    
    
    def SST_pointwise_detrending(self, run, tavg='yrly', degree=2, time=None):
        """ calculates the trends of ocean fields

        input:
        SST    .. xr DataArray
        degree .. degree of polynomial to remove

        output:
        SST_dt .. pointwise detrended SST field

        7 secs for lpd run, 40 seconds
        """
        print('detrending SST pointwise')
        assert degree in [1, 2]
        if run in ['ctrl', 'rcp']:          MASK = boolean_mask('ocn'     , 0)
        elif run in ['lpi', 'lpd', 'lc1']:  MASK = boolean_mask('ocn_low' , 0)
        (jm, im) = MASK.shape
        fn = f'{path_prace}/SST/SST_{tavg}_{run}.nc'
        da = xr.open_dataset(fn, decode_times=False).SST
        SST = self.select_time(da.where(MASK), time)
        SST = SST.where(MASK>0).fillna(-9999)
        Nt = SST.values.shape[0]
        A = SST.values.reshape((Nt, im*jm))

        SST_pf = np.polyfit(SST.time, A, degree)
        
        pf0 = A[0,:].copy()
        pf1 = A[0,:].copy()
        pf0 = SST_pf[0,:]
        pf1 = SST_pf[1,:]
        if degree==1:
            # SST_dt = pf0*SST.time - pf1
            detrend_signal = 'linear'
        elif degree==2:
            pf2 = A[0,:].copy()
            pf2 = SST_pf[2,:]
            A_dt = np.expand_dims(SST.time**2             , 1).dot(np.expand_dims(SST_pf[0,:], 0)) \
                 + np.expand_dims(SST.time                , 1).dot(np.expand_dims(SST_pf[1,:], 0)) \
                 + np.expand_dims(np.ones((len(SST.time))), 1).dot(np.expand_dims(SST_pf[2,:], 0))
            # detrend_signal = 'quadratic'
        dt = 'pwdt'  # pointwise detrended

        fn_new = f'{path_prace}/SST/SST_yrly_{dt}_{run}_{time[0]}_{time[1]}.nc'
        SST_dt = SST.copy()
        SST_dt.values = (A-A_dt).reshape((Nt,jm,im))
        SST_dt.to_netcdf(fn_new)
        print(f'created {fn_new}')
        return
    
    
    def two_factor_detrending(self, SST):
        print(f'additional two factor detrending for had SST')
        # load CMIP5 multi-model means
        forcing_natural = xr.open_dataarray(f'{path_prace}/GMST/CMIP5_natural.nc', decode_times=False)
        forcing_anthro  = xr.open_dataarray(f'{path_prace}/GMST/CMIP5_anthro.nc' , decode_times=False)
        forcing_all     = xr.open_dataarray(f'{path_prace}/GMST/CMIP5_all.nc'    , decode_times=False)

        for forcing in [forcing_natural, forcing_anthro, forcing_all]:
            forcing.coords['time'] = (forcing.time-9)*365

        forcings = forcing_natural.to_dataframe(name='natural').join(
                     [forcing_anthro.to_dataframe( name='anthro'),
                      forcing_all.to_dataframe(name='all')])

        SST_stacked = SST.stack(z=('latitude', 'longitude'))
        ds_anthro   = SST_stacked[0,:].squeeze().copy()
        ds_natural  = SST_stacked[0,:].squeeze().copy()

        # multiple linear regression
        X = sm.add_constant(forcings[['anthro', 'natural']])
        for i, coordinate in enumerate(SST_stacked.z):
            y = SST_stacked[:, i].values
            model = sm.OLS(y, X).fit()
            ds_anthro[i] = model.params['anthro']
            ds_natural[i] = model.params['natural']

        beta_anthro  = ds_anthro .unstack('z')
        beta_natural = ds_natural.unstack('z')
        
        # output
        ds = xr.merge([{'forcing_anthro': forcing_anthro}, {'beta_anthro': beta_anthro}])
        ds.to_netcdf(f'{path_prace}/SST/SST_yrly_beta_anthro_had.nc')
        ds = xr.merge([{'forcing_natural': forcing_natural}, {'beta_natural':beta_natural}])
        ds.to_netcdf(f'{path_prace}/SST/SST_yrly_beta_natural_had.nc')

        SST_dt = SST - beta_anthro*forcing_anthro - beta_natural*forcing_natural

        # two factor detrending
        fn = f'{path_prace}/SST/SST_yrly_tfdt_had.nc'
        SST_dt.to_netcdf(fn)
        print(f'detrended had SST file written out to:\n{fn}')
        return
       
    
    def select_time(self, data, time):
        """ if time is not `full`, return subselected data
        time .. either `full` or (start_year, end_year) tuple
        """
        if time!=None:  # use subset in time
            t0, t1 = data.time[0], data.time[1]
            
            # time in days
            if np.isclose(t1-t0, 365, atol=1) or np.isclose(t1-t0, 30, atol=2):
                time_coords = tuple(365*t+31 for t in time)
            
            # monthly data with time in years
            elif np.isclose(t1-t0, 1) or np.isclose(t1-t0, 1/12):
                time_coords = tuple(time)
                
            data = data.sel({'time':slice(*time_coords)})
            # sometimes an additional data point is selected, need to remove that
            if len(data.time) in [150, 151, 251, 1789, 3001]:  data = data.isel(time=slice(0,-1))
        else:  pass  # return original data
        return data
    
    
    def remove_superfluous_files(self, fn):
        print('removing superfluous files')
        for x in glob.glob(fn):
            os.remove(x) 
        
        
    def shift_had(self, da):
        """ shifts lons to [0,360] to make Pacific contiguous """
        return da.assign_coords(longitude=(da.longitude+360)%360).roll(longitude=180, roll_coords=True)
    
    def shift_ocn_rect(self, da, back=False):
        """ shifts t_lon between original [0,360] and [-180, 180] intervals
        the latter makes the Altantic points contiguous """
        if back==False:
            return da.assign_coords(t_lon=(da.t_lon - 360*((da.t_lon-da.t_lon[0])//180))).roll(t_lon=450, roll_coords=True)
        else:
            return da.assign_coords(t_lon=(da.t_lon+360)%360).roll(t_lon=180, roll_coords=True)
    
    def shift_ocn_low(self, da, back=False):
        """ shifts nlon between curvilinear [0,320] and [-160,160] frame
        the latter allows for a contiguous Atlantic """
        if back==False:
            return da.assign_coords(nlon=(da.nlon - 320*(da.nlon//160))).roll(nlon=160, roll_coords=True)
        else: 
            return da.assign_coords(nlon=(da.nlon+320)%320).roll(nlon=160, roll_coords=True)
    
    
    def focus_data(self, da):
        """ drops data outside rectangle around Pacific """
        if 't_lat' in da.coords:  # ctrl
            lat, lon = 't_lat', 't_lon'
        elif 'nlat' in da.coords:  # lpd
            lat, lon = 'nlat', 'nlon'
        elif 'latitude' in da.coords:  # had, ersst, cobe
            lat, lon = 'latitude', 'longitude'
        else:  raise ValueError('xr DataArray does not have the right lat/lon coords.')
        da = da.dropna(dim=lat, how='all')
        da = da.dropna(dim=lon, how='all')
        return da