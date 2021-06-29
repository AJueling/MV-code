import os
import numpy as np

# POP files: 
#  75-275: /projects/0/samoc/pop/tx0.1/output/run_henk_mixedbc/tavg_rectgrid/
# 276-326: /projects/0/samoc/pop/tx0.1/output/run_henk_mixedbc_extravars_viebahn/tavg_rectgrid/

def CESM_filename(domain, run, y, m, d=0, name=None):
    """ filename creation 
    
    input:
    domain   .. (str) 'ocn', 'ocn_rect', 'ocn_low', 'atm', 'ice'
    run      .. (str) 'ctrl', 'rcp', 'lpd', 'lpi', 'pop'
    y        .. (int) year
    m        .. (int) month; 
                      if 0, then name of yearly avg. file
                      if 13, then monthly multifile name
    d
    name     .. (str) added to yrly file
                      if d==31 and domain=='ocn', then 'SST' or 'SSH must be provided'
    
    output:
    file     .. (str) filename
    """
    assert domain in ['ocn', 'ocn_rect', 'ocn_low', 'atm', 'ice']
    assert run in ['ctrl',         # high res present day control run
                   'rcp',          # high res RCP8.5 run
                   'ptx', 'pgx',   # POP tx0.1, gx1 binary files for control runs
                   'lpd',          # low res present day control run with CESM versions 1.12 (>1300 years)
                   'lc1',          # present day low res CESM 1.04 (branched off at year 1000 from `lpd`)
                   'lpi',          # low res pre-industrial control run
                   'pop',          # high res ocean only run ocn_rect
                   'lr1', 'lr2',   # low res RCP8.5 runs
                   'hq',           # high res quadroupling run
                   'ld',           # low res doubling run
                   'lq']           # low res quadroupling run
    assert type(y)==np.dtype(int) and type(m)==np.dtype(int) and type(d)==np.dtype(int)
    assert m>=0 and m<14
    assert d>=0 and d<33
    
    time = f'{y:04}-{m:02}'
    time2 = f'{y:04}{m:02}'
    
    if m==13:  # returning multi-file name
        time = '*'
    
    # yearly output
    if m==0:  
        if domain in ['ocn', 'ocn_low']:
            file = f'{path_prace}/{run}/ocn_yrly_{name}_{y:04}.nc'
    
        elif domain=='ocn_rect':
            if run=='ctrl':
                file = f'{path_prace}/ctrl_rect/{name}_yrly_{y:04}.interp900x602.nc'
            elif run=='rcp':
                file = f'{path_yrly_rcp}/ocn_yrly_{name}_{y:04}.interp900x602.nc'
            elif run=='pop':
                if y<276:
                    file = f'{path_ocn_rect_pop1}/yearly/{popstr}.avg{y:04}.nc'
                elif y>=276:
                    file = f'{path_ocn_rect_pop2}/{popstr}.avg{y:04}.nc'
                    
        elif domain=='atm':
            if run=='ctrl':
                file = f'{path_yrly_ctrl}/atm_yrly_{name}_{y:04}.nc'
            elif run=='rcp':
                file = f'{path_yrly_rcp}/atm_yrly_{name}_{y:04}.nc'
            elif run=='lpd':
                if name==None:
                    file = f'{path_atm_lpd}/yearly/{lpdstr}.cam.h0.avg{y:04}.nc'
                else:
                    # file = f'{path_atm_lpd}/yearly/atm_yrly_{name}_{y:04}.nc'
                    # should be path_yrly_lpd
                    file = f'{path_yrly_lpd}/atm_yrly_{name}_{y:04}.nc'
            elif run=='lc1':
                file = f'{path_yrly_lc1}/atm_yrly_{name}_{y:04}.nc'
            elif run=='lpi':
                file = f'{path_yrly_lpi}/atm_yrly_{name}_{y:04}.nc'
            elif run=='lr1':
                file = f'{path_yrly_lr1}/atm_yrly_{name}_{y:04}.nc'
            elif run=='lr2':
                file = f'{path_yrly_lr2}/atm_yrly_{name}_{y:04}.nc'
            elif run=='hq':
                file = f'{path_yrly_hq}/atm_yrly_{name}_{y:04}.nc'
            elif run=='lq':
                file = f'{path_yrly_lq}/atm_yrly_{name}_{y:04}.nc'
            elif run=='ld':
                file = f'{path_yrly_ld}/atm_yrly_{name}_{y:04}.nc'
    
        elif domain=='ice':
            if run=='ctrl':
                file = f'{path_yrly_ctrl}/ice_yrly_{name}_{y:04}.nc'
            elif run=='rcp':
                file = f'{path_yrly_rcp}/ice_yrly_{name}_{y:04}.nc'
            else:  raise ValueError('not implemented')
    
    # monthly output
    elif m>0 and d==0: 
        if domain=='ocn':
            if run=='ctrl':
                file = f'{path_ocn_ctrl}/{spinup}.pop.h.{time}.nc'
            elif run=='rcp':
                file = f'{path_ocn_rcp}/{rcpstr}.pop.h.{time}.nc'
            elif run=='hq':
                file = f'{path_ocn_hq}/{hqstr}.pop.h.{time}.nc'
            # the following are technically `ocn_low`
            elif run=='lpd':
                file = f'{path_ocn_lpd}/{lpdstr}.pop.h.{time}.nc'
            elif run=='lc1':
                file = f'{path_ocn_lc1}/{lc1str}.pop.h.{time}.nc'
            elif run=='lpi':
                file = f'{path_ocn_lpi}/{lpistr}.pop.h.{time}.nc'
            elif run=='lr1':
                file = f'{path_ocn_lr1}/{lr1str}.pop.h.{time}.nc'
            elif run=='lr2':
                file = f'{path_ocn_lr2}/{lr2str}.pop.h.{time}.nc'
            elif run=='lq':
                file = f'{path_ocn_lq}/{lqstr}.pop.h.{time}.nc'
            elif run=='ld':
                file = f'{path_ocn_ld}/{ldstr}.pop.h.{time}.nc'

        elif domain=='ocn_rect':
            if run=='ctrl':
                file = f'{path_ocn_rect_ctrl}/{spinup}.pop.h.{time}.interp900x602.nc'
            elif run=='rcp':
                file = f'{path_ocn_rect_rcp}/{rcpstr}.pop.h.{time}.interp900x602.nc'
            elif run=='pop':
                if y<276:
                    file = f'{path_ocn_rect_pop1}/monthly/{popstr}.{time2}.interp900x602.nc'
                elif y>=276:
                    file = f'{path_ocn_rect_pop2}/{popstr}.{time2}.interp900x602.nc'

        elif domain=='ocn_low':
            if run=='lpd':
                file = f'{path_ocn_lpd}/{lpdstr}.pop.h.{time}.nc'
            elif run=='lpi':
                file = f'{path_ocn_lpi}/{lpistr}.pop.h.{time}.nc'
            elif run=='lc1':
                file = f'{path_ocn_lc1}/{lc1str}.pop.h.{time}.nc'
            elif run=='lr1':
                file = f'{path_ocn_lr1}/{lr1str}.pop.h.{time}.nc'
            elif run=='lr2':
                file = f'{path_ocn_lr2}/{lr2str}.pop.h.{time}.nc'
            elif run=='ld':
                file = f'{path_ocn_ld}/{ldstr}.pop.h.{time}.nc'
            elif run=='lq':
                file = f'{path_ocn_lq}/{lqstr}.pop.h.{time}.nc'

        elif domain=='atm':
            if run=='ctrl':
                file = f'{path_atm_ctrl}/{spinup}.cam2.h0.{time}.nc'
            elif run=='rcp':
                file = f'{path_atm_rcp}/{rcpstr}.cam2.h0.{time}.nc'
            elif run=='lpd':
                file = f'{path_atm_lpd}/monthly/{lpdstr}.cam.h0.{time}.nc'
#                 raise ValueError('monthly files are not available for lpd run!')
            elif run=='lc1':
                file = f'{path_atm_lc1}/{lc1str}.cam2.h0.{time}.nc'
            elif run=='lpi':
                file = f'{path_atm_lpi}/{lpistr}.cam2.h0.{time}.nc'
            elif run=='lr1':
                file = f'{path_atm_lr1}/{lr1str}.cam.h0.{time}.nc'
            elif run=='lr2':
                file = f'{path_atm_lr2}/{lr2str}.cam.h0.{time}.nc'
            elif run=='hq':
                file = f'{path_atm_hq}/{hqstr}.cam2.h0.{time}.nc'
            elif run=='ld':
                file = f'{path_atm_ld}/{ldstr}.cam.h0.{time}.nc'
            elif run=='lq':
                file = f'{path_atm_lq}/{lqstr}.cam.h0.{time}.nc'

        elif domain=='ice':
            if run=='ctrl':
                file = f'{path_ice_ctrl}/{spinup}.cice.h.{time}.nc'
            elif run=='lpd':
                file = f'{path_ice_lpd}/{lpdstr}.cice.h.{time}.nc'
            elif run=='rcp':
                file = f'{path_ice_rcp}/{rcpstr}.cice.h.{time}.nc'
            elif run=='hq':
                file = f'{path_ice_hq}/{hqstr}.cice.h.{time}.nc'
            elif run=='lc1':
                file = f'{path_ice_lc1}/{lc1str}.cice.h.{time}.nc'
            elif run=='lr1':
                file = f'{path_ice_lr1}/{lr1str}.cice.h.{time}.nc'
            elif run=='lq':
                file = f'{path_ice_lq}/{lqstr}.cice.h.{time}.nc'
            else:  raise ValueError('not implemented')
    
    # daily output
    elif d>0 and d<33:
        daily_error = 'daily data only for `ctrl` and `rcp`, `ocn` implemented'
        if domain=='ocn':
            if d<32:  # SSH, velocity data with daily files
                if run=='ctrl':
                    file = f'{path_ctrl}/OUTPUT/ocn/hist/daily/{spinup}.pop.hm.{time}-{d:02d}.nc'
                elif run=='rcp':
                    file = f'{path_rcp}/OUTPUT/ocn/hist/daily/{rcpstr}.pop.hm.{time}-{d:02d}.nc'
                else:  raise ValueError(daily_error)
            elif d==32:  # SST data: monthly aggregated files with daily fields
                if run=='ctrl':
                    file = f'{path_ctrl}/OUTPUT/ocn/hist/daily/{spinup}.pop.h.nday1.{time}-01.nc'
                elif run=='rcp':
                    file = f'{path_rcp}/OUTPUT/ocn/hist/daily/{rcpstr}.pop.h.nday1.{time}-01.nc'
                    try:
                        assert os.path.exists(file)
                    except:
                        for dd in np.arange(2,32):
                            fn = f'{path_rcp}/OUTPUT/ocn/hist/daily/{rcpstr}.pop.h.nday1.{time}-{dd:02d}.nc'
                            if os.path.exists(fn):
                                file = fn
                                break
                            else: 
                                continue
                else:  raise ValueError(daily_error)
                
        else:  raise ValueError(daily_error)
        
    # check existence of files
    if m<13 and os.path.exists(file)==False:
        print(f'The file "{file}" does not exist')
        
    return file


# STRINGS

spinup  = 'spinup_pd_maxcores_f05_t12'
rcpstr  = 'rcp8.5_co2_f05_t12'
lr1str  = 'rcp8.5_co2_f09_g16'
lr2str  = 'rcp8.5_co2_f09_g16.002'
lpdstr  = 'spinup_B_2000_cam5_f09_g16'
lc1str  = 'spinup_pd_maxcores_f09_g16'
lpistr  = 'b.PI_1pic_f19g16_NESSC_control'
popstr  = 't.t0.1_42l_nccs01'
hqstr   = 'b.e10.B2000_CAM5.f05_t12.pd_control.4xco2.001'
ldstr   = 'b.e10.B2000_CAM5.f09_g16.pd_control.2xco2.001'
lqstr   = 'b.e10.B2000_CAM5.f09_g16.pd_control.4xco2.001'


# PATHS

# my output (write)
path_results = '/home/ajueling/CESM/results'
path_data    = '/home/ajueling/CESM/data'
path_andre   = '/home/ajueling'
path_prace   = '/projects/0/prace_imau/prace_2013081679/andre'
path_samoc   = path_prace

# CESM data (should read only)
path_CESM104 = '/projects/0/prace_imau/prace_2013081679/cesm1_0_4'
path_CESM112 = '/projects/0/acc/cesm/cesm1_1_2'
path_CESM105 = '/projects/0/acc/cesm/cesm1_0_5'

path_ctrl = f'{path_CESM104}/{spinup}'
path_rcp  = f'{path_CESM104}/{rcpstr}'
path_lpd  = f'{path_CESM112}/{lpdstr}'
path_lc1  = f'{path_CESM104}/f09_g16/{lc1str}'
path_lpi  = f'{path_CESM105}/{lpistr}'
path_pop  = f'/projects/0/samoc/pop/tx0.1'
path_lr1  = f'{path_CESM112}/{lr1str}'
path_lr2  = f'{path_CESM112}/{lr2str}'
path_hq   = f'{path_CESM104}/{hqstr}'
path_ld   = f'{path_CESM112}/{ldstr}'
path_lq   = f'{path_CESM112}/{lqstr}'

# grid
path_ocn_grid = f'{path_CESM104}/inputdata/ocn/pop/tx0.1v2/grid/'

# currently running
path_run_ctrl = f'{path_ctrl}/run'
path_run_rcp  = f'{path_rcp}/run'

# then copied to
path_ocn_ctrl = f'{path_ctrl}/OUTPUT/ocn/hist/monthly'
path_atm_ctrl = f'{path_ctrl}/OUTPUT/atm/hist/monthly'
path_ice_ctrl = f'{path_ctrl}/OUTPUT/ice/hist/monthly'

path_ocn_rcp  = f'{path_rcp}/OUTPUT/ocn/hist/monthly'
path_atm_rcp  = f'{path_rcp}/OUTPUT/atm/hist/monthly'
path_ice_rcp  = f'{path_rcp}/OUTPUT/ice/hist/monthlies'

path_ocn_lpd  = f'{path_lpd}/OUTPUT/ocn/hist/monthly'
path_atm_lpd  = f'{path_lpd}/OUTPUT/atm/hist'
path_ice_lpd  = f'{path_lpd}/OUTPUT/ice/hist'


path_ocn_lc1  = f'{path_lc1}/run'#/wrong_init_TS'
path_atm_lc1  = f'{path_lc1}/run'#/wrong_init_TS'
path_ice_lc1  = f'{path_lc1}/run'#/wrong_init_TS'

path_ocn_lpi  = f'{path_lpi}/OUTPUT/ocn/hist/monthly'
path_atm_lpi  = f'{path_lpi}/OUTPUT/atm/hist/monthly'

path_ocn_lr1  = f'{path_lr1}/OUTPUT/ocn/hist/monthly'
path_atm_lr1  = f'{path_lr1}/OUTPUT/atm/hist/monthly'
path_ice_lr1  = f'{path_lr1}/OUTPUT/ice/hist'

# path_ocn_lr2  = f'{path_lr2}/OUTPUT/ocn/hist/monthly'
path_ocn_lr2  = f'{path_lr2}/run'
# path_atm_lr2  = f'{path_lr2}/OUTPUT/atm/hist/monthly'
path_atm_lr2  = f'{path_lr2}/run'
path_ice_lr2  = f'{path_lr2}/OUTPUT/ice/hist/monthly'

path_ocn_hq   = f'{path_hq}/OUTPUT/ocn/hist/monthly'
path_atm_hq   = f'{path_hq}/OUTPUT/atm/hist/monthly'
path_ice_hq   = f'{path_hq}/OUTPUT/ice/hist'

path_ocn_ld   = f'{path_ld}/OUTPUT/ocn/hist/monthly'
path_atm_ld   = f'{path_ld}/OUTPUT/atm/hist/monthly'
path_ice_ld   = f'{path_ld}/OUTPUT/ice/hist/monthly'

path_ocn_lq   = f'{path_lq}/OUTPUT/ocn/hist/monthly'
path_atm_lq   = f'{path_lq}/OUTPUT/atm/hist/monthly'
path_ice_lq   = f'{path_lq}/OUTPUT/ice/hist'

# interpolated to rectangular 0.4 deg grid
path_ocn_rect_ctrl = f'{path_ctrl}/OUTPUT/ocn/hist/monthly_rect'
path_atm_ctrl_rect = f'{path_ctrl}/OUTPUT/atm/hist/monthly_rect'

path_ocn_rect_rcp  = f'{path_rcp}/OUTPUT/ocn/hist/monthly_rect'
path_atm_rcp_rect  = f'{path_rcp}/OUTPUT/atm/hist/monthly_rect'

path_ocn_rect_pop1 = f'{path_pop}/output/run_henk_mixedbc/tavg_rectgrid'
path_ocn_rect_pop2 = f'{path_pop}/output/run_henk_mixedbc_extravars_viebahn/tavg_rectgrid'


# OHC files created by Rene
path_rene          = '/projects/0/prace_imau/prace_2013081679/rene/CESM'
path_ohc_rene      = f'{path_rene}/OceanHeatContent/Global_0.1'
path_ohc_rene_rect = f'{path_rene}/OceanHeatContent/Global_0.4'

# yearly files
path_yrly_ctrl = f'{path_prace}/ctrl'
path_yrly_rcp  = f'{path_prace}/rcp'
path_yrly_lpd  = f'{path_prace}/lpd'
path_yrly_lc1  = f'{path_prace}/lc1'
path_yrly_lpi  = f'{path_prace}/lpi'
path_yrly_lr1  = f'{path_prace}/lr1'
path_yrly_lr2  = f'{path_prace}/lr2'
path_yrly_hq   = f'{path_prace}/hq'
path_yrly_ld   = f'{path_prace}/ld'
path_yrly_lq   = f'{path_prace}/lq'

# FILES

grid_file  = f'{path_CESM104}/inputdata/ocn/pop/tx0.1v2/grid/horiz_grid_200709.ieeer8'

# example files to use for tests
file_ex_ocn_ctrl = CESM_filename(domain='ocn', run='ctrl', y= 200, m=1)
file_ex_ocn_rcp  = CESM_filename(domain='ocn', run='rcp' , y=2000, m=1)
file_ex_ocn_lpd  = CESM_filename(domain='ocn', run='lpd' , y= 200, m=1)
file_ex_ocn_lc1  = CESM_filename(domain='ocn', run='lc1' , y=   1, m=1)
# file_ex_ocn_lpi  = CESM_filename(domain='ocn', run='lpi' , y=1600, m=1)
# some files are commented out as they have been moved to the archive

# daily data
# file_ex_ocn_daily_SST_ctrl = CESM_filename(domain='ocn', run='ctrl', y= 249, m=1, d=32, name='SST')
# file_ex_ocn_daily_SSH_ctrl = CESM_filename(domain='ocn', run='ctrl', y= 249, m=1, d=1, name='SSH')
# file_ex_ocn_daily_SST_rcp  = CESM_filename(domain='ocn', run='rcp' , y=2001, m=1, d=32, name='SST')
# file_ex_ocn_daily_SSH_rcp  = CESM_filename(domain='ocn', run='rcp' , y=2001, m=1, d=1, name='SSH')

file_ex_ocn_rect  = f'{path_ocn_rect_ctrl}/{spinup}.pop.h.0200-01.interp900x602.nc'

file_ex_atm_ctrl = CESM_filename(domain='atm', run='ctrl', y= 200, m=1)
file_ex_atm_rcp  = CESM_filename(domain='atm', run='rcp' , y=2000, m=1)
file_ex_atm_lpd  = CESM_filename(domain='atm', run='lpd' , y= 200, m=0)
file_ex_atm_lc1  = CESM_filename(domain='atm', run='lc1' , y=   1, m=1)
file_ex_atm_lpi  = CESM_filename(domain='atm', run='lpi' , y=3000, m=1)

file_ex_ice_rcp  = f'{path_ice_rcp}/{rcpstr}.cice.h.2000-01.nc'
file_ex_ice_yrly = f'{path_ice_rcp}/{rcpstr}.cice.h.avg2000.nc'

file_HadISST     = f'{path_data}/HadISST/HadISST_sst.nc'

# derived data
file_ex_ohc_hires = f'{path_ohc_rene}/OHC_0200-01_All.nc'

file_ex_ocn_TEMP_PD_yrly = f'{path_prace}/ctrl/ocn_yrly_TEMP_PD_0200.nc'
file_ex_atm_T_T850_U_V_yrly  = f'{path_prace}/ctrl/atm_yrly_T_T850_U_V_0200.nc'


file_geometry = f'{path_ocn_grid}/dzbc_pbc_s2.0_200709.ieeer8'

file_RMASK_ocn      = f'{path_prace}/grid/RMASK_ocn.nc'
file_RMASK_ocn_rect = f'{path_prace}/grid/RMASK_ocn_rect.nc'
file_RMASK_ocn_low  = f'{path_prace}/grid/RMASK_ocn_low.nc'
file_RMASK_ocn_had  = f'{path_prace}/grid/RMASK_ocn_had.nc'