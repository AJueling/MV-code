import os
import numpy as np
import xarray as xr

from paths import path_results, file_ex_ocn_ctrl, grid_file
from constants import imt, jmt
from read_binary import read_binary_2D_double
from xr_DataArrays import xr_DZ, dll_dims_names

# def generate_lats_lons(grid_file):
#     """
#     genrates lats and lons fields and shifts them so they are increasing 
#     (important for plotting with Basemap)
#     """
#     imt,jmt = 3600,2400
#     lats = read_binary_2D_double(grid_file,imt,jmt,1)
#     lons = read_binary_2D_double(grid_file,imt,jmt,2)
    
#     shift = np.zeros((jmt),dtype='int')
#     for j in range(jmt):
#         if j<jmt-1:  b = imt-np.argmin(lons[:,j])
#         if j==jmt-1: b = 900
#         lats[:,j] = 180/np.pi*np.roll(lats[:,j],b)
#         lons[:,j] = 180/np.pi*np.roll(lons[:,j],b)
#         shift[j]  = b
#     lons[imt-1,jmt-1] = 90.
    
#     return lats, lons, shift
    

def generate_lats_lons(domain):
    """
    generates lats and lons fields (no shift)
    """
    assert domain in ['ocn']
    lats = read_binary_2D_double(grid_file,imt,jmt,1)
    lons = read_binary_2D_double(grid_file,imt,jmt,2)
    lats = lats.T*180/np.pi  # rad to deg
    lons = np.roll(lons*180/np.pi+180, int(imt/2), axis=0).T  # rad to deg; same 
    return lats, lons 


def shift_field(field,shift):
    """
    shifts a 2D (imt,jmt) field
    """
    imt,jmt = 3600,2400
    shifted = np.zeros((imt,jmt))
    for j in range(jmt):
        shifted[:,j]  = np.roll(field[:,j],shift[j])
    return shifted


def create_dz_mean(domain):
    """ average depth [m] per level of """
    assert domain in ['ocn', 'ocn_low', 'ocn_rect']
    
    (z, lat, lon) = dll_dims_names(domain)
    
    fn = f'{path_results}/geometry/dz_mean_{domain}.nc'
    if os.path.exists(fn):
        dz_mean = xr.open_dataarray(fn, decode_times=False)
    else:
        DZT     = xr_DZ(domain)
        dz_mean = DZT.where(DZT>0).mean(dim=(lat, lon))
        dz_mean.to_netcdf(fn)
        
    return dz_mean


def create_tdepth(domain):
    """
    input:
    domain .. 'ocn'
    
    output:
    tdepth .. np array
    """
    assert domain=='ocn'
    
    fn = f'{path_results}/geometry/tdepth.csv'
    if os.path.exists(fn):
        tdepth = np.genfromtxt(fn, delimiter=',')
    else:
        ds = xr.open_dataset(file_ex_ocn_ctrl, decode_times=False)
        tdepth = ds.z_t.values/1e2
        np.savetxt(fn, tdepth, delimiter=',')
        
    return tdepth


def find_array_idx(array, val):
    """ index of nearest value in array to val 
    
    input:
    array .. array like
    value .. value to be approximately in array
    """
    idx = (np.abs(array - val)).argmin()
    return idx
    

def find_indices_near_coordinate(ds, lat, lon):
    """ finding the nlon/nlat indices for a given lon/lat position
    returns floats
    """
    distance = np.sqrt((ds.TLAT.where(ds.REGION_MASK>0)-lat)**2 + (ds.TLONG.where(ds.REGION_MASK>0)-lon)**2)
    nlons, nlats = np.meshgrid(ds.nlon, ds.nlat)
    nlons_, nlats_ = ds.TLONG.copy(), ds.TLAT.copy()
    nlons_.values, nlats_.values = nlons, nlats
    nlon_ = nlons_.where(distance==distance.min(), drop=True)
    nlat_ = nlats_.where(distance==distance.min(), drop=True)
    return (nlon_.values[0,0], nlat_.values[0,0])


def shift_ocn_low(da, back=False):
    """ shifts nlon between curvilinear [0,320] and [-160,160] frame
    the latter allows for a contiguous Atlantic """
    if back==False:
        return da.assign_coords(nlon=(da.nlon - 320*(da.nlon//160))).roll(nlon=160, roll_coords=True)
    else: 
        return da.assign_coords(nlon=(da.nlon+320)%320).roll(nlon=160, roll_coords=True)


def shift_had(da, back=False):
    """ shifts lons from [-180,180] to [0,360] to make Pacific contiguous """
    if back==True:
        return da.assign_coords(longitude=(da.longitude-360*(da.longitude//180))).roll(longitude=180, roll_coords=True)        
    else:
        return da.assign_coords(longitude=(da.longitude+360)%360).roll(longitude=180, roll_coords=True)