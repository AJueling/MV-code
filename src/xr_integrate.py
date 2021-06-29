import numpy as np
import xarray as xr

from constants import R_earth
from xr_DataArrays import dll_from_arb_da


def xr_int_global(da, AREA, DZ):
    """ global volume integral *[m^3] """
    (z, lat, lon) = dll_from_arb_da(da)
    return (da*AREA*DZ).sum(dim=[z, lat, lon])  # 0D



def xr_int_global_level(da, AREA, DZ):
    """ global volume integral *[m^3] """
    (z, lat, lon) = dll_from_arb_da(da)
    return (da*AREA*DZ).sum(dim=[lat, lon])  # 1D (z)



def xr_int_vertical(da, DZ):
    """ vertical integral *[m] """
    (z, lat, lon) = dll_from_arb_da(da)
    return (da*DZ).sum(dim=z)  # 2D (lat, lon)



def xr_int_zonal(da, HTN, LATS, AREA, DZ):
    """ integral along depth and zonal coordinates *[m^2] rectangular grid"""
    shape = np.shape(da)
    assert shape[-2:]==np.shape(HTN)[-2:]
    assert shape[-2:]==np.shape(DZ)[-2:]
    (z, lat, lon) = dll_from_arb_da(da)
    
    if shape[-1] in [900, 320]:  # rectangular `ocn_low` or `ocn_rect` grid
#         print(np.shape(da))
#         print(np.shape(HTN))
#         print(np.shape(DZ))
        int_zonal = (da*HTN*DZ).sum(dim=[z, lon])  # 1D (lat)
        
    elif shape[-1]==3600:        # tripolar grid
        int_vert  = xr_int_vertical(da, DZ)  # 2D
        int_zonal = xr_zonal_int_bins(int_vert, LATS, AREA)

    return int_zonal



def xr_int_zonal_level(da, HTN, LATS, AREA, DZ, dx=1):
    """ zonal integrals for each level *[m] rectangular grid"""
    (z, lat, lon) = dll_from_arb_da(da)
    shape = np.shape(da)
    assert shape[-2:]==np.shape(HTN)[-2:]
    assert shape[-2:]==np.shape(DZ)[-2:]
    
    if shape[-1] in [900, 320]:  # rectangular grid
        int_zonal_level = (da*HTN).sum(dim=[lon])  # 2D (z, lat)
        
    elif shape[-1]==3600:        # tripolar grid
        lat_bins, lat_centers, lat_width = lat_binning(dx)
        km = len(da[z])
        dz = DZ.max(dim=(lon,lat))

        # construct new xr DataArray
        # assert 'time' in da.coords
        lat_bin_name = f'TLAT_bins'
        if da.coords['time'].size==1:  # single time files
            array = np.zeros((km, len(lat_centers)))
            coords = {z: da.coords[z], lat_bin_name: lat_centers}
            int_zonal_level = xr.DataArray(data=array, coords=coords, dims=(z, lat_bin_name))
            for k in range(km):
                da_k = (da[k,:,:]*DZ[k,:,:]).drop('z_t')
                int_zonal_level[k,:] = xr_zonal_int_bins(da_k, LATS, AREA)/dz[k]
        else:
            array = np.zeros((da.coords['time'].size, km, len(lat_centers)))
            coords = {'time': da.coords['time'], z: da.coords[z], lat_bin_name: lat_centers}
            int_zonal_level = xr.DataArray(data=array, coords=coords, dims=('time', z, lat_bin_name))
            for k in range(km):
                da_k = (da[:,k,:,:]*DZ[k,:,:]).drop('z_t')
                int_zonal_level[:,k,:] = xr_zonal_int_bins(da_k, LATS, AREA)/dz[k]
        
    return int_zonal_level



def xr_zonal_int_bins(da, LATS, AREA, dx=1):
    """ integral over dx wide latitude bins
    integrates da with AREA, then divides by width of zonal strip dx
        
    input:
    da          .. 2D xr DataArray to be "zonally" integrated 
    LATS        .. 2D xr DataArray latitude values of each cell
    AREA        .. 2D xr DataArray
    dx          .. width of latitude bands in degrees
    lat_name    .. xa/AREA coordinate name of the latitude variable
    
    output:
    xa_zonal_int  .. 1D xr DataArray
    
    lat centers can be accessed through xa_zonal_int.coords[f'{lat_name}_bins']
    """
    
    assert type(da)==xr.core.dataarray.DataArray
    assert type(AREA)==xr.core.dataarray.DataArray
    assert np.shape(da)[-2:]==np.shape(AREA)
    
    (z, lat, lon) = dll_from_arb_da(da)
    
    lat_bins, lat_centers, lat_width = lat_binning(dx)
    
    da_new = da*AREA
    da_zonal_int = da_new.groupby_bins(LATS, lat_bins, labels=lat_centers).sum(dim=f'stacked_{lat}_{lon}')/lat_width
    
    return da_zonal_int



def lat_binning(dx):
    """ create latitude bins """
    lat_width = dx*R_earth*np.pi/180
    lat_bins = np.arange(-90, 90+dx, dx)
    lat_centers = np.arange(-90+dx/2, 90, dx)
    return lat_bins, lat_centers, lat_width



def xr_vol_int(xa, AREA, DZ, levels=False, zonal=False):
    """ volume integral of xarray *[m^3]

    input:
    xa                  .. 3D xr DataArray with data to be integrated
    AREA                .. 2D xr DataArray of cell areas
    DZ                  .. 3D xr DataArray of cell depths
    levels              .. option to output results for all level
    zonal               .. option to output zonal integrals

    output:
    integral            .. float integral
    int_levels          .. integrals of each level
    xa_zonal_int        .. 1D array of vert.+zonally integrated quantity
    xa_zonal_level_int  .. 2D (km, lat_bin) *[m^2] (integrated in depth and lon)
    xa_zonal_level_mean .. 2D (km, lat_bin) *[m^1] 
                           (weighted by bottom cell depth)
    """
    assert type(xa)==xr.core.dataarray.DataArray
    assert len(np.shape(xa))==3
    assert type(AREA)==xr.core.dataarray.DataArray
    assert type(DZ)==xr.core.dataarray.DataArray
    assert np.shape(AREA)==np.shape(xa)[-2:]
    assert np.shape(DZ)==np.shape(xa)[-3:]
    
    if zonal==True:
        dx = 1  # latitude bin width
        if np.shape(DZ)==(2,3,4):           # simple test case
            lat_name = 'y'
        elif np.shape(DZ)==(42,2400,3600):  # hires ocean
            lat_name = 'nlat'
        elif np.shape(DZ)==(30,384,576):    # atm fields
            lat_name = 'lat'
        else:
            raise ValueError('unknown shape: lat_name not implemented')
        assert lat_name in DZ.coords
    
    if levels==False:
        integral = np.sum(xa[:,:,:]*AREA[:,:]*DZ[:,:,:]).item()
        
        if zonal==False:  # just global integral
            return integral
        
        elif zonal==True:
            xa_vert = xr_int_along_axis(xa, DZ, 0)
            xa_zonal_int = xr_zonal_int(xa_vert, AREA, dx, lat_name)
            return integral, xa_zonal_int
        
    elif levels==True:
        km = len(xa[:,0,0])
        int_levels = np.zeros((km))
        for k in range(km):
            int_levels[k] = np.sum(xa[k,:,:]*AREA[:,:]*DZ[k,:,:]).item()
        integral = np.sum(int_levels)
        
        if zonal==False:
            return integral, int_levels
        
        if zonal==True:
            ONES = AREA.copy()
            ONES[:,:] = 1.
            for k in range(km):
                xa_zonal_int = xr_zonal_int(xa[k,:,:]*DZ[k,:,:], AREA, dx, lat_name)
                DZ_zonal_int = xr_zonal_int(DZ[k,:,:]          , ONES, dx, lat_name)
                if k==0:
                    xa_zonal_level_int  = np.zeros((km, len(xa_zonal_int)))
                    xa_zonal_level_mean = np.zeros((km, len(xa_zonal_int)))
                xa_zonal_level_int[k,:]  = xa_zonal_int
                xa_zonal_level_mean[k,:] = xa_zonal_int/DZ_zonal_int
            return integral, int_levels, xa_zonal_level_int, xa_zonal_level_mean

        
        
def xr_int_along_axis(xa, DZ, axis):
    """ integral of xr DataArray along a specific axis 
    
    input:
    xa   .. 3D xr DataArray of quantity to be integrated
    DZ   .. 3D xr DataArray of vertical cell extents [m]
    axis .. int axis to be integrated over
    
    output:
    int  .. 2D xr DataArray of integrated quantitity
    """
    assert type(axis)==np.dtype(int) or axis in xa.dims
    assert np.shape(xa)==np.shape(DZ)
    assert axis<=len(np.shape(xa))
    
    integral = np.sum(xa*DZ, axis=axis)
    
    return integral
        
    

def xr_vol_int_regional(xa, AREA, DZ, MASK):
    """ volumen integral with regional MASK
    
    input:
    xa, AREA, DZ         .. same as in 'xr_vol_int'
    MASK                 .. 2D xr DataArray of booleans with the same dimensions as xa 
    
    output:
    integral, int_levels .. same as in 'xr_vol_int'
    
    """
    assert type(xa)==xr.core.dataarray.DataArray
    assert type(AREA)==xr.core.dataarray.DataArray
    assert type(DZ)==xr.core.dataarray.DataArray
    assert np.shape(AREA)==np.shape(xa)[-2:]
    assert np.shape(DZ)==np.shape(xa)[-3:]
    assert np.dtype(MASK)==np.dtype('bool')

    # determine min/max i/j of masked region
    (imin, imax, jmin, jmax) = find_regional_coord_extent(MASK)
    
    xa_reg   = xa.where(MASK)[:,jmin:jmax+1,imin:imax+1]
    AREA_reg = AREA.where(MASK)[jmin:jmax+1,imin:imax+1]
    DZ_reg   = DZ.where(MASK)[:,jmin:jmax+1,imin:imax+1]
    
    integral, int_levels = xr_vol_int(xa_reg, AREA_reg, DZ_reg)
   
    return integral, int_levels



def find_regional_coord_extent(MASK):
    """ finds coordinates of a boolean mask
    
    input:
    MASK .. 2D xr DataArray of booleans
    
    output:
    (imin, imax, jmin, jmax) .. lon/lat extent of True area
    """
    assert type(MASK)==xr.core.dataarray.DataArray
    
    jmin = np.where(MASK)[0].min()
    jmax = np.where(MASK)[0].max()
    imin = np.where(MASK)[1].min()
    imax = np.where(MASK)[1].max()
    
    return (imin, imax, jmin, jmax)



def xr_vol_mean(xa, AREA, DZ):
    """ mean over quantity stored in xa
    
    input:
    xa   .. 3D xr DataArray of quantity
    AREA .. 2D xr DataArray of cell surface area
    DZ   .. 3D xr DataArray of cell depths
    
    output:
    mean .. (float)
    """
    assert type(xa)==xr.core.dataarray.DataArray
    assert type(DZ)==xr.core.dataarray.DataArray
    assert type(AREA)==xr.core.dataarray.DataArray
    assert np.shape(xa)==np.shape(DZ)
    assert np.shape(xa[0,:,:])==np.shape(AREA)    
    
    integral    = xr_vol_int(xa, AREA, DZ, levels=False, zonal=False)
    ONES        = xa.copy()
    ONES[:,:,:] = 1.
    volume      = xr_vol_int(ONES, AREA, DZ, levels=False, zonal=False)
    mean        = integral/volume
    
    return mean



def xr_surf_int(xa, AREA):
    """ surface integral of xarray DataArray *[m^2]
    
    input:
    xa       .. 2D xr DataArray
    AREA     .. 2D xr DataArray of surface area
    
    output:
    integral .. float integrated
    """
    assert type(xa)==xr.core.dataarray.DataArray
    assert type(AREA)==xr.core.dataarray.DataArray
    assert np.shape(xa)==np.shape(AREA)
    assert len(np.shape(xa))==2
    
    integral = np.sum(xa*AREA)
    
    return integral.item()



def xr_surf_mean(xa, AREA):
    """ mean over a surface *[1] 
    
    input:
    xa   .. 2D xr DataArray of quantity
    AREA .. 2D xr DataArrayof cell surfaces
    
    output:
    mean .. (float) mean of quantity in xa
    """
    assert type(xa)==xr.core.dataarray.DataArray
    assert type(AREA)==xr.core.dataarray.DataArray
    assert np.shape(xa)==np.shape(AREA)
    assert len(np.shape(xa))==2

    integral  = xr_surf_int(xa, AREA)
    ONES      = xa.copy()
    ONES[:,:] = 1.
    surface   = xr_surf_int(ONES, AREA)
    mean      = integral/surface
    
    return mean



def xr_zonal_mean(xa, AREA, dx, lat_name):
    """ area weighted mean over dx wide latitude bins
        
    input:
    xa            .. 2D xr DataArray
    AREA          .. 2D xr DataArray
    dx            .. width of latitude bands
    lat_name      .. xa/AREA coordinate name of the latitude variable
    
    output:
    xa_zonal_mean .. 1D xr DataArray
    """
    
    assert type(xa)==xr.core.dataarray.DataArray
    assert type(AREA)==xr.core.dataarray.DataArray
    assert len(np.shape(xa))==2
    assert np.shape(xa)==np.shape(AREA)
    assert dx>180/len(AREA[0,:])
    
    xa_zonal_int   = xr_zonal_int(xa, AREA, dx, lat_name)
    AREA_zonal_int = xr_zonal_int(AREA/AREA, AREA, dx, lat_name)
    
    xa_zonal_mean  = xa_zonal_int/AREA_zonal_int

    return xa_zonal_mean