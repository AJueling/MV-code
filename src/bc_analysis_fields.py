"""
contains 3 classes:
"""

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

from ba_analysis_dataarrays import AnalyzeDataArray


class AnalyzeField(AnalyzeDataArray):
    """ functions to analyze single 3D xr fields
    > trend map
    > std map
    > autocorrelation map
    
    """
    def __init__(self, field=None):
        """
        field .. xr.DataArray
        """
        self.field = field
        
        
    def determine_domain(self, A):
        """ guesses domain from shape of DataArray A """
        if np.shape(A)==(2400,3600):
            domain = 'ocn'
        elif np.shape(A)==(602, 900):
            domain = 'ocn_rect'
        elif np.shape(A) in [(384, 320), (320, 384)]:
            domain = 'ocn_low'
        elif np.shape(A)==(180, 360):
            domain = 'ocn_had'
        else:
            raise ValueError('not a known shape')
        return domain
    
    
    def load_SST_dt_field(self):
        return
    
    def load_SST_autocorrelation_maps(self):
        return
    
    def make_linear_trend_map(self, fn):
        """ """
        self.load_SST_data()
        return
    
        
    def make_standard_deviation_map(self, fn=None):
        ds = self.field.std(dim='time')
        if fn is not None:  ds.to_netcdf(fn)
        return ds
    
    
    def make_autocorrelation_map(self, fn=None):
        ds = self.autocorrelation(self.field)
        if fn is not None:  ds.to_netcdf(fn)
        return ds
    
    
    def regrid(self, field_to_regrid, field_unchanged, method='bilinear'):

        def rename_coords(A, back=False):
            domain = self.determine_domain(A)
            dll_coords = dll_coords_names(domain)
            if back==False:
                A = A.rename({dll_coords[1]: 'lat', dll_coords[2]: 'lon'})
            elif back==True:
                A = A.rename({'lat': dll_coords[1], 'lon': dll_coords[2]})
            return A
        
#         def add_lat_lon_b(A):
#             """ for cinservative method the arrays need grid bounds, e.g. `lat_b`"""
#             domain = self.determine_domain(A)
#             A = A.to_dataset()  # cannot add unused coordinates to DataArray
#             if domain in ['ocn_rect', 'ocn_had']:
#                 A['lat_b'] = np.linspace(-90,90,len(A['lat'])+1)
#                 A['lon_b'] = np.linspace(-180,180,len(A['lon'])+1)
#             elif domain=='ocn':
#                 lats, lons = generate_lats_lons(domain)
#             elif domain in ['ocn', 'ocn_low']:
#                 A['lat_b'] = np.linspace(-90,90,len(A['lat'])+1)
#                 A['lon_b'] = np.linspace(-180,180,len(A['lon'])+1)
#             return A

        def correct_low2had_boundary(A):
            """ corrects zonal boundary issues when transforming from ocn_low to ocn_had """
            A405 = (2*A.shift(longitude=1)+1*A.shift(longitude=-2))/3
            A395 = (1*A.shift(longitude=2)+2*A.shift(longitude=-1))/3
            A = xr.where(A.longitude==-40.5, A405, A)
            A = xr.where(A.longitude==-39.5, A395, A)
            return A
                    
        assert np.size(field_unchanged)<np.size(field_to_regrid)
        
        field_to_regrid = rename_coords(field_to_regrid)
        field_unchanged = rename_coords(field_unchanged)
        
        if method=='conservative':
            field_unchanged = add_lat_lon_b(field_unchanged)
            field_to_regrid = add_lat_lon_b(field_to_regrid)
        if self.determine_domain(field_to_regrid)=='ocn':
            lats, lons = generate_lats_lons('ocn')
            field_to_regrid['lat'].values = lats
            field_to_regrid['lon'].values = lons
        periodic = True
        if self.determine_domain(field_to_regrid)=='ocn_low':
            periodic = False
            field_to_regrid = field_to_regrid.transpose('nlon', 'nlat')
            field_to_regrid['lat'] = field_to_regrid['lat'].transpose()
            field_to_regrid['lon'] = field_to_regrid['lon'].transpose()
        
        regridder = xe.Regridder(field_to_regrid.to_dataset(),
                                 field_unchanged, method,
                                 reuse_weights=True, periodic=periodic)
        field_regridded = regridder(field_to_regrid)
        field_regridded = rename_coords(field_regridded, back=True)
        if self.determine_domain(field_to_regrid)=='ocn_low':
            field_regridded = correct_low2had_boundary(field_regridded)
            field_regridded = field_regridded.transpose()
        return field_regridded
    
    
    def regrid_to_lower_resolution(self, field_A, field_B, method=None):
        """ regrids either self.field or other_field to the lower of either resolutions
        returns the pair (self.field, other field) that are regridded keeping that order
        """
        for f in [field_A, field_B]:
            print(type(f))
            assert type(f)==xr.core.dataarray.DataArray
            assert len(np.shape(f))==2
            
        if np.size(field_A)==np.size(field_B):
            print('the fields are the same size already, no regridding necessary')
            return field_A, field_B
        
        elif np.size(field_A)>np.size(field_B):
            print('regridding field_A')
            field_to_regrid = field_A
            field_unchanged = field_B
            field_regridded = self.regrid(field_to_regrid, field_unchanged)
            return field_regridded, field_unchanged        
            
        elif np.size(field_A)<np.size(field_B):
            print('regridding field_B')
            field_to_regrid = field_B
            field_unchanged = field_A
            field_regridded = self.regrid(field_to_regrid, field_unchanged)
            return field_unchanged, field_regridded
        
    
    def spatial_correlation(self, field_A, field_B, method=None, selection=None):
        """ correlate two 2D fields """
        if np.shape(field_A)!=np.shape(field_B):  # have to regrid
            A, B = self.regrid_to_lower_resolution(field_A, field_B)
        else:
            A, B = field_A, field_B
        assert np.shape(A)==np.shape(B)
        domain = self.determine_domain(A)
        
        AREA = xr_AREA(domain)
        MASK = boolean_mask(domain=domain, mask_nr=0)
        if type(selection)==int:
            MASK = boolean_mask(domain=domain, mask_nr=selection)
        elif type(selection)==dict:
            MASK, AREA = MASK.sel(selection), AREA.sel(selection)
            A, B = A.sel(selection), B.sel(selection)
            
        D = np.any(np.array([np.isnan(A).values, np.isnan(B).values, (MASK==0).values]), axis=0)
        A = xr.where(D, np.nan, A   ).stack(z=('latitude', 'longitude')).dropna(dim='z')
        B = xr.where(D, np.nan, B   ).stack(z=('latitude', 'longitude')).dropna(dim='z')
        C = xr.where(D, np.nan, AREA).stack(z=('latitude', 'longitude')).dropna(dim='z')
        d = DescrStatsW(np.array([A.values, B.values]).T, weights=C)
        spatial_corr_coef = d.corrcoef[0,1]
        
        return spatial_corr_coef
        
    