import os
import dask
import numpy as np
import xarray as xr

from paths import path_ocn_ctrl, path_ocn_rcp
from paths import path_atm_ctrl, path_atm_rcp
from paths import path_yrly_ctrl, path_yrly_rcp
from paths import rcpstr, spinup, CESM_filename
from filters import lowpass, chebychev, notch, deseasonalize  # old code imports those from here
from constants import abs_zero
from xr_integrate import xr_surf_mean, xr_zonal_mean
from xr_DataArrays import xr_AREA



class IterateOutputCESM:
    """ iterator over all CESM ctrl/rcp filenames
    automatically detects the last file
    
    example:
    >for year, month, filename in IterateOutputCESM('ocn', 'ctrl', 'monthly'):
    >    print(year, month, filename)
    """
    
    def __init__(self, domain, run, tavg, name=None):
        assert domain in ['ocn', 'ocn_rect', 'ocn_low', 'atm', 'ice']
        assert run in ['ctrl', 'rcp', 'lpd', 'lc1', 'lpi', 'pop', 'lr1', 'lr2', 'hq', 'ld', 'lq']
        assert tavg in ['monthly', 'yrly', 'daily']
        
        self.domain = domain
        self.run    = run
        self.tavg   = tavg
        self.stop   = False
        self.name   = name
        
        if run=='ctrl':
            if domain=='ice':
                self.year = 200          
            else:
                self.year =  40# 1            
        elif run=='rcp':  self.year = 2000
        elif run=='pop':  self.year =  125
        elif domain=='atm' and tavg=='monthly' and run=='lpd':
            self.year = 500
        elif run=='lpd':  self.year =  154
        elif run=='lc1':  self.year =    1
        elif run=='lr1':  self.year = 2000
        elif run=='lr2':  self.year = 2000
        elif run=='hq':   self.year = 2000
        elif run=='ld':   self.year = 2000
        elif run=='lq':   self.year = 2000
        elif run=='lpi':  
            if domain in ['ocn', 'ocn_low']:  self.year  = 1600
            if domain=='atm':                 self.year  = 2876
                
        if tavg=='daily':         
#             elif run=='rcp':  pass  # `self.year` set before; all years should have daily data
#             else: raise ValueError('no other daily data implemented')
            self.month = 1
            if domain=='ocn':
                assert name in ['SST', 'SSH'] 
                if name=='SST':
                    self.day = 32  # this signals CESM_filename to put out SST file
                    if run=='ctrl':  self.year = 249
                elif name=='SSH':
                    self.day =  1  # this returns truly daily files
                    if run=='ctrl':  self.year = 227
            elif domain=='atm':
                self.day =  1
        elif tavg=='monthly':
            self.month = 1
            self.day   = 0
        elif tavg=='yrly':
            self.month = 0
            self.day   = 0
            
    def file(self):
        if self.tavg=='daily':
            if self.domain in ['ocn', 'atm']:
                if self.run in ['ctrl', 'rcp']:
                    filename = CESM_filename(self.domain, self.run, self.year, self.month, d=self.day, name=None)     
                else:
                    raise ValueError('only `ctrl` and `rcp` daily run output available')
            
            else:
                raise ValueError('only `ocn` and `atm` daily data implemented')
        elif self.tavg=='monthly':
            filename = CESM_filename(self.domain, self.run, self.year, self.month)
            
        elif self.tavg=='yrly':  # files are not created by the model
            if self.domain in ['ocn', 'ocn_low', 'ocn_rect']:
                if self.name==None:
                    raise ValueError('must provide (variables part of) name for yrly file')
                else:
                    filename = CESM_filename(domain=self.domain, run=self.run, y=self.year, m=0, d=0, name=self.name)
            elif self.domain=='atm':
                if self.run in ['ctrl', 'rcp', 'lpi', 'hq', 'lc1', 'ld', 'lq', 'lr1', 'lpd']:
                    filename = CESM_filename(domain=self.domain, run=self.run, y=self.year, m=0, d=0, name=self.name)
                # elif self.run in ['lpd']:  # yrly files are written out already
                #     filename = CESM_filename(domain=self.domain, run=self.run, y=self.year, m=0, d=0, name=self.name)
#                     if self.name!=None:  print("name is ignored, as yearly files existed already")
#                     filename = CESM_filename(domain=self.domain, run=self.run, y=self.year, m=0, d=0, name=None)

        if os.path.exists(filename)==False:
            if (self.tavg=='daily' and self.run=='ctrl' and self.year<301) or \
               (self.tavg=='daily' and self.run=='rcp' and self.name=='SSH' and self.year<2101):
                print(f'{self.year} {self.month} {self.day}  skipped as it does not exist (non-fatal):\n{filename}')
            else:
                self.stop = True
        return filename
    
    def __iter__(self):
        return self
    
    def __len__(self):
        length = 0
        year   = self.year
        month  = self.month
        while os.path.exists(self.file()):
            if self.tavg=='monthly':
                self.month += 1
                length +=1
                if self.month==13:
                    self.month = 1
                    self.year +=1
            elif self.tavg=='yrly':
                self.year +=1
                length +=1
        return length

    def __next__(self):
        new_file = self.file()
        y = self.year
        m = self.month
        d = self.day
        if self.stop:
            raise StopIteration
        else:
            if self.tavg=='daily':
                if self.domain=='ocn' and self.name=='SST':  # monthly aggregated daily fields
                    if self.month==12:
                        self.month = 1
                        self.year +=1
                    else:
                        self.month += 1
                    return y, m, new_file
                
                elif self.domain=='ocn' and self.name=='SSH' or self.domain=='atm':  # daily files
                    # last day of the month (<Dec)
                    if (self.day==28 and self.month==2) or \
                       (self.day==30 and self.month in [2,4,6,9,11]) or \
                       (self.day==31 and self.month in [1,3,5,7,8,10]):
                        self.day    = 1
                        self.month += 1
                        
                        
                    elif self.day==31 and self.month==12:  # Dec. 31, last day of the year
                        self.day   = 1
                        self.month = 1
                        self.year += 1
                    else:  # any other day of the year
                        self.day += 1
                    return y, m, d, new_file
            
            elif self.tavg=='monthly':
                if self.run=='hq' and self.year==2039 and self.month==8:
                    self.month = 1
                    self.year = 2040
                    
                if self.month==12:
                    self.month = 1
                    self.year +=1
                else:
                    self.month += 1
                return y, m, new_file
            
            elif self.tavg=='yrly':
                self.year += 1
                return y, 0, new_file



def ncfile_list(domain, run, tavg, name=None):
    list_files = []
    for (y,m,s) in IterateOutputCESM(domain=domain, run=run, tavg=tavg, name=name):
        assert os.path.exists(s)
        list_files.append(s)
    return list_files