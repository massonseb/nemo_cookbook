#!/usr/bin/env python
# coding: utf-8

import xarray as xr
from scipy.ndimage import binary_dilation
from nemo_cookbook.lbc import lbc_nfd

def ApplyLbc(xarr: xr.DataArray,
            iperio: bool = False,
            nftype: str = None,
            ihls: int = 1) -> xr.DataArray:

    ndimorg = xarr.ndim
    
    if ndimorg == 2:
        xarr = xarr.expand_dims(nav_lev=[1])
        
    dimname0 = xarr.dims[0]
    dimname1 = xarr.dims[1]
    dimname2 = xarr.dims[2]
    
    xarr = xarr.rename({xarr.dims[2]:"x", xarr.dims[1]:"y", xarr.dims[0]:"nav_lev"})
    
    if iperio:
        xarr[0,:, 0] = xarr[0,:,-2]   # 1st = before last
        xarr[0,:,-1] = xarr[0,:, 1]   # last = 2nd
        
    if nftype is not None:
        xarr = lbc_nfd(c_NFtype=nftype, cd_nat="T", ihls=1, ptab=xarr, psgn=1)

    xarr = xarr.rename({xarr.dims[2]:dimname2, xarr.dims[1]:dimname1, xarr.dims[0]:dimname0})
        
    if ndimorg == 2:
        xarr = xarr[0,:,:]

    return xarr
