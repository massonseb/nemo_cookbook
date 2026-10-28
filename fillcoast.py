#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.ndimage import binary_dilation, convolve
from nemo_cookbook.applylbc import ApplyLbc

def FillCoast(xarr: xr.DataArray,
            iperio: bool = False,
            nftype: str = None,
            docopy: bool = False) -> xr.DataArray:

    # we will need to access the 8 neighbors so we add 1 halo and apply lbc
    arrxl = xarr.pad(i=1, j=1, mode="constant", constant_values=np.nan)
    arrxl = ApplyLbc(arrxl, iperio, nftype)

    # Ocean mask
    ocean = ~np.isnan(arrxl.data)
    
    # 3x3 neighborhood
    kernel = np.ones((3, 3), dtype=int)
    
    # Coastline land cells
    coast = binary_dilation(ocean, structure=kernel) & ~ocean
    
    # Sum of neighboring ocean values
    neighbor_sum = convolve(
        np.nan_to_num(arrxl.data, nan=0.0),
        kernel,
        mode="constant",
        cval=0.0,
    )
    
    # Number of neighboring ocean cells
    neighbor_count = convolve(
        ocean.astype(int),
        kernel,
        mode="constant",
        cval=0,
    )
    
    coast = coast[1:-1,1:-1]
    neighbor_sum = neighbor_sum[1:-1,1:-1]
    neighbor_count = neighbor_count[1:-1,1:-1]

    filled = xarr.copy(deep=docopy)
    filled.data[coast] = neighbor_sum[coast] / neighbor_count[coast]
    
    return filled
