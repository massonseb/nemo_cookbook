#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xr
import matplotlib.tri as tri
from scipy.ndimage import binary_dilation
from nemo_cookbook.lbc import lbc_nfd
from nemo_cookbook.applylbc import ApplyLbc

def DefTriang(x: np.ndarray,
              y: np.ndarray,
              mask: xr.DataArray[bool],
              iperio: bool = False,
              nftype: str = None,
              forland: bool = False,
              triang = None):  # -> tuple[np.ndarray, np.ndarray, np.ndarray, tri.Triangulation]:
    
    nx = mask.shape[1]
    # mask to 0/1 instead of False/True
    maskxl = mask.astype(int)
    # extend mask array of 1 point in each direction, pad it with 0
    maskxl = maskxl.pad(i=1, j=1, mode="constant",constant_values=0)
    # apply lbc
    maskxl = ApplyLbc(maskxl, iperio, nftype, 1)
    
    if forland:
        # Warning: xarray will try to match the x and y coordinates.
        mask2up = maskxl[1:-1,1:-1].data + maskxl[2:  ,2:].data
        mask2dn = maskxl[2:  ,1:-1].data + maskxl[1:-1,2:].data
    
    # set mask to 1 if one of its 8 neighbors is 1
    maskxl = binary_dilation(maskxl, structure=np.ones((3, 3))).astype(float)
    maskxl[maskxl== 0.] = np.nan
    
    msknan = x * y * np.ravel(maskxl[1:-1,1:-1])
    del maskxl
        
    mskok = ~np.isnan(msknan)
    #del msknan
    xx = x[mskok]
    yy = y[mskok]
    
    # build the triangulation if needed
    triang = tri.Triangulation(xx, yy)
    
    # triang is not used to plot the land/sea mask, just remove the triangle that are completly inland
    # triang.set_mask( np.ravel(~mask)[triang.triangles].all(axis=1) )
    if not forland:
        return mskok, triang

    # if triang is used plot the land/sea mask:
    #  --> we must potentially modify triang to properly draw the corners of the land/sea mask

    # preparatory stage

    # optimisation based on chatgpt suggestion...
    triangle_dict = {
        tuple(sorted(t)): i
        for i, t in enumerate(triang.triangles)
    }

    # list are faster to concatenate arrays
    trilist = [ ]
    trilist.append( triang.triangles.copy() )
    xlist = [ ]
    xlist.append( xx.copy() )
    ylist = [ ]
    ylist.append( yy.copy() )
    fullsz = xx.size
    
    revind = -np.ones(mask.size,dtype=int)
    revind[mskok] = np.arange(mskok.sum())
    
    allind = mask.copy(data = np.arange(mask.size).reshape(mask.shape))
    allind = allind.pad(i=1, j=1, mode="constant",constant_values=-1)
    allind = ApplyLbc(allind, iperio, nftype, 1)

    allind00 = np.ravel(allind.data[1:-1,1:-1])   # 
    allind01 = np.ravel(allind.data[1:-1,2:  ])   # i+1
    allind10 = np.ravel(allind.data[2:  ,1:-1])   # j+1
    allind11 = np.ravel(allind.data[2:  ,2:  ])   # i+1, j+1
    del allind
    
    allind00 = revind[allind00[mskok]]
    allind01 = revind[allind01[mskok]]
    allind10 = revind[allind10[mskok]]
    allind11 = revind[allind11[mskok]]

    mask2up = np.ravel(mask2up)[mskok]
    mask2dn = np.ravel(mask2dn)[mskok]

    # loop on the 4 cases that must be treated
    for ii in np.arange(4):
        
        if   ii == 0:   # bottom left or upper right mask is different from the 4 others
            #  xx xo ox oo
            #  ox xx oo xo
            sel = ( mask2up == 1 ) * np.isin( mask2dn, [0, 2] )
        elif ii == 1:   # bottom right or upper left mask is different from the 4 others
            #  xx ox xo oo
            #  xo xx oo ox
            sel = ( mask2dn == 1 ) * np.isin( mask2up, [0, 2] )
        elif ii == 2:   # upward diagonal is land, downward's one is ocean
            #  ox
            #  xo
            sel = ( mask2up == 2 ) * ( mask2dn == 0 )
        elif ii == 3:   # downward diagonal is land, upward's one is ocean
            #  xo
            #  ox
            sel = ( mask2up == 0 ) * ( mask2dn == 2 )
        
        ind00 = np.where(sel)[0]   # indexes of the selected points
        ind01 = allind01[ind00]
        ind10 = allind10[ind00]
        ind11 = allind11[ind00]
        
        # define the twin triangles that must not be used and must therefore be replaced if present
        # case 0: non working twin triangles share the downward diagonal
        # case 1: non working twin triangles share the upward diagonal
        # case 2 and 3: none of the triangles works
        if ii%2 == 0:   # twin triangles share the downward diagonal
            tri1 = np.column_stack( (ind00, ind01, ind10) )   # _\
            tri2 = np.column_stack( (ind11, ind10, ind01) )   # \|
        else:           # twin triangles share the upward diagonal
            tri1 = np.column_stack( (ind00, ind01, ind11) )   # /|
            tri2 = np.column_stack( (ind11, ind10, ind00) )   # |/
        if ii == 2:     # add the twin triangles share the upward diagonal
            tri1 = np.append(tri1, np.column_stack( (ind00, ind01, ind11) ), axis=0)   # /|
            tri2 = np.append(tri2, np.column_stack( (ind11, ind10, ind00) ), axis=0)   # |/ 
        if ii == 3:     # add the twin triangles share the downward diagonal
            tri1 = np.append(tri1, np.column_stack( (ind00, ind01, ind10) ), axis=0)   # _\
            tri2 = np.append(tri2, np.column_stack( (ind11, ind10, ind01) ), axis=0)   # \|

        if ii > 1:
            ind00 = np.append(ind00, ind00)
            ind01 = np.append(ind01, ind01)
            ind10 = np.append(ind10, ind10)
            ind11 = np.append(ind11, ind11)

        # do we use some of the twin triangles that are not working
        idx1 = [
            triangle_dict.get(tuple(sorted(t)), -1)
            for t in tri1
        ]
        idx1 = np.array(idx1)
        idx2 = [
            triangle_dict.get(tuple(sorted(t)), -1)
            for t in tri2
        ]
        idx2 = np.array(idx2)
        idx = np.where( (idx1 != -1) & (idx2 != -1) )[0]

        # replace the twin triangles that are not working:
        # case 0: the new twin triangles must share the upward diagonal
        # case 1: the new twin triangles must share the downward diagonal
        # case 2 and 3: must replace by 4 triangles that share the central point that must be added and defined to 0.5
        if idx.size > 0:
            ind00 = ind00[idx]
            ind01 = ind01[idx]
            ind10 = ind10[idx]
            ind11 = ind11[idx]
            idx1 = idx1[idx]
            idx2 = idx2[idx]

            if   ii == 0:   # replace by new twin triangles must share the upward diagonal
                trilist[0][idx1,...] = np.column_stack( (ind00, ind01, ind11) )   # /|
                trilist[0][idx2,...] = np.column_stack( (ind11, ind10, ind00) )   # |/ 
            elif ii == 1:   # replace by new twin triangles must share the downward diagonal
                trilist[0][idx1,...] = np.column_stack( (ind00, ind01, ind10) )   # _\
                trilist[0][idx2,...] = np.column_stack( (ind11, ind10, ind01) )   # \|
            else:
                indnew = fullsz + np.arange(ind00.size)
                fullsz = fullsz + ind00.size
                # redefine the twin triangles
                trilist[0][idx1,...] = np.column_stack( (ind00, ind01 , indnew) )
                trilist[0][idx2,...] = np.column_stack( (ind00, indnew, ind10) )
                # add 2 new ones (use list for perf issues)
                
                xlist.append( np.mean( [ xx[ind00],xx[ind01],xx[ind10],xx[ind11] ], axis=0 ) )
                ylist.append( np.mean( [ yy[ind00],yy[ind01],yy[ind10],yy[ind11] ], axis=0 ) )
                trilist.append( np.column_stack( (ind01, ind11, indnew) ) )
                trilist.append( np.column_stack( (ind11, ind10, indnew) ) )

    # list -> np.array
    xx = np.concatenate(xlist, axis=0)
    yy = np.concatenate(ylist, axis=0)

    # validate the new triangulation
    triang = tri.Triangulation(xx, yy, triangles = np.concatenate(trilist, axis=0) )

    # add new point (case 2 and 3) in the mask
    nadd = xx.size - mskok.sum()
    if nadd == 0:
        return np.ravel(mask)[mskok].astype(float), triang

    msk1d = np.ravel(~mask)[mskok]   # True is land
    msk1d = np.pad(msk1d, (0, nadd), constant_values=False)

    # remove the triangles that are competely in land
   # triang.set_mask( msk1d[triang.triangles].all(axis=1) )

    # Set the mask to 1 (oce), 0 (land) and 0.5 for the new point added for case 2 and 3
    msk1d = (~msk1d).astype(float)
    msk1d[-nadd:] = 0.5
    
    return msk1d, triang
