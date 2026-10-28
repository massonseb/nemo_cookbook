#!/usr/bin/env python
# coding: utf-8

import xarray as xr
from nemo_cookbook import NEMODataTree

def DefNEMODataTree(domain_filepath,gridT_filepath):
    ds_domain = xr.open_dataset(domain_filepath)
    ds_domain = ds_domain.drop_vars(["nav_lon", "nav_lat", "nav_lev"])
    ds_domain = ds_domain.rename_dims({'z': 'nav_lev'})
    
    ds_gridT = xr.open_dataset(gridT_filepath)
    
    datasets = {"parent": {"domain": ds_domain, "gridT": ds_gridT}}
    nemo = NEMODataTree.from_datasets(datasets=datasets, iperio=True, nftype="F") #, read_mask=True)
    return nemo
