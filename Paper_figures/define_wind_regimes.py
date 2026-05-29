# import numpy as np
# import xarray as xr

# def wind_times(barra_regime_ds: xr.Dataset):    
#     winds = barra_regime_ds.wind_dir.compute()
#     ne = winds[(winds>=0)&(winds<=90)].time.values
#     se = winds[(winds>90)&(winds<=180)].time.values
#     sw = winds[(winds>180)&(winds<=270)].time.values
#     nw = winds[(winds>270)&(winds<=360)].time.values
#     return ne,se,sw,nw