import os
import glob
import sys
import functools
import warnings
import dask
import numpy as np
import pandas as pd
import xarray as xr
import pyproj
import wradlib as wrl
from distributed import Client, LocalCluster

warnings.filterwarnings('ignore', message='Sending large graph')

def utc_to_lst_shift(ds, longitude_center: float):
    """
    Shift xarray data from UTC to Local Solar Time by offsetting the 
    time coordinate. Adds the LST offset to time labels directly —
    does NOT roll/wrap data values.

    Args:
        ds:               xarray Dataset or DataArray with a 'time' dimension.
        longitude_center: Representative longitude of the region (decimal degrees).

    Returns:
        Dataset/DataArray with time coordinate relabelled to LST.
    """
    offset_hours = round(longitude_center * 4 / 60)
    offset = pd.Timedelta(hours=offset_hours)

    # Shift only the time coordinate labels, data values stay aligned
    ds_lst = ds.assign_coords(time=ds.time + offset)
    return ds_lst

def wind_times(barra_regime_ds: xr.Dataset, longitude_center: float):
    """
    Classify wind direction at LST solar noon (hour=12), then assign
    all hours of that day to the same wind regime.

    Args:
        barra_regime_ds:  xarray Dataset with 'wind_dir' and 'time' dimension.
        longitude_center: Representative longitude for LST conversion.

    Returns:
        ne, se, sw, nw: arrays of datetime64 timestamps (all hours) 
                        belonging to each wind regime day.
    """
    # 1. Shift full dataset to LST
    ds_lst = utc_to_lst_shift(barra_regime_ds, longitude_center)

    # 2. Extract wind_dir and compute (needed for boolean indexing)
    winds = ds_lst.wind_dir.compute()

    # 3. Select only solar noon (LST hour=12) for regime classification
    winds_noon = winds.sel(time=winds.time.dt.hour == 12)

    # 4. Classify noon wind direction → get dates (not full timestamps)
    ne_dates = winds_noon.time.values[(winds_noon.values >= 0)   & (winds_noon.values <= 90)]
    se_dates = winds_noon.time.values[(winds_noon.values > 90)   & (winds_noon.values <= 180)]
    sw_dates = winds_noon.time.values[(winds_noon.values > 180)  & (winds_noon.values <= 270)]
    nw_dates = winds_noon.time.values[(winds_noon.values > 270)  & (winds_noon.values <= 360)]

    # 5. Convert noon timestamps → date-only for day matching
    def noon_to_all_hours(regime_noon_times):
        """Given noon timestamps, return all LST timestamps on those days."""
        regime_dates = set(
            pd.Timestamp(t).date() for t in regime_noon_times
        )
        all_times = pd.DatetimeIndex(winds.time.values)
        mask = all_times.normalize().map(lambda d: d.date() in regime_dates)
        return winds.time.values[mask]

    ne_lst = noon_to_all_hours(ne_dates)
    se_lst = noon_to_all_hours(se_dates)
    sw_lst = noon_to_all_hours(sw_dates)
    nw_lst = noon_to_all_hours(nw_dates)

    # 6. Convert LST timestamps back to UTC by subtracting the offset
    offset_hours = round(longitude_center * 4 / 60)
    offset = pd.Timedelta(hours=offset_hours)

    ne = ne_lst - offset
    se = se_lst - offset
    sw = sw_lst - offset
    nw = nw_lst - offset

    return ne, se, sw, nw

def mask_radar_data(
    zarr_path: str,
    bb_mask_path: str = None,
    land_mask_path: str = None,
    ocean_mask_path: str = None,
    coastal: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Load zarr rainrate, apply beam blockage, x-extent, and land/ocean masks.
    Returns two masked DataArrays — one for land, one for ocean.

    Args:
        zarr_path:      path to zarr store
        bb_mask_path:   path to beam blockage mask netCDF (Cairns only)
        land_mask_path: path to land/ocean mask netCDF (True = ocean)
        ocean_mask_path: path to ocean mask netCDF (True = land)
        coastal:        if True, apply coastal mask
    Returns:
        rr_ocean: (time, y, x) rain rate masked to ocean pixels
        rr_land:  (time, y, x) rain rate masked to land pixels
    """
    ds = xr.open_zarr(zarr_path, chunks='auto').chunk({'time': 100, 'y': -1, 'x': -1})

    # --- base mask: beam blockage + x-extent (Townsville and Cairns) ---
    if coastal:
        try:
            bb_mask = xr.open_dataset(bb_mask_path).blockage_mask.drop_vars(['lat', 'lon'])
        except:
            bb_mask = xr.open_dataset(bb_mask_path)
        # maskx   = ds.x > -145.75 # for Cairns
        base_mask = bb_mask #& maskx
        # --- land/ocean mask ---
        land_mask = xr.open_dataset(land_mask_path)['__xarray_dataarray_variable__']  # True = ocean
        ocean_mask = xr.open_dataset(ocean_mask_path)['__xarray_dataarray_variable__']  # True = land

        # --- apply combined masks ---
        rr_ocean = ds.rainrate.where(base_mask &  land_mask)   # ocean pixels only
        rr_land  = ds.rainrate.where(base_mask &  ocean_mask)   # land pixels only

        return rr_ocean, rr_land
    else:
        rr_ocean = ds.rainrate 

        return rr_ocean

def get_radar_under_wind_regimes(radar_ds: xr.DataArray, barra_regimes_ds: xr.Dataset,willis: bool = False,):
    """
    Select masked radar data for times under each wind regime.
    Returns hourly mean and IQR (25th, 75th percentile) across time and space.
    """
    window_size  = np.timedelta64(30, 'm')
    barra_regime_ds = xr.DataArray(barra_regimes_ds)
    regime_times = pd.DatetimeIndex(barra_regimes_ds).values.astype('datetime64[ns]')
    radar_times  = radar_ds.time.values.astype('datetime64[ns]')[:, None]

    in_window = np.any(
        (radar_times >= regime_times - window_size) &
        (radar_times <= regime_times + window_size),
        axis=1,
    )

    selected = radar_ds.isel(time=in_window).drop_duplicates(dim='time')
    drop_vars = [v for v in ['lat', 'lon'] if v in selected.coords]
    if drop_vars:
        selected = selected.drop_vars(drop_vars)
        
    hours     = np.arange(0, 24)
    mean_list = []

    for h in hours:
        # load one hour at a time — never load full array
        hour_mask = (selected.time.dt.hour == h).values
        hour_data = selected.isel(time=hour_mask).load()
        mean_list.append(float(hour_data.mean(skipna=True)))
        del hour_data
        import gc; gc.collect()
        print(f'    hour {h:02d} done', flush=True)

    return xr.DataArray(mean_list, dims=['hour'], coords={'hour': hours})
    # # else:
    # selected = radar_ds.isel(time=in_window).drop_duplicates(dim='time').compute()
    
    # # hourly mean over time, x, y
    # mean = selected.groupby('time.hour').mean(['time', 'x', 'y'])

    # return mean

SITES = {
    'towns': {
        'radar_id': '73',
        'zarr_path':        '/scratch/v46/ac9768/radar/towns_rainrate.zarr',
        'bb_mask_path':     '/home/563/ac9768/GBR/scripts/Paper_figures/radar_masks/bb_mask_towns.nc',
        'land_mask_path':   '/home/563/ac9768/GBR/scripts/Paper_figures/radar_masks/bbANDland_mask_towns.nc',
        'ocean_mask_path':  '/home/563/ac9768/GBR/scripts/Paper_figures/radar_masks/bbANDocean_mask_towns.nc',
        'a': 125,
        'b': 1.3,
    },
    'cairns': {
        'radar_id': '19',
        'zarr_path': '/scratch/v46/ac9768/radar/cairns_rainrate.zarr',
        'bb_mask_path': '/home/563/ac9768/GBR/scripts/Paper_figures/radar_masks/bb_mask_cairns.nc', 
        'land_mask_path': '/home/563/ac9768/GBR/scripts/Paper_figures/radar_masks/bbANDland_mask_cairns.nc',
        'ocean_mask_path': '/home/563/ac9768/GBR/scripts/Paper_figures/radar_masks/bbANDocean_mask_cairns.nc',
        'a': 85,
        'b': 1.35,
    },
    'willis': {
        'radar_id': '41',
        'zarr_path': '/scratch/v46/ac9768/radar/willis_rainrate.zarr',
        'bb_mask_path': None,
        'land_mask_path': None,
        'ocean_mask_path': None,
        'a': 85,
        'b': 1.35,
    },
}

if __name__ == '__main__':
    site = sys.argv[1]
    cfg  = SITES[site]
    # if site in ['towns', 'cairns']:
    #     n_workers = int(os.environ.get('PBS_NCPUS', 4))
    #     cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, memory_limit='8GB')
    #     cluster = LocalCluster(
    #         n_workers=4,
    #         threads_per_worker=1,
    #         memory_limit='45GB',
    #         local_directory='/scratch/v46/ac9768/tmp'
    #     )
        # client = Client(cluster)
        # print(f"Dask dashboard: {client.dashboard_link}", flush=True)
    # else:
    dask.config.set(scheduler='synchronous')  # single-threaded, no workers
    print('Running single-threaded (no dask cluster)', flush=True)

    # --- wind regimes ---
    barra_towns  = xr.open_dataset("/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa_winds_townsville.nc",   engine="h5netcdf", chunks="auto")
    barra_cairns = xr.open_dataset("/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa_winds_cairns.nc",       engine="h5netcdf", chunks="auto")
    barra_willis = xr.open_dataset("/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa_winds_willis_island.nc",engine="h5netcdf", chunks="auto")

    lon_towns  = 146.5509
    lon_cairns = (144.27374 + 147.09222) / 2
    lon_willis = (148.55927 + 151.36993) / 2

    print('Define wind regimes by 12LST 850hPa BARRA-R2 wind direction', flush=True)
    ne_towns,  se_towns,  sw_towns,  nw_towns  = wind_times(barra_towns,  lon_towns)
    ne_cairns, se_cairns, sw_cairns, nw_cairns = wind_times(barra_cairns, lon_cairns)
    ne_willis, se_willis, sw_willis, nw_willis = wind_times(barra_willis, lon_willis)

    # wind_regime_times = {
    #     'towns':  {'ne': ne_towns,  'se': se_towns,  'sw': sw_towns,  'nw': nw_towns},
    #     'cairns': {'ne': ne_cairns, 'se': se_cairns, 'sw': sw_cairns, 'nw': nw_cairns},
    #     'willis': {'ne': ne_willis, 'se': se_willis, 'sw': sw_willis, 'nw': nw_willis},
    # }
    wind_regime_times = {
        'towns':  { 'se': se_towns},
        'cairns': {'ne': ne_cairns, 'se': se_cairns, 'sw': sw_cairns, 'nw': nw_cairns},
        'willis': {'ne': ne_willis, 'se': se_willis, 'sw': sw_willis, 'nw': nw_willis},
    }
    wind_regimes = wind_regime_times[site]

    # --- masks ---
    print('Mask radar data — beam blockage, x-extent, land/ocean', flush=True)
    if site in ['towns', 'cairns']:
        rr_ocean, rr_land = mask_radar_data(
            cfg['zarr_path'],
            bb_mask_path    = cfg['bb_mask_path'],
            land_mask_path  = cfg['land_mask_path'],
            ocean_mask_path = cfg['ocean_mask_path'],
            coastal          = True, 
        )
        rr_land = rr_land  # explicit for clarity
    else:
        rr_ocean = mask_radar_data(cfg['zarr_path'], coastal=False)
        rr_land  = None

    # --- compute mean per regime, save immediately to avoid memory buildup ---
    print('Create regime dictionary', flush=True)
    out_base = '/home/563/ac9768/GBR/scripts/Paper_figures/Fig6_12LST_windregime_def'

    for regime, regime_periods in wind_regimes.items():
        if rr_land is not None:
            print(f'  processing {regime} land...', flush=True)
            mean_land = get_radar_under_wind_regimes(rr_land.rainrate, regime_periods,willis=False)
            out = f"{out_base}/{site}_{regime}_land_mean_rainrate.zarr"
            mean_land.rename('rainrate').to_zarr(out, mode='w')
            print(f"  saved {regime} land_mean → {out}", flush=True)
            del mean_land
            
        print(f'  processing {regime} ocean...', flush=True)
        # if site=='willis':
        mean_ocean = get_radar_under_wind_regimes(rr_ocean.rainrate, regime_periods, willis=True)
        # else:
        #     mean_ocean = get_radar_under_wind_regimes(rr_ocean, regime_periods, willis=False)
        out = f"{out_base}/{site}_{regime}_ocean_mean_rainrate.zarr"
        mean_ocean.rename('rainrate').to_zarr(out, mode='w')
        print(f"  saved {regime} ocean_mean → {out}", flush=True)
        del mean_ocean

        if rr_land is not None:
            print(f'  processing {regime} land...', flush=True)
            mean_land = get_radar_under_wind_regimes(rr_land.rainrate, regime_periods,willis=False)
            out = f"{out_base}/{site}_{regime}_land_mean_rainrate.zarr"
            mean_land.rename('rainrate').to_zarr(out, mode='w')
            print(f"  saved {regime} land_mean → {out}", flush=True)
            del mean_land

    print('Completed.', flush=True)
    # if site in ['towns', 'cairns']:
    #     client.close()
    #     cluster.close()
    # else:
    #     pass
# if __name__ == '__main__':
#     site = sys.argv[1]
#     cfg  = SITES[site]

#     # n_workers = int(os.environ.get('PBS_NCPUS', 4))
#     # cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, memory_limit='8GB')
#     cluster = LocalCluster(
#         n_workers=4,
#         threads_per_worker=1,
#         memory_limit='45GB',   # 4 × 45GB = 180GB
#         local_directory='/scratch/v46/ac9768/tmp'
#     )
#     client  = Client(cluster)
#     print(f"Dask dashboard: {client.dashboard_link}", flush=True)

#     # --- Define wind regimes ---
#     barra_towns = xr.open_dataset("/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa_winds_townsville.nc", engine="h5netcdf",chunks="auto")
#     barra_cairns = xr.open_dataset("/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa_winds_cairns.nc", engine="h5netcdf",chunks="auto")
#     barra_willis = xr.open_dataset("/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa_winds_willis_island.nc", engine="h5netcdf",chunks="auto")
#     lon_towns = 146.5509
#     lon_cairns = (144.27374 + 147.09222) / 2  
#     lon_willis = (148.55927 + 151.36993) / 2
#     print('Define wind regimes by 12LST 850hPa BARRA-R2 wind direction', flush=True)
#     ne_towns, se_towns, sw_towns, nw_towns = wind_times(barra_towns, lon_towns)
#     ne_cairns, se_cairns, sw_cairns, nw_cairns = wind_times(barra_cairns, lon_cairns)
#     ne_willis, se_willis, sw_willis, nw_willis = wind_times(barra_willis, lon_willis)
    
#     wind_regime_times = {
#         'towns':  {'ne': ne_towns,  'se': se_towns,  'sw': sw_towns,  'nw': nw_towns},
#         'cairns': {'ne': ne_cairns, 'se': se_cairns, 'sw': sw_cairns, 'nw': nw_cairns},
#         'willis': {'ne': ne_willis, 'se': se_willis, 'sw': sw_willis, 'nw': nw_willis},
#     }
    
#     wind_regimes = wind_regime_times[site]
#     print('Mask radar data — beam blockage, x-extent, land/ocean', flush=True)
#     if site in ['towns', 'cairns']:
#         rr_ocean, rr_land = mask_radar_data(
#             cfg['zarr_path'],
#             land_mask_path      = cfg['land_mask_path'],
#             ocean_mask_path = cfg['ocean_mask_path'],
#             cairns         = (site == 'cairns'),
#         )
#     else:
#         rr_ocean = mask_radar_data(cfg['zarr_path'], cairns=False) 

#     print('Create regime dictionary', flush=True)
#     results = {}
#     for regime, regime_periods in wind_regimes.items():
#         print(f'  processing {regime}...', flush=True)
#         if site in ['towns', 'cairns']:
#             mean_ocean  = get_radar_under_wind_regimes(rr_ocean, regime_periods) 
#             mean_land = get_radar_under_wind_regimes(rr_land,  regime_periods)
#             results[regime] = {
#                 'ocean_mean': mean_ocean, 
#                 'land_mean':  mean_land,  
#             }
#         else:
#             mean_ocean = get_radar_under_wind_regimes(rr_ocean, regime_periods)
#             results[regime] = {
#                 'ocean_mean': mean_ocean, 
#             }

#     print('Save data to zarr', flush=True)
#     for regime, stats in results.items():
#         for stat, da in stats.items():
#             out = f"/home/563/ac9768/GBR/scripts/Paper_figures/Fig6_12LST_windregime_def/{site}_{regime}_{stat}_rainrate.zarr"
#             da.to_zarr(out, mode='w')
#             print(f"  saved {regime} {stat} → {out}", flush=True)

#     print('Completed.', flush=True)
#     client.close()
#     cluster.close()
