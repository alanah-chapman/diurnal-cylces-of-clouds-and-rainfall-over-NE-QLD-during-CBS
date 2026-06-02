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
        ### townsville
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

    selected = radar_ds
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

    wind_regime_times = {
        'towns':  { 'clim': barra_towns.time.values},
        'cairns': {'clim': barra_cairns.time.values},
        'willis': {'clim': barra_willis.time.values},
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
    out_base = '/home/563/ac9768/GBR/scripts/Paper_figures/Fig11_diurnal_clim'

    for regime, regime_periods in wind_regimes.items():
        if rr_land is not None:
            print(f'  processing {regime} land...', flush=True)
            ### townsville
            # mean_land = get_radar_under_wind_regimes(rr_land.rainrate, regime_periods,willis=False)
                ### cairns
            mean_land = get_radar_under_wind_regimes(rr_land, regime_periods,willis=False)
            out = f"{out_base}/{site}_{regime}_land_mean_rainrate.zarr"
            mean_land.rename('rainrate').to_zarr(out, mode='w')
            print(f"  saved {regime} land_mean → {out}", flush=True)
            del mean_land
            
        print(f'  processing {regime} ocean...', flush=True)
        # if site=='willis':
        # mean_ocean = get_radar_under_wind_regimes(rr_ocean.rainrate, regime_periods, willis=True)
        # else:
        #     
        mean_ocean = get_radar_under_wind_regimes(rr_ocean, regime_periods, willis=False)
        out = f"{out_base}/{site}_{regime}_ocean_mean_rainrate.zarr"
        mean_ocean.rename('rainrate').to_zarr(out, mode='w')
        print(f"  saved {regime} ocean_mean → {out}", flush=True)
        del mean_ocean

    print('Completed.', flush=True)