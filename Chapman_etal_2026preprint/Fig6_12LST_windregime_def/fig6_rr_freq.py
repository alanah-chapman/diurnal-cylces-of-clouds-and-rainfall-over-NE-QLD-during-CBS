import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from distributed import Client, LocalCluster

warnings.filterwarnings('ignore', message='Sending large graph')

def utc_to_lst_shift(ds, longitude_center: float):
    offset_hours = round(longitude_center * 4 / 60)
    offset = pd.Timedelta(hours=offset_hours)
    return ds.assign_coords(time=ds.time + offset)


def wind_times(barra_regime_ds: xr.Dataset, longitude_center: float):
    ds_lst     = utc_to_lst_shift(barra_regime_ds, longitude_center)
    winds      = ds_lst.wind_dir.compute()
    winds_noon = winds.sel(time=winds.time.dt.hour == 12)

    ne_dates = winds_noon.time.values[(winds_noon.values >= 0)  & (winds_noon.values <= 90)]
    se_dates = winds_noon.time.values[(winds_noon.values > 90)  & (winds_noon.values <= 180)]
    sw_dates = winds_noon.time.values[(winds_noon.values > 180) & (winds_noon.values <= 270)]
    nw_dates = winds_noon.time.values[(winds_noon.values > 270) & (winds_noon.values <= 360)]

    def noon_to_all_hours(regime_noon_times):
        regime_dates = set(pd.Timestamp(t).date() for t in regime_noon_times)
        all_times    = pd.DatetimeIndex(winds.time.values)
        mask         = all_times.normalize().map(lambda d: d.date() in regime_dates)
        return winds.time.values[mask]

    ne_lst = noon_to_all_hours(ne_dates)
    se_lst = noon_to_all_hours(se_dates)
    sw_lst = noon_to_all_hours(sw_dates)
    nw_lst = noon_to_all_hours(nw_dates)

    offset = pd.Timedelta(hours=round(longitude_center * 4 / 60))
    return ne_lst - offset, se_lst - offset, sw_lst - offset, nw_lst - offset


def process_hour_freq(ds):
    """
    Sum the number of non raining points (value = 0) across x, y and time ignoring Nans. Sum the
    number of raining points. Compute those values. Calculate the total of valid data points (raining + non-raining). Calculate the frequency
    where the number of raining points is divided by the total valid points * 100.
    
    Parameters:
    - ds (str): Path to the netCDF file(s).
    
    Returns:
    - int: Frequency of raining points (%)
    """
    total_points = (ds.rainrate == 0).where(~ds.rainrate.isnull()).count(dim=['time','x', 'y'])
    number_of_non_raining_points = ds.rainrate.where(((ds.rainrate>=0)&(ds.rainrate<0.1))).count(dim=['x', 'y', 'time'])
    number_of_raining_points = (total_points) - (number_of_non_raining_points)
    freq = (number_of_raining_points / total_points) * 100
    return freq

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
        rr_ocean = ds.where(base_mask &  land_mask)   # ocean pixels only
        rr_land  = ds.where(base_mask &  ocean_mask)   # land pixels only
        #towns
        # rr_ocean = ds.rainrate.where(base_mask &  land_mask)   # ocean pixels only
        # rr_land  = ds.rainrate.where(base_mask &  ocean_mask)   # land pixels only
        return rr_ocean, rr_land
    else:
        rr_ocean = ds 

        return rr_ocean


def get_freq_under_wind_regimes(radar_ds: xr.DataArray, regime_times: np.ndarray):
    """
    Select radar data within ±30min of each regime timestamp,
    fill NaN (no rain) with 0, then compute hourly rain frequency.

    Args:
        radar_ds:     (time, y, x) masked rain rate DataArray
        regime_times: array of UTC datetime64 timestamps for the regime

    Returns:
        xr.DataArray: hourly rain frequency (%) with dim 'hour'
    """
    window_size = np.timedelta64(30, 'm')
    all_times   = radar_ds.time.values.astype('datetime64[ns]')[:, None]
    reg_times   = pd.DatetimeIndex(regime_times).values.astype('datetime64[ns]')

    # vectorised window matching — no Python loop over time
    in_window = np.any(
        (all_times >= reg_times - window_size) &
        (all_times <= reg_times + window_size),
        axis=1,
    )

    selected = radar_ds.isel(time=in_window).drop_duplicates(dim='time')

    # fill NaN → 0 (non-raining); NaN only where mask excluded pixels
    # selected = selected.fillna(0).where(selected.notnull() | selected.isnull())

    result = selected.drop_vars(['lat','lon']).groupby('time.hour')
    result = result.apply(process_hour_freq)
    return result 


SITES = {
    'towns': {
        'zarr_path':        '/scratch/v46/ac9768/radar/towns_rainrate.zarr',
        'bb_mask_path':     '/home/563/ac9768/GBR/scripts/Chapman_etal_2026preprint/radar_masks/bb_mask_towns.nc',
        'land_mask_path':   '/home/563/ac9768/GBR/scripts/Chapman_etal_2026preprint/radar_masks/bbANDland_mask_towns.nc',
        'ocean_mask_path':  '/home/563/ac9768/GBR/scripts/Chapman_etal_2026preprint/radar_masks/bbANDocean_mask_towns.nc',
        'barra_path':       '/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa_winds_townsville.nc',
        'lon_center':       146.5509,
    },
    'cairns': {
        'zarr_path':        '/scratch/v46/ac9768/radar/cairns_rainrate.zarr',
        'bb_mask_path':     '/home/563/ac9768/GBR/scripts/Chapman_etal_2026preprint/radar_masks/bb_mask_cairns.nc',
        'land_mask_path':   '/home/563/ac9768/GBR/scripts/Chapman_etal_2026preprint/radar_masks/bbANDland_mask_cairns.nc',
        'ocean_mask_path':  '/home/563/ac9768/GBR/scripts/Chapman_etal_2026preprint/radar_masks/bbANDocean_mask_cairns.nc',
        'barra_path':       '/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa_winds_cairns.nc',
        'lon_center':       145.683,
    },
    'willis': {
        'zarr_path':        '/scratch/v46/ac9768/radar/willis_rainrate.zarr',
        'bb_mask_path':     None,
        'land_mask_path':   None,
        'ocean_mask_path':  None,
        'barra_path':       '/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa_winds_willis_island.nc',
        'lon_center':       149.9646,
    },
}

if __name__ == '__main__':
    site = sys.argv[1]
    cfg  = SITES[site]

    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        memory_limit='45GB',
        local_directory='/scratch/v46/ac9768/tmp'
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}", flush=True)

    barra = xr.open_dataset(cfg['barra_path'], engine='h5netcdf', chunks='auto')
    print('Define wind regimes by 12LST 850hPa BARRA-R2 wind direction', flush=True)
    ne, se, sw, nw = wind_times(barra, cfg['lon_center'])
    wind_regimes = {'ne': ne, 'se': se, 'sw': sw, 'nw': nw}

    print('Mask radar data — beam blockage, x-extent, land/ocean', flush=True)
    if site in ['towns', 'cairns']:
        rr_ocean, rr_land = mask_radar_data(
            cfg['zarr_path'],
            land_mask_path  = cfg['land_mask_path'],
            ocean_mask_path = cfg['ocean_mask_path'],
            bb_mask_path    = cfg['bb_mask_path'],
            coastal          = True,
        )
    else:
        rr_ocean = mask_radar_data(cfg['zarr_path'], coastal=False)
        rr_land  = None

    print('Compute rain frequency per wind regime', flush=True)
    out_base = '/home/563/ac9768/GBR/scripts/Chapman_etal_2026preprint/Fig6_12LST_windregime_def'

    for regime, regime_times in wind_regimes.items():
        print(f'  processing {regime} ocean...', flush=True)
        ocean_freq = get_freq_under_wind_regimes(rr_ocean, regime_times)
        out = f"{out_base}/{site}_{regime}_ocean_freq.zarr"  # use site not hardcoded
        ocean_freq.rename('freq').to_zarr(out, mode='w')
        print(f"  saved {regime} ocean_freq → {out}", flush=True)
        del ocean_freq

        if rr_land is not None:
            print(f'  processing {regime} land...', flush=True)
            land_freq = get_freq_under_wind_regimes(rr_land, regime_times)
            out = f"{out_base}/{site}_{regime}_land_freq.zarr"
            land_freq.rename('freq').to_zarr(out, mode='w')
            print(f"  saved {regime} land_freq → {out}", flush=True)
            del land_freq

    print('Completed.', flush=True)
    client.close()
    cluster.close()
