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

# def wind_times(barra_regime_ds: xr.Dataset, longitude_center: float):    
#     winds = barra_regime_ds.wind_dir.compute()
#     ne = winds[(winds>=0)&(winds<=90)].time.values
#     se = winds[(winds>90)&(winds<=180)].time.values
#     sw = winds[(winds>180)&(winds<=270)].time.values
#     nw = winds[(winds>270)&(winds<=360)].time.values
#     return ne, se, sw, nw

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

    # Shift only the time coordinate labels, data values stay aligned
    # ds_lst = ds.assign_coords(time=ds.time + offset)

    
    # offset_hours = round(longitude_center * 4 / 60)
    # offset = np.timedelta64(offset_hours, 'h')

    ne = ne_lst - offset
    se = se_lst - offset
    sw = sw_lst - offset
    nw = nw_lst - offset

    return ne, se, sw, nw

def mask_radar_data(
    zarr_path: str,
    mask_path: str = '/home/563/ac9768/GBR/scripts/Paper_figures/bbANDlat_mask_cairns.nc',
    lat_min: float = -17.0,
    lat_max: float = -16.6,
    cairns: bool = False,
    bb_threshold: float = 1.2,
) -> xr.DataArray:
    """
    Load zarr rainrate, apply lat band selection and validity masks,
    return mean over y (Hovmoller slice).

    Args:
        zarr_path:    path to zarr store from save_rainrate_zarr
        mask_path:    path to netCDF file containing beam blockage mask for Cairns radar
        lat_min:      southern lat boundary for Hovmoller slice
        lat_max:      northern lat boundary for Hovmoller slice
        cairns:       if True, apply beam blockage mask
        bb_threshold: beam blockage mask threshold (mm/h mean)
    Returns:
        xr.DataArray: (time, x) mean rain rate along lat slice
    """
    ds = xr.open_zarr(zarr_path,chunks='auto').chunk({'time': 100, 'y': -1, 'x': -1})

    if cairns:
        mask = xr.open_dataset(mask_path).blockage_mask.drop_vars(['lat','lon'])
        maskx = ds.x>-145.25
    
        mean_rr = ds.rainrate.where((mask)&(maskx))
    else:
        mean_rr = ds.rainrate

    # Compute time-mean once for masking — load only rainrate
   ####time_mean = ds.rainrate.mean('time', skipna=True).compute()

    # Valid mask: pixels where measurements exist over time
    #####valid_mask = time_mean.notnull()

    # if cairns:
    #     # Beam blockage mask: pixels with suspiciously high mean rain rate
    #     bb_mask = time_mean >= bb_threshold
    #     mask = valid_mask & bb_mask
    # else:
    ####mask = valid_mask

    # Fill individual NaNs (no rain) with 0, then apply spatial mask
    #####rain = ds.rainrate.fillna(0).where(mask)

    return mean_rr.mean('y')

def get_radar_under_wind_regimes(radar_ds: xr.DataArray, barra_regimes_ds: xr.Dataset):
    """
    Select masked radar data for times under each wind regime.
    Uses vectorised time masking — no Python loop.
    """
    window_size   = np.timedelta64(30, 'm')
    barra_regime_ds = xr.DataArray(barra_regimes_ds)
    regime_times  = pd.DatetimeIndex(barra_regimes_ds).values.astype('datetime64[ns]')
    radar_times   = radar_ds.time.values.astype('datetime64[ns]')[:, None]

    in_window = np.any(
        (radar_times >= regime_times - window_size) &
        (radar_times <= regime_times + window_size),
        axis=1,
    )

    results = radar_ds.isel(time=in_window).drop_duplicates(dim='time')
    grpby   = results.groupby('time.hour').mean('time')
    return grpby # UTC

SITES = {
    'cairns': {
        'radar_id': '19',
        'zarr_path': '/scratch/v46/ac9768/radar/cairns_rainrate.zarr',
        'a': 85,
        'b': 1.35,
    },
    'willis': {
        'radar_id': '41',
        'zarr_path': '/scratch/v46/ac9768/radar/willis_rainrate.zarr',
        'a': 85,
        'b': 1.35,
    },
}


if __name__ == '__main__':
    site = sys.argv[1]
    cfg  = SITES[site]

    n_workers = int(os.environ.get('PBS_NCPUS', 4))
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, memory_limit='8GB')
    client  = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}", flush=True)

    # --- Define wind regimes ---
    
    barra_cairns = xr.open_dataset("/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa_winds_cairns.nc", engine="h5netcdf",chunks="auto")
    barra_willis = xr.open_dataset("/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa_winds_willis_island.nc", engine="h5netcdf",chunks="auto")

    lon_cairns = (144.27374 + 147.09222) / 2  # ≈ 145.68°E
    lon_willis = (148.55927 + 151.36993) / 2
    print('Define wind regimes by 12LST 850hPa BARRA-R2 wind direction', flush=True)
    ne_cairns, se_cairns, sw_cairns, nw_cairns = wind_times(barra_cairns, lon_cairns)
    ne_willis, se_willis, sw_willis, nw_willis = wind_times(barra_willis, lon_willis)
    
    wind_regimes = {
        'ne': ne_cairns,
        'se': se_cairns,
        'sw': sw_cairns,
        'nw': nw_cairns,
    }    # willis island regime times differ slightly to Cairns --> for comparison and analysis of propagation the Cairns regime time periods are used

    print('Mask radar data - account for beam blockage and NANs', flush=True)
    if site =='cairns':
        mask_rr = mask_radar_data(cfg['zarr_path'],cairns=True)    
    else:
        mask_rr = mask_radar_data(cfg['zarr_path'])    

    print('Create regime dictionary', flush=True)
    results = {}
    for regime, regime_periods in wind_regimes.items():
        results[regime] = get_radar_under_wind_regimes(mask_rr, regime_periods)

    print('Save data to zarr', flush=True)
    for regime, da in results.items():
        out = f"/scratch/v46/ac9768/radar/{site}_{regime}_hovmoller.zarr"
        da.to_zarr(out, mode='w')
        print(f"  saved {regime} → {out}", flush=True)
    
    print('Completed.', flush=True)
    client.close()
    cluster.close()