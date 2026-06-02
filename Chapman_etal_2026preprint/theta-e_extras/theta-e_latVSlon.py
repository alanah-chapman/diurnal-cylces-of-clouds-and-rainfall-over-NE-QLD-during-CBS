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
from functools import partial
from distributed import Client, LocalCluster
# from dask_setup import setup_dask_client, MultiNodeConfig

warnings.filterwarnings('ignore', message='Sending large graph')

def list_barra_paths(var: str):
    """Create list of file paths for the BARRA-R2 variable.
    Using a base path, once the variable is chosen all file paths within the years and months are returned as a list.

    Args:
        var (str): String of radar ID number; towns = 73, cairns = "19", willis island = "41"
        temp_dir (str): Directory to store .nc files, default = "/scratch/v46/ac9768/radar_grndref/"
    Returns:
        list: List of extracted .nc file path strings
    """
    base = "/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/"+var+"/latest/"
    # analysis years
    year = np.arange(1979,2025,1)
    # analysis months
    month = ['01','02','03','04']
    list = sorted(
        f
        for y in year
        for m in month
        for f in glob.glob(base+var+f"_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{y}{m}-{y}{m}.nc")
    )

    if not list:
        raise FileNotFoundError("No files found — check the path.")
        
    list.sort()
    return list

def preprocess(ds, variable: str):
    return (
        ds[variable]
        .sel(lat=slice(-21.5,-14),lon=slice(143,152))
    )

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

if __name__ == '__main__':
    var_name = sys.argv[1]
    site     = sys.argv[2]  # 'cairns', 'towns', or 'willis'
    print(f"Processing variable '{var_name}' for site '{site}'...")

    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        memory_limit='45GB',
        local_directory='/scratch/v46/ac9768/tmp'
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}", flush=True)

    # define site configs and wind regimes
    site_config = {
        'cairns': {
            'barra_path': "/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa-winds_cairns.nc",
            'lon': (144.27374 + 147.09222) / 2,
        },
        'towns': {
            'barra_path': "/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa-winds_towns.nc",
            'lon': (145.12054 + 147.9812) / 2,
        },
        'willis': {
            'barra_path': "/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa-winds_willis.nc",
            'lon': (148.55927 + 151.36993) / 2,
        },
    }

    if site not in site_config:
        raise ValueError(f"Unknown site '{site}'. Choose from: {list(site_config.keys())}")
   
    cfg = site_config[site]
    barra = xr.open_dataset(
        cfg['barra_path'], engine="h5netcdf", chunks="auto"
    ).sel(time=slice('1979-01-01T00:00:00.000000000', '2024-05-01T00:00:00.000000000'))

    ne, se, sw, nw = wind_times(barra, cfg['lon'])

    regimes = {'ne': ne, 'se': se, 'sw': sw, 'nw': nw}

    output_dir = '/home/563/ac9768/GBR/scripts/Chapman_etal_2026preprint/theta-e_extras/'
    os.makedirs(output_dir, exist_ok=True)

    # create dataset for the variable 
    var_fps = list_barra_paths(var_name)

    var = xr.open_mfdataset(
        var_fps, preprocess=partial(preprocess, variable=var_name),
        engine='h5netcdf', parallel=True
    )

    results = {}
    for regime, times in regimes.items():
        results[regime] = var.sel(time=times).mean(['time'])

    print("Saving results to NetCDF files...")
    paths    = [f'{output_dir}{var_name}_{r}_regime_{site}_1979-2024_pbs.nc' for r in ['nw', 'ne', 'sw', 'se']]
    datasets = [results[r] if isinstance(results[r], xr.Dataset) else results[r].to_dataset(name=var_name) for r in ['nw', 'ne', 'sw', 'se']]
    xr.save_mfdataset(datasets, paths)

    # output_dir = '/home/563/ac9768/GBR/scripts/Chapman_etal_2026preprint/theta-e_extras/'
    # os.makedirs(output_dir, exist_ok=True)

    # for regime in ['sw', 'se']:
    #     results[regime].to_netcdf(f'{output_dir}{var_name}_{regime}_regime_cairns_1979-2024_pbs.nc')
            
    client.close()
    cluster.close()
