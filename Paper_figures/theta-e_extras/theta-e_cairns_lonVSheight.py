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
    ds = ds.sel(lat=slice(-17, -16.6), lon=slice(143, 152))
    
    # Extract all variables and combine along pressure dimension
    var_vars = {var: ds[var] for var in ds.data_vars if var.startswith(variable)}
    
    if var_vars:
        # Extract pressure levels from variable names (e.g., 'wa1000' -> 1000)
        pressure_levels = []
        var_data_arrays = []
        
        for var in sorted(var_vars.keys()):
            # Extract pressure from variable name
            pressure = int(var.replace(variable, ''))
            pressure_levels.append(pressure)
            var_data_arrays.append(var_vars[var])
        
        # Sort by pressure (descending: 1000 → 200 hPa) — OUTSIDE loop
        sorted_indices = np.argsort(pressure_levels)[::-1]
        pressure_levels = [pressure_levels[i] for i in sorted_indices]
        var_data_arrays = [var_data_arrays[i] for i in sorted_indices]
        
        # Combine into single DataArray with pressure dimension
        var_combined = xr.concat(var_data_arrays, dim='pressure')
        var_combined['pressure'] = pressure_levels
        
        # Create new dataset with just the combined variable
        ds = ds.drop_vars(list(var_vars.keys()))
        ds[variable] = var_combined
    
    return ds

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

    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        memory_limit='45GB',
        local_directory='/scratch/v46/ac9768/tmp'
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}", flush=True)

    Ps = [1000, 950, 925, 850, 700, 600, 500, 400, 300, 200]

    # define wind regimes
    barra_cairns = xr.open_dataset(
        "/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa-winds_cairns.nc",
        engine="h5netcdf", chunks="auto"
    ).sel(time=slice('1979-01-01T00:00:00.000000000', '2024-05-01T00:00:00.000000000'))
    lon_cairns = (144.27374 + 147.09222) / 2
    ne_cairns, se_cairns, sw_cairns, nw_cairns = wind_times(barra_cairns, lon_cairns)

    regimes = {
        'ne': ne_cairns,
        'se': se_cairns,
        'sw': sw_cairns,
        'nw': nw_cairns,
    }

    output_dir = '/home/563/ac9768/GBR/scripts/Paper_figures/theta-e_extras/'
    os.makedirs(output_dir, exist_ok=True)

    if var_name == 'RH':
        # Load ta and hus across pressure levels
        ta_fps, hus_fps = [], []
        for P in Ps:
            ta_fps.extend(list_barra_paths('ta' + str(P)))
            hus_fps.extend(list_barra_paths('hus' + str(P)))

        ta  = xr.open_mfdataset(ta_fps,  preprocess=partial(preprocess, variable='ta'),  engine='h5netcdf', parallel=True)
        hus = xr.open_mfdataset(hus_fps, preprocess=partial(preprocess, variable='hus'), engine='h5netcdf', parallel=True)

        results = {}
        for regime, times in regimes.items():
            ta_sel  = ta.sel(time=times)['ta']    # (time, pressure, lat, lon)
            hus_sel = hus.sel(time=times)['hus']

            # Saturation vapour pressure (Magnus-Tetens)
            T_C = ta_sel - 273.15
            e_s = 611.2 * np.exp((17.67 * T_C) / (T_C + 243.5))  # Pa

            # Actual vapour pressure from specific humidity
            # pressure coord is in hPa → convert to Pa
            p_Pa = ta_sel.pressure * 100.0
            e = (hus_sel * p_Pa) / (0.622 + 0.378 * hus_sel)

            rh = (e / e_s * 100.0).clip(0, 100)  # % — clip to physical range
            rh.name = 'RH'

            results[regime] = rh.mean(['lat', 'time'])
            print(f"Computed RH for {regime} regime.")

    else:
        var_fps = []
        for P in Ps:
            var_fps.extend(list_barra_paths(var_name + str(P)))

        var = xr.open_mfdataset(
            var_fps, preprocess=partial(preprocess, variable=var_name),
            engine='h5netcdf', parallel=True
        )

        results = {}
        for regime, times in regimes.items():
            results[regime] = var.sel(time=times).mean(['lat', 'time'])

    print("Saving results to NetCDF files...")
    paths = [f'{output_dir}{var_name}_{r}_regime_cairns_1979-2024_pbs.nc' for r in ['nw', 'ne', 'sw', 'se']]
    datasets = [results[r] if isinstance(results[r], xr.Dataset) else results[r].to_dataset(name=var_name) for r in ['nw', 'ne', 'sw', 'se']]
    xr.save_mfdataset(datasets, paths)

    client.close()
    cluster.close()

##in case errors, original code:
# if __name__ == '__main__':
#     var_name = sys.argv[1]

#     # n_workers = int(os.environ.get('PBS_NCPUS', 4))
#     cluster = LocalCluster(
#         n_workers=4,
#         threads_per_worker=1,
#         memory_limit='45GB',
#         local_directory='/scratch/v46/ac9768/tmp'
#     )
#     client = Client(cluster)
#     print(f"Dask dashboard: {client.dashboard_link}", flush=True)

#     # client, cluster, dask_tmp = setup_dask_client(mode="auto", workload_type="cpu")

#     Ps = [1000, 950, 925, 850, 700, 600, 500, 400, 300, 200] #hPa
#     var_fps = []
#     for P in Ps:
#         var_fps.extend(list_barra_paths(var_name+str(P)))

#     var = xr.open_mfdataset(var_fps,preprocess=partial(preprocess, variable=var_name),engine='h5netcdf',parallel=True)

#     # define wind regimes
#     barra_cairns = xr.open_dataset("/g/data/q90/ac9768/GBR/barra-2/barra-2_850hPa-winds_cairns.nc",
#                                    engine="h5netcdf",chunks="auto").sel(time=slice('1979-01-01T00:00:00.000000000','2024-05-01T00:00:00.000000000'))
#     lon_cairns = (144.27374 + 147.09222) / 2
#     ne_cairns, se_cairns, sw_cairns, nw_cairns = wind_times(barra_cairns, lon_cairns)

#     # compute var for regimes
#     regimes = { 'ne': ne_cairns, 
#                 'se': se_cairns,
#                 'sw': sw_cairns,
#                 'nw': nw_cairns,
#            }
#     results = {}
#     for regime, times in regimes.items():
#         results[regime] = var.sel(time=times).mean(['lat','time'])
        
#     # Save results to NetCDF files
#     output_dir = '/home/563/ac9768/GBR/scripts/Paper_figures/theta-e_extras/'
#     os.makedirs(output_dir, exist_ok=True)

#     for regime in ['nw', 'ne', 'sw', 'se']:
#         results[regime].to_netcdf(f'{output_dir}{var_name}_{regime}_regime_cairns_1979-2024_pbs.nc')
        
#     client.close() 
#     cluster.close()




    