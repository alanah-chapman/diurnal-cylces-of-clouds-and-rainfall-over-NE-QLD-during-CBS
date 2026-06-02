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

warnings.filterwarnings('ignore', message='Sending large graph')

def list_barra_paths(var: str):
    base = "/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/"+var+"/latest/"
    year = np.arange(1979,2025,1)
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
    
    var_vars = {var: ds[var] for var in ds.data_vars if var.startswith(variable)}
    
    if var_vars:
        pressure_levels = []
        var_data_arrays = []
        
        for var in sorted(var_vars.keys()):
            pressure = int(var.replace(variable, ''))
            pressure_levels.append(pressure)
            var_data_arrays.append(var_vars[var])
        
        sorted_indices = np.argsort(pressure_levels)[::-1]
        pressure_levels = [pressure_levels[i] for i in sorted_indices]
        var_data_arrays = [var_data_arrays[i] for i in sorted_indices]
        
        var_combined = xr.concat(var_data_arrays, dim='pressure')
        var_combined['pressure'] = pressure_levels
        
        ds = ds.drop_vars(list(var_vars.keys()))
        ds[variable] = var_combined
    
    return ds

def utc_to_lst_shift(ds, longitude_center: float):
    offset_hours = round(longitude_center * 4 / 60)
    offset = pd.Timedelta(hours=offset_hours)
    ds_lst = ds.assign_coords(time=ds.time + offset)
    return ds_lst

def wind_times(barra_regime_ds: xr.Dataset, longitude_center: float):
    ds_lst = utc_to_lst_shift(barra_regime_ds, longitude_center)
    winds = ds_lst.wind_dir.compute()
    winds_noon = winds.sel(time=winds.time.dt.hour == 12)

    ne_dates = winds_noon.time.values[(winds_noon.values >= 0)   & (winds_noon.values <= 90)]
    se_dates = winds_noon.time.values[(winds_noon.values > 90)   & (winds_noon.values <= 180)]
    sw_dates = winds_noon.time.values[(winds_noon.values > 180)  & (winds_noon.values <= 270)]
    nw_dates = winds_noon.time.values[(winds_noon.values > 270)  & (winds_noon.values <= 360)]

    def noon_to_all_hours(regime_noon_times):
        regime_dates = set(pd.Timestamp(t).date() for t in regime_noon_times)
        all_times = pd.DatetimeIndex(winds.time.values)
        mask = all_times.normalize().map(lambda d: d.date() in regime_dates)
        return winds.time.values[mask]

    ne_lst = noon_to_all_hours(ne_dates)
    se_lst = noon_to_all_hours(se_dates)
    sw_lst = noon_to_all_hours(sw_dates)
    nw_lst = noon_to_all_hours(nw_dates)

    offset_hours = round(longitude_center * 4 / 60)
    offset = pd.Timedelta(hours=offset_hours)

    ne = ne_lst - offset
    se = se_lst - offset
    sw = sw_lst - offset
    nw = nw_lst - offset

    return ne, se, sw, nw

if __name__ == '__main__':
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        memory_limit='45GB',
        local_directory='/scratch/v46/ac9768/tmp'
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}", flush=True)

    Ps = [1000, 950, 925, 850, 700, 600, 500, 400, 300, 200]

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

    # Load ta and hus across all pressure levels
    ta_fps, hus_fps = [], []
    for P in Ps:
        ta_fps.extend(list_barra_paths('ta' + str(P)))
        hus_fps.extend(list_barra_paths('hus' + str(P)))

    ta  = xr.open_mfdataset(ta_fps,  preprocess=partial(preprocess, variable='ta'),  engine='h5netcdf', parallel=True)
    hus = xr.open_mfdataset(hus_fps, preprocess=partial(preprocess, variable='hus'), engine='h5netcdf', parallel=True)

    for regime, times in regimes.items():
        print(f"Computing RH for {regime} regime...")

        ta_sel  = ta.sel(time=times)['ta']    # (time, pressure, lat, lon)
        hus_sel = hus.sel(time=times)['hus']

        # Saturation vapour pressure (Magnus-Tetens)
        T_C = ta_sel - 273.15
        e_s = 611.2 * np.exp((17.67 * T_C) / (T_C + 243.5))  # Pa

        # Actual vapour pressure from specific humidity
        p_Pa = ta_sel.pressure * 100.0
        e = (hus_sel * p_Pa) / (0.622 + 0.378 * hus_sel)

        rh = (e / e_s * 100.0).clip(0, 100)  # % — full (time, pressure, lat, lon)
        rh.name = 'RH'

        out_path = f'{output_dir}RH_{regime}_regime_cairns_1979-2024_pbs.nc'
        print(f"  Saving to {out_path} ...")
        rh.to_dataset(name='RH').to_netcdf(out_path)
        print(f"  Done.")

    client.close()
    cluster.close()