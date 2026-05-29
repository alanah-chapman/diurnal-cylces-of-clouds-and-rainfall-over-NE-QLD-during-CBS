import os
import glob
import sys
import functools
import warnings
import dask
import numpy as np
import xarray as xr
import pyproj
import wradlib as wrl
from distributed import Client, LocalCluster

warnings.filterwarnings('ignore', message='Sending large graph')

def path_to_radar_ds(radar_site_no: str):
    """Create list of file paths for the chosen radar ID.
    Walks the full prcp-crate directory and returns all .zip-contained .nc files.

    Args:
        radar_site_no (str): String of radar ID number; cairns = "19", willis island = "41"
    Returns:
        list: List of extracted .nc file path strings
    """
    files = sorted(glob.glob(f"/scratch/v46/ac9768/radar_grndref/radar_{radar_site_no}_*/*.nc"))
    return files

def preprocess_convert_dbz_to_rainrate(file_list: list, a: float, b: float, lat_slice: bool=True) -> xr.Dataset:
    """
    Convert reflectivity (dBZ) to rain rate (mm/h) using the Z-R relationship:
        Z = a * R^b  →  R = (Z / a)^(1/b)
    Args:
        file_list: list of .nc file paths
        a:         Z-R coefficient
        b:         Z-R exponent
    Returns:
        xarray Dataset with 'rainrate' variable (mm/h)
    """
    # Compute projection grid once from reference file
    with xr.open_dataset(file_list[0]) as ref:
        x_grid, y_grid = np.meshgrid(ref.x.data, ref.y.data)
        proj = pyproj.Proj(
            proj='aea',
            lat_1=ref.proj.standard_parallel[0],
            lat_2=ref.proj.standard_parallel[1],
            lat_0=ref.proj.latitude_of_projection_origin,
            lon_0=ref.proj.longitude_of_central_meridian,
            x_0=0, y_0=0
        )
        lon_grid, lat_grid = proj(x_grid * 1000, y_grid * 1000, inverse=True)

    def _preprocess(ds, a, b, lat_grid, lon_grid, lat_slice):
        lat_da   = xr.DataArray(lat_grid, dims=['y', 'x'])
        lon_da   = xr.DataArray(lon_grid, dims=['y', 'x'])
        if lat_slice is True:
            lat_band = (lat_grid > -17) & (lat_grid < -16.6)
            dBZ = ds.reflectivity.where(lat_band)
        else:
            dBZ = ds.reflectivity
        Z   = wrl.trafo.idecibel(dBZ)
        R   = wrl.zr.z_to_r(Z, a=a, b=b)

        ds = ds.assign(lat=lat_da, lon=lon_da, rainrate=R)
        ds['rainrate'].attrs.update({'units': 'mm/h', 'long_name': 'rain rate'})

        if 'valid_time' in ds:
            ds = ds.rename({'valid_time': 'time'}).set_coords('time')
            if 'time' not in ds.dims:
                ds = ds.expand_dims('time')

        return ds

    return xr.open_mfdataset(
        file_list,
        preprocess=functools.partial(_preprocess, a=a, b=b,
                                     lat_grid=lat_grid,
                                     lon_grid=lon_grid,
                                     lat_slice=lat_slice),
        parallel=True,
        combine="by_coords",
    )
    
SITES = {
    'towns': {
        'radar_id': '73',
        'zarr_path': '/scratch/v46/ac9768/radar/towns_rainrate.zarr',
        'a': 125,
        'b': 1.3,
    },
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

def save_rainrate_zarr_batched(file_list, out_path, a, b, batch_size=1000):
    # Check how many files already written if zarr exists
    start_batch = 0
    if os.path.exists(out_path):
        existing = xr.open_zarr(out_path)
        n_existing = len(existing.time)
        start_batch = n_existing // batch_size
        print(f"Resuming from batch {start_batch+1} ({n_existing} timesteps already saved)", flush=True)

    for i, batch_start in enumerate(range(0, len(file_list), batch_size)):
        if i < start_batch:
            continue  # skip already completed batches
        batch = file_list[batch_start:batch_start + batch_size]
        n_batches = len(file_list) // batch_size + 1
        print(f"  batch {i+1}/{n_batches} — files {batch_start}:{batch_start+len(batch)}", flush=True)

        ds = preprocess_convert_dbz_to_rainrate(batch, a=a, b=b, lat_slice=False) ### lat_slice=False returns full radar domain
        ds_out = ds[['rainrate', 'lat', 'lon']].set_coords(['lat', 'lon'])

        if i == 0:
            ds_out.to_zarr(out_path, mode='w')
        else:
            ds_out.to_zarr(out_path, mode='a', append_dim='time')

        print(f"  batch {i+1} done", flush=True)


if __name__ == '__main__':
    site = sys.argv[1]
    cfg  = SITES[site]

    n_workers = int(os.environ.get('PBS_NCPUS', 4))
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, memory_limit=None)
    client  = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}", flush=True)

    files = path_to_radar_ds(cfg['radar_id'])
    print(f"{site}: {len(files)} files found", flush=True)

    # Always call — resume logic is handled inside save_rainrate_zarr_batched
    save_rainrate_zarr_batched(files, cfg['zarr_path'], cfg['a'], cfg['b'], batch_size=1000)
    print(f"{site} zarr complete: {cfg['zarr_path']}", flush=True)

    client.close()
    cluster.close()