#!/bin/bash
#PBS -N fig6_mean_rr
#PBS -P v46
#PBS -q normal
#PBS -l ncpus=48
#PBS -l mem=192GB
#PBS -l jobfs=10GB
#PBS -l walltime=06:00:00
#PBS -l wd
#PBS -l storage=gdata/q90+scratch/v46+gdata/xp65
#PBS -j oe
#PBS -o /home/563/ac9768/GBR/scripts/Paper_figures/logs/fig6_mean_rr.log

set -e

# Load conda environment
module use /g/data/xp65/public/modules
module load conda/analysis3

# point dask spill to scratch 
export DASK_TEMPORARY_DIRECTORY=/scratch/v46/ac9768/tmp
mkdir -p /scratch/v46/ac9768/tmp

cd /home/563/ac9768/GBR/scripts/Paper_figures/Fig6_12LST_windregime_def/  # path to save_radar_zarr.py  

echo "Starting towns..."
python3 fig6_mean_rr.py towns

# echo "Starting cairns..." 
# python3 fig6_mean_rr.py cairns

# echo "Starting willis..."
# python3 fig6_mean_rr.py willis

echo "All done."