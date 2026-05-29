#!/bin/bash
#PBS -N save_radar_zarr
#PBS -P v46
#PBS -q normal
#PBS -l ncpus=48
#PBS -l mem=192GB
#PBS -l walltime=06:00:00
#PBS -l wd
#PBS -l storage=gdata/q90+gdata/v46+scratch/v46+gdata/xp65
#PBS -j oe
#PBS -o /home/563/ac9768/GBR/scripts/Paper_figures/logs/save_radar_zarr.log

set -e

# Load conda environment
module use /g/data/xp65/public/modules
module load conda/analysis3

cd /home/563/ac9768/GBR/scripts/Paper_figures/  # path to save_radar_zarr.py

# echo "Starting towns..." 
# python3 save_radar_zarr.py towns

# echo "Starting cairns..." 
# python3 save_radar_zarr.py cairns

echo "Starting willis..."
python3 save_radar_zarr.py willis

echo "All done."