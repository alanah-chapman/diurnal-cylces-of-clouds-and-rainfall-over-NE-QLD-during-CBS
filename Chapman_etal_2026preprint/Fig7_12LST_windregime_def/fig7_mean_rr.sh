#!/bin/bash
#PBS -N calc_mean_rr
#PBS -P q90
#PBS -q normal
#PBS -l ncpus=48
#PBS -l mem=192GB
#PBS -l walltime=06:00:00
#PBS -l wd
#PBS -l storage=gdata/q90+scratch/v46+gdata/xp65
#PBS -j oe
#PBS -o /scratch/v46/ac9768/radar/logs/calc_mean_rr.log

set -e

# Load conda environment
module use /g/data/xp65/public/modules
module load conda/analysis3

cd /home/563/ac9768/GBR/scripts/Paper_figures/Fig7_12LST_windregime_def/  # path to save_radar_zarr.py  

# echo "Starting cairns..." 
# python3 calc_mean_rr.py cairns

echo "Starting willis..."
python3 fig7_mean_rr.py willis

echo "All done."