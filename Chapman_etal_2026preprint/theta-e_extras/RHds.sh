#!/bin/bash
#PBS -N RHds
#PBS -P v46
#PBS -q normal
#PBS -l ncpus=48
#PBS -l mem=192GB
#PBS -l jobfs=10GB
#PBS -l walltime=10:00:00
#PBS -l wd
#PBS -l storage=gdata/q90+scratch/v46+gdata/xp65+gdata/gb02+gdata/ob53
#PBS -j oe
#PBS -o /home/563/ac9768/GBR/scripts/Paper_figures/theta-e_extras/RHds.log

# Load conda environment
module use /g/data/xp65/public/modules
module load conda/analysis3

# module use /g/data/gb02/public/modules/
# module load dask_setup

# point dask spill to scratch 
export DASK_TEMPORARY_DIRECTORY=/scratch/v46/ac9768/tmp
mkdir -p /scratch/v46/ac9768/tmp

cd /home/563/ac9768/GBR/scripts/Paper_figures/theta-e_extras/  

echo "Starting analysis..."
python3 RHds.py 'RH'

echo "All done."