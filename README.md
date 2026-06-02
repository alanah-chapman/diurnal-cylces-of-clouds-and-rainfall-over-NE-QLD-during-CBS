# Diurnal Cycles of Cloud and Rainfall over North-East Queensland During the Coral Bleaching Season

This repository contains code written for:

1. **Chapman et al. (2026, pre-print)** — *Diurnal cycles of cloud and rainfall over North-East Queensland during the coral bleaching season*
   - Data processing, analysis, and all figure production
2. **Zhao et al. (2026, pre-print)** — *Wind-regime modulation of cloud, precipitation, and surface heat budget: case studies over the central Great Barrier Reef*
   - Figure 2
3. **Masters thesis** — *Diurnal Cycles of Clouds and Precipitation During the Coral Bleaching Season*

The code analyses the following datasets, all available on NCI:
- Australian Bureau of Meteorology level 2 radar datasets
- BARRA-R2 regional reanalysis
- Level 1 Himawari-8 atmospheric and cloud products

## Repository Structure

1. `Chapman_et_al_2026preprint/`

## Repository Structure

```
├── Chapman_etal_2026preprint/          # Figure production for Chapman et al. (2026)
    ├── F01_StudyDomain.ipynb           # Figure 1
    ├── F02_Snapshots.ipynb             # Figure 2
    ├── F03_BARRA-R2_WindRegimes.ipynb  # Figure 3 BARRA-R2 850 hPa winds
    ├── F04_F05_Eqpt.ipynb              # Figure 4 & 5
    │   └── theta-e_extras/             # Contains scripts for Appendix A Figs. A1-A2 
    ├── F06_DiurnalCycles.ipynb         # Figure 6
    ├── Fig6_12LST_windregime_def/      # Radar processing
    │   ├── fig6_mean_rr.py  
    │   ├── fig6_mean_rr.sh
    │   ├── fig6_rr_freq.py
    │   └── fig6_rr_freq.sh
    ├── F07_Hovmoller.ipynb             # Figure 7
    ├── Fig7_12LST_windregime_def/      # Radar processing
    │   ├── fig7_mean_rr.py  
    │   └── fig7_mean_rr.sh
    ├── F08_StationSurfaceWinds.ipynb   # Figure 8
    ├── FA1.ipynb                       # Appendix A Figure A1
    ├── TableA2_BARRA-R2850hPa_stats.ipynb              # Appendix A Table A2
    ├── BARRA-R2_processing.ipynb       # BARRA2 data processing
    ├── Regrid-H8-dataset.ipynb         # H8 data processing 
    ├── save_radar_zarr.py/             # Radar data processing
    ├── save_radar_zarr.sh/             # Radar data processing
    ├── Radar-processing/               # Radar data processing instructions
    │   ├── Radar-processing_Step1.ipynb             
    │   ├── Radar-processing_Step2.ipynb             
    │   ├── Radar-processing_Step3.ipynb   
    │   └── Radar-processing_Step4.ipynb   
    ├── radar_masks/                    # Store .nc radar masks       
    └── radar_anc/                      # Store .nc radar longitude data arrays       
```