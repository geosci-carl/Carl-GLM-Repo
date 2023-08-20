# Carl-GLM-Repo
This repo contains my work on creating a weather generator of the HUC-4 (0410) Western Lake Erie Subregion. 
Created by Carl Fredrick G. Aquino. Questions? Contact me at geosci.carl@gmail.com.

## To reproduce my work:
1. Install all package dependencies listed in environment.yml using `conda env create --file environment.yml`
3. Activate environment using `conda activate geo3`
5. Download daymet data (nc files) into 'daymet_data' subfolder using instructions found on https://github.com/ornldaac/daymet-TDStiles-batch
6. To create a weather generator for the Portage River HUC-12 location (41000100502), run `weather_generator_portage.py`.
7. To create a weather generator for the HUC-04 (0410) Western Lake Erie Subregion, run `weather_generator.py`.
8. To create a yearly precip correlation matrix for the HUC-12 locations, run `Daymet_initial_analysis.py`.

## Note for Windows Machines:
I had an issue running hmmlearn (hidden markov model plugin) on my Windows machine.  Here was my solution:
1. Completely uninstall Anaconda/Miniconda
2. Install Conda using Miniforge (https://github.com/conda-forge/miniforge)
3. Install all packages through conda-forge (within Miniforge, follow steps 1 and 2 above to install and activate environment)
