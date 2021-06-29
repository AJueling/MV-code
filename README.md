# MV-code
Code for Ocean Science publication "Effects of strongly eddying oceans on multidecadal climate variability in the CESM"

I did my best to share the complete code here and document it reasonably well. There are many functions that are not used for the final figures, but function calls should be obvious. Unfortunately, without access to the data on SurfSara's Cartesius computer, it will not be possible to recreate the plots. The total CESM output is several terabytes and so no easily made publicly available.

There is an `environment.yml` from which a conda environment can (in principle) be recreated. I have found that this often fails in practice, but you can simply create a new conda environment and install the necessary packages (https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), should there be an error you can specify the version as in the `environment.yml` file.

## Repository Structure

```
MV-code
│   README.md
│   LICENSE
│   environment.yml      (conda environments file)
│
└───src
│    (start here)
│    │MV-paper.ipynb             notebook to create main figures
│    │SST_index_generation.py    pipeline to recreate all derived data files
│
│    (other notebooks)
│    │comparing_spectra.ipynb    notebook exploring different ways to compare spectra
│    │other_SST_products.ipynb   notebook to create appendix figure
│
│    (auxiliary functions)
│    │ab_derivation_SST.py
│    │ba_analysis_dataarrays.py
│    │bb_analysis_timeseries.py
│    │bc_analysis_fields.py
│    │bd_analysis_indices.py
│    │constants.py
│    │filters.py
│    │grid.py
│    │maps.py
│    │OHC.py
│    │paths.py
│    │plotting.py
│    │read_binary.py
│    │regions.py
│    │timeseries.py
│    │xr_DataArrays.py
│    │xr_integrate.py
│    │xr_regression.py
```