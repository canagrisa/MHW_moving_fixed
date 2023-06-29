
[![DOI](https://zenodo.org/badge/637936104.svg)](https://zenodo.org/badge/latestdoi/637936104)

# Rosselló et al. (2023) Code Repository

This repository provides all the code required for data downloading, preprocessing, and analysis to generate the results (figures and tables) presented in Rosselló et al. (2023). The code is mainly composed by jupyter notebooks and relies heavily on two Python packages specifically developed for this purpose:

- [**CMIP_data_retriever**](https://github.com/canagrisa/CMIP_data_retriever): A package for downloading CMIP model data relevant to the region and period of study.
- [**MHW_metrics**](https://github.com/canagrisa/MHW_metrics): A package for computing Marine Heatwave (MHW) metrics on netCDF4 files.


# Description of the Repository

This repository provides the code and resources needed to reproduce the results, figures, and tables in [Rosselló et al. (2023)](https://www.frontiersin.org/articles/10.3389/fmars.2023.1168368/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Marine_Science&id=1168368). The repository is organized into the following folders and files:

- `code/`
    - `data_retrieve/`: Code to download raw data for the study.
        - For model data, the CMIP_data_retriever package is used.
        - For satellite data, data is downloaded using CMEMS API calls (requires a CMEMS account).
    - `preprocessing/`: Code for preprocessing the model and satellite data.
        - `cmip_mhw.ipynb`: Compute MHW metrics for all CMIP6 models.
        - `satellite_mhw.ipynb`: Compute MHW metrics for 1982-2021 SST dataset for the Mediterranean Sea.
        - `mean_cmip_tseries.ipynb`: Compute yearly SST timeseries for each CMIP6 model, variant, and scenario.
        - `mean_sst_tseries.ipynb`: Compute yearly SST timeseries for satellite data.
    - `figures/`: Folder containing a Jupyter Notebook for each figure in the article.
    - `tables/`: Folder containing a Jupyter Notebook for each table in the article.
    - `utils.py`: Utility functions used throughout the repository.
    - `plot_utils.py`: Utility functions for plotting and visualizations.
- `results/`: Contains the processed datasets the article draws upon (MHW metrics for satellite and model data, trends, etc.).
- `figures/`: Generated figures.
- `tables/`: Generated tables.
- `data/`: Initially empty. Where raw data is stored if downloaded from `data_retrieve/` files.



<!-- - **Data Downloading**: The raw satellite and model data are not included in the repository due to their large size. However, we provide code to download the relevant raw satellite and model data for the study using the `CMIP_data_retriever` package.
- **Data Preprocessing**: The code for preprocessing the raw data, including data cleaning, filtering, and transformation, is included in this repository.
- **Data Analysis**: The code for performing data analysis, such as calculating MHW metrics using the `MHW_metrics` package, is provided.
- **Results**: The results of the study, including the generated data files, are included in the repository.
- **Figures and Tables**: The code to regenerate the figures and tables present in the article is provided, as well as the final versions of the figures and tables. -->


## Support and Contributions

For support or to report bugs, please open a GitHub issue or email me at 


