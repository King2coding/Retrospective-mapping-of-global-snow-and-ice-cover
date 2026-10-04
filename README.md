# Retrospective mapping of global snow and ice cover

Research code supporting the reconstruction of daily global snow and ice cover
before the start of the Global Multisensor Automated Snow and Ice Mapping
System (GMASI) record. The project evaluates climatological and machine-learning
approaches using ERA5 surface variables and applies the selected approach to
extend the GMASI record back to 1 January 1980.

## Associated publication

Kumah, K. K., O. Zandi, and A. Behrangi, 2025: Retrospective Mapping of
Global Snow and Ice Cover Beyond the Satellite Observational Era. *Earth and
Space Science*, **12**(5), e2024EA004171.
[https://doi.org/10.1029/2024EA004171](https://doi.org/10.1029/2024EA004171)

## Published dataset

Kumah, K. K., O. Zandi, and A. Behrangi, 2024: *Global Snow and Ice Cover
(1980–1987): An Extended GMASI Dataset*. University of Arizona Research Data
Repository, Version 1.
[https://doi.org/10.25422/AZU.DATA.28012532.V1](https://doi.org/10.25422/AZU.DATA.28012532.V1)

The published dataset provides globally continuous daily surface-cover
classification on a 0.1° latitude–longitude grid from 1 January 1980 through
30 June 1987. Data files are distributed through the repository DOI rather
than duplicated in this code repository.

## Related AGU presentation

Kumah, K. K., O. Zandi, and A. Behrangi, 2024: “On the Retrospective Mapping
of Global Snow and Ice Cover Beyond the Satellite Observational Era.” AGU Fall
Meeting 2024, *Artificial Intelligence and Machine Learning for the Cryosphere
III*, presented 13 December 2024.
[AGU abstract 1599765](https://agu.confex.com/agu/agu24/meetingapp.cgi/Paper/1599765)

## Repository status

This repository preserves the research scripts used during development and
evaluation. The source files have not been refactored merely for presentation
on GitHub. Some scripts contain rain-server paths and expect externally stored
GMASI, ERA5, Rutgers snow-cover, ancillary, and generated data.

## Workflow overview

The repository contains scripts for:

- retrieving and preprocessing GMASI and ERA5 variables;
- generating daily climatology and fractional-cover predictors;
- training the ML-E and ML-EC Random Forest approaches;
- applying consistency checks in the ML-ECC approach;
- extending GMASI from mid-1987 back to 1980;
- validating the reconstruction against GMASI during 1988–1991; and
- producing the statistical analyses and figures used in the publication.

The principal model variants are:

- **ML-E:** ERA5 and static geographic predictors;
- **ML-EC:** ML-E predictors plus GMASI-derived climatological predictors; and
- **ML-ECC:** ML-EC with additional spatial and temporal consistency checks.

## Key scripts

| File | Purpose |
| --- | --- |
| `ML-E_and_ML-EC.py` | Trains and evaluates the primary Random Forest reconstruction approaches. |
| `ML-ECC.py` | Applies additional consistency checks to the ML-EC reconstruction. |
| `ML-EC_extending_GMASI_from_mid1987-1980.py` | Extends the selected reconstruction through the pre-GMASI period. |
| `creating_static_predictors_for_ML-EC_based_on_1992-2022_data.py` | Creates climatological predictors used by ML-EC. |
| `autosnow_extension-evaluation.py` | Evaluates reconstruction performance against observed GMASI classes. |
| `Autosnow_Extension-Results_Evaluation_and_Analysis.py` | Produces publication-oriented evaluation and analysis. |
| `Analysis_of_extended_GMASI_data.py` | Analyzes the completed 1980–1987 extension. |
| `autosnow_extending_dataretrieval.py` | Retrieves and organizes input data for the workflow. |
| `era5_download_global_variables.py` and `ERA5land_code.py` | Supporting ERA5 acquisition scripts. |
| `eval_functions.py`, `plotting_functions.py`, and `util_functions.py` | Shared evaluation, visualization, and processing utilities. |

## Data and environment requirements

Input and generated data are not stored in this repository. See
[`DATA.md`](DATA.md) for the principal datasets and authoritative access links.

Imports in the scripts indicate a scientific Python environment using packages
such as `numpy`, `pandas`, `xarray`, `netCDF4`, `scipy`, `scikit-learn`,
`matplotlib`, `seaborn`, `cartopy`, `rasterio`, and `joblib`. Exact historical
versions were not recorded consistently, so a fully pinned environment cannot
be stated without additional evidence.

## Citation

Please cite the *Earth and Space Science* article and the dataset DOI when using
the reconstructed data. Machine-readable metadata is provided in
[`CITATION.cff`](CITATION.cff).

## License

No software reuse license has yet been assigned. The article and dataset have
their own stated terms; those terms do not automatically establish a license
for this source-code repository.

## Contact

Kwabena Kingsley Kumah — [GitHub profile](https://github.com/King2coding)
