# Diverging Trends in Neurodevelopmental Disorder Diagnoses in Kazakhstan (2014–2024)

This repository contains datasets and analysis code used in the epidemiological investigation of national trends in neurodevelopmental disorder (NDD) diagnoses among children in Kazakhstan between 2014 and 2024. The study analyzes registry-based prevalence patterns, regional heterogeneity, and diagnostic composition across major NDD categories, including autism spectrum disorder (ASD), attention-deficit/hyperactivity disorder (ADHD), intellectual disability (ID), and other developmental disorders. 

## Associated Manuscript

Baikatov A, Assemov A, Alimbetov D, Kustubayeva A.
*Diverging Trends in Neurodevelopmental Disorder Diagnoses in Kazakhstan (2014–2024): a National Registry Study.*

## Repository Contents

### `Trends_in_NDDs_Analysis.ipynb`

Main analysis notebook containing:

* data preprocessing and harmonization,
* epidemiological calculations,
* statistical analyses,
* prevalence estimation,
* regression modeling,
* regional comparisons,
* and figure generation used in the manuscript.

The workflow reproduces the preprocessing, statistical analyses, and visualizations described in the study. Analyses were conducted using Python scientific computing and statistical libraries in Google Colab. 

### `data/ndd_df.csv`

Processed and aggregated epidemiological dataset derived from anonymized national registry records spanning 2014–2024.

The dataset contains aggregated demographic, diagnostic, temporal, and regional variables used for:

* registry-based prevalence estimation,
* longitudinal trend analyses,
* multivariable statistical modeling,
* and generation of publication figures and tables.

No directly identifiable participant-level information is included in the public release.

### `data/govstat/`

Folder containing yearly denominator datasets derived from official demographic statistics published by the Bureau of National Statistics of the Republic of Kazakhstan. 

These datasets were used for:

* population normalization,
* prevalence calculations,
* demographic standardization,
* and region-level epidemiological comparisons.

## Study Overview

The study used a repeated cross-sectional registry-based design to investigate longitudinal trends in pediatric NDD diagnoses across Kazakhstan. Annual registry-based prevalence was calculated using national population denominators for children aged ≤15 years. Statistical analyses included negative binomial regression, Bayesian hierarchical modeling, mixed-effects logistic regression, and regional correlation analyses. 

## Reproducibility

This repository is intended to support transparency and reproducibility of the analytical workflow presented in the manuscript. The provided notebook and datasets allow reproduction of the primary statistical analyses, summary tables, and figures reported in the study.

## Requirements

Main Python packages used:

* pandas
* numpy
* scipy
* statsmodels
* matplotlib
* seaborn
* scikit-learn
* pymc / bayesian modeling libraries
* jupyter

## Data Availability

The datasets included in this repository are aggregated and processed for research reproducibility purposes. Due to institutional and ethical restrictions, raw individual-level registry data are not publicly distributed.

## Citation

If using this repository or associated datasets, please cite the corresponding publication once available.

## Ethics Statement

The original study received ethical approval from the Local Ethical Committee of al-Farabi Kazakh National University (Protocol № IRB-A843). The analyses were conducted exclusively on anonymized registry-derived data. 

## License

This repository is shared for academic and non-commercial research purposes.
