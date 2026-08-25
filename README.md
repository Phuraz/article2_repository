# Code and supplementary materials for A Bayesian Predictive Tail Evaluation Framework for Bitcoin Block Arrival Delays

This repository contains the analysis notebooks, figures, tables, and data used for the study, “A Bayesian Predictive Tail Evaluation Framework for Bitcoin
Block Arrival Delays.”

## Contents
- `Final_Hawkes_Bitcoin_data_PyMC_Model_Comparison_Py310.ipynb`: Initial model-fitting experimentation, including code for saving fitted model objects to disk using `az.to_netcdf()`. Most model-fitting blocks are commented out but can be uncommented and rerun where required. The saved model-output files are not included because they exceed GitHub’s file-size limit. Consequently, to reproduce the workflow from scratch, the model-fitting code must first be executed and the resulting outputs saved locally before running `REVIEWERS_Refactored_Final_Bitcon_PyMC_Bayes_Analysis.ipynb`.
- `REVIEWS_Refactored_Final_Bitcon_PyMC_Bayes_Analysis.ipynb`: Final analysis notebook containing the reported visualisations, predictive comparisons, and evaluation results. It performs minimal model fitting; instead, previously fitted model outputs are loaded from disk using `az.from_netcdf()`.
- `functions.py`: reusable Python functions called by the main notebook, including model-fitting and evaluation utilities.
- `figs/`: manuscript figures.
- `tables/`: manuscript tables.
- `bitcoin_blocks_and_transactions_sorted.csv`: analysis dataset.

The analyses were implemented in Python: 3.10.13 using PyMC: 5.25.1. in Windows Subsystem for Linux (WSL). Sampling used four chains, 1,200 tuning iterations, and 1,200 posterior draws per chain.
