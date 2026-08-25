# Code and supplementary materials for A Bayesian Predictive Tail Evaluation Framework for Bitcoin Block Arrival Delays

This repository contains the analysis notebooks, figures, tables, and data used for the study, “A Bayesian Predictive Tail Evaluation Framework for Bitcoin
Block Arrival Delays.”

## Contents

- `REVIEWS_Refactored_Final_Bitcon_PyMC_Bayes_Analysis.ipynb.`: Is the final analysis contain all the visualisation and comparisons. Minimal model fitting, models are rather loaded from disk via az.from_netcdf().
- `Final_Hawkes_Bitcoin_data_PyMC_Model_Comparison_Py310`: Initial experimentation with model fitting which into the disk via az.to_netcdf(). Most model fitting logic commented out, can be uncommented and executed.
- `functions.py`: reusable Python functions called by the main notebook, including model-fitting and evaluation utilities.
- `figs/`: manuscript figures.
- `tables/`: manuscript tables.
- `bitcoin_blocks_and_transactions_sorted.csv`: analysis dataset.

The analyses were implemented in Python using PyMC. Sampling used four chains, 1,200 tuning iterations, and 1,200 posterior draws per chain. Package versions and a more streamlined reproduction workflow will be added in a post-review update.
