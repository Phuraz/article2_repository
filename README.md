# Code and supplementary materials for A Bayesian Predictive Tail Evaluation Framework for Bitcoin Block Arrival Delays

This repository contains the analysis notebooks, figures, tables, and data used for the study, “A Bayesian Predictive Tail Evaluation Framework for Bitcoin
Block Arrival Delays.”

## Contents

- `REVIEWERS_Refactored_Final_Bitcon_PyMC_Bayes_Analysis.ipynb.`: final Hawkes-model analysis and results.
- `Refactored_Final_Bitcon_PyMC_Bayes_Analysis.ipynb`: original final analysis notebook. It contains the fitted-model workflow and saved model outputs used in the analysis.
- `Recursion_Hawkes_...ipynb`: Hawkes likelihood recursion implementation.
- `Refactored_Hawkes_Model_Selection.ipynb`: model-selection analysis.
- `functions.py`: reusable Python functions called by the main notebook, including model-fitting and evaluation utilities.
- `figs/`: manuscript figures.
- `tables/`: manuscript tables.
- `bitcoin_blocks_and_transactions_sorted.csv`: analysis dataset.

The analyses were implemented in Python using PyMC. Sampling used four chains, 1,200 tuning iterations, and 1,200 posterior draws per chain. Package versions and a more streamlined reproduction workflow will be added in a post-review update.
