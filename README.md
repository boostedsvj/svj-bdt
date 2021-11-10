# Environment setup

In `conda` (or another virtual environment), do:

```
pip install notebook
pip install tqdm
pip install xgboost
pip install matplotlib
pip install uptools
pip install seutils
```

Then clone this repository.


# Prepare the data

```
python dataset.py signal
# python dataset.py bkg  # DEPRECATED, use job file instead
```

This requires the `gfal` command line tools to be installed.


# Train the BDT

See [the notebook](bdt.ipynb).


# Produce histograms

In environment with `htcondor` and `qondor` installed:

```
qondor-submit combine_hist_jobs.py
```

This will produce `.npz` files with an array `mt` and `score`, which can be used to cut for certain BDT scores and produce `mt` histograms.

