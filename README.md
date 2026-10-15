# BiasCalibTwin

Implementation of **Bias-separated Bayesian state estimation for asynchronous process monitoring.**

## Install

Use Python 3.11 in an isolated environment. Run from the repository root:

```bash
python -m pip install -e .
```

## Data

Export complete [Tennessee Eastman](https://depts.washington.edu/control/LARRY/TE/download.html) trajectories as NPZ: `sim,truth: (N,T,4)`, `sensor,mask,age: (N,T,S)`, `control: (N,T,U)`, `metro,metro_mask: (N,T,2)`, `run: (N,)`, `split: train/val/test`, and `observation: (S,4)`. The four coordinates are reactor/separator thermal deviations then G/H composition. Metrology values are meaningful only where their mask is true. Group complete runs and operating-condition combinations before export. Sensor matrix/units and controls must match the simulator observation model.

## Workflow

```bash
python -m biascalibtwin prepare --input data/process.npz
python -m biascalibtwin train
python -m biascalibtwin run
python -m biascalibtwin evaluate
python -m biascalibtwin plot
```

Main commands accept `--config config.json` and `--output runs/default`. Each output directory records `settings.json`. Keep independent seeds, datasets and conditions in separate directories.

## Configuration

`width` sizes source GRU states. `bias_step` scales latent discrepancy increments. `sensor_noise,metro_noise` define likelihood scales in the exported measurement units. `kl_weight,physics_weight,gate_weight,calibration_weight` combine likelihood/prior, balance, source-reliability and CRPS terms. `sample_minutes` converts detection delay. `uncertainty_quantile` fixes the source-validation abstention operating point. Additional optimizer, batch-size and model-width defaults are in `config.json`.

## Outputs

`data.npz` stores training-only normalization and the fitted simulator transition. `model.pt` stores causal source encoders/gates. `predictions.npz` retains posterior moments, source messages, reliability and information shares. `metrics.csv` reports unmeasured-composition normalized RMSE/MAE, NLL, Gaussian CRPS, coverage/error, physics residual, source reliability correlation/shares, alarm balanced accuracy/F1/FAR/miss/delay and abstention.

## Method and comparisons

The simulator message subtracts an explicit slow latent bias. Sensor input includes missingness and age. Sparse metrology contributes zero precision when unavailable. Source-quality gates scale positive Gaussian precision, and fusion adds the predictive prior. Training uses source likelihoods, prior regularization, physics, anchor reliability and CRPS without dense test truth. Evaluation truth is read only for metrics. The balance transition is a fitted linear simulator surrogate plus a learned residual; the abstention setting is a frozen validation quantile.

Controls: `full`, `no-bias`, `fixed-gates`, `no-physics`, `sensor-only`, `simulator-only`. Choose `--variant` and use a separate training/output directory.

```bash
python -m biascalibtwin.compare --inputs "runs/*/metrics.csv" --reference full --metrics normalized_rmse crps coverage_error --unit seed
```

Comparison averages rows within independent units before bootstrap intervals and paired Wilcoxon/Holm inference. For participant/client analyses choose those independent unit columns instead of counting recordings as replicates.

