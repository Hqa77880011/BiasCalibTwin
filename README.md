# BiasCalibTwin

Implementation of **Bias-separated Bayesian state estimation for asynchronous process monitoring.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Export complete [Tennessee Eastman](https://depts.washington.edu/control/LARRY/TE/download.html) trajectories as NPZ: `sim,truth: (N,T,4)`, `sensor,mask,age: (N,T,S)`, `control: (N,T,U)`, `metro,metro_mask: (N,T,2)`, `run: (N,)`, `split: train/val/test`, and `observation: (S,4)`. The four coordinates are reactor/separator thermal deviations then G/H composition. Metrology values are meaningful only where their mask is true. Group complete runs and operating-condition combinations before export. Sensor matrix/units and controls must match the simulator observation model.

```bash
python -m biascalibtwin prepare --input data/process.npz --config config.json --output runs/full
```

## Training and inference

```bash
python -m biascalibtwin train --config config.json --output runs/full
python -m biascalibtwin run --config config.json --output runs/full
```

`config.json` records the method settings and seed `11`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m biascalibtwin evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `no-bias`, `fixed-gates`, `no-physics`, `sensor-only`, `simulator-only`.

Repeat the same data and configuration in a separate directory, for example:

```bash
python -m biascalibtwin prepare --input data/process.npz --config config.json --output runs/no-bias --variant no-bias
python -m biascalibtwin train --config config.json --output runs/no-bias --variant no-bias
python -m biascalibtwin run --config config.json --output runs/no-bias --variant no-bias
python -m biascalibtwin evaluate --config config.json --output runs/no-bias --variant no-bias
```

## Outputs

All main outputs are written below `runs/full/`.

`data.npz` stores training-only normalization and the fitted simulator transition. `model.pt` stores causal source encoders/gates. `predictions.npz` retains posterior moments, source messages, reliability and information shares. `metrics.csv` reports unmeasured-composition normalized RMSE/MAE, NLL, Gaussian CRPS, coverage/error, physics residual, source reliability correlation/shares, alarm balanced accuracy/F1/FAR/miss/delay and abstention.

```bash
python -m biascalibtwin plot --config config.json --output runs/full
```
