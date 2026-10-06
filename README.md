# Scope
This repository implements the Atlas stabilization benchmark described in the [paper](https://arxiv.org/abs/2609.33451) presenting [DAOCP](https://github.com/AlbertoZaupa/daocp). The qpOASES version required for this benchmark is
[AlbertoZaupa/pyqpoases](https://github.com/AlbertoZaupa/pyqpoases), which includes a Python wrapper.


## Atlas MPC benchmark

Stabilizes the linearized Atlas humanoid (58 states, 29 inputs) with input-constrained
MPC: horizon 30, 300 simulation steps per simulation. Model data comes from
[ReLUQP.jl](https://github.com/RoboticExplorationLab/ReLUQP.jl).

Install the Python dependencies from `requirements.txt`. Build DAOCP, HPIPM and qpOASES from source, together with their Python interfaces.
Update `export.sh` with the shared-library and Python interface paths, then source
it before running the benchmark:

```bash
python -m pip install -r requirements.txt
source ./export.sh
```

The benchmark runner accepts different flags

```bash
python benchmark.py
python benchmark.py --solver hpipm --nreps 10 --save
python benchmark.py --solver daqp --nreps 5 --save
python benchmark.py --solver qpoases --nreps 100 --save
```

Defaults: `--solver daocp`, `--nreps 1`, saving disabled. `--save` writes a unique
`results/<solver>_<timestamp>.npz` with per-repetition timings, iterations, and
state/input deviations from equilibrium. Data paths are relative to the runner.

Timings are in milliseconds and measure each Python solve call, excluding setup,
problem updates, and simulation. Each repetition creates a fresh solver; successive
steps retain its warm start. DAOCP, DAQP and qpOASES retain the original settings; HPIPM
uses `speed_abs` and tolerances of `1e-3`, so numerical
accuracy differs between solvers.