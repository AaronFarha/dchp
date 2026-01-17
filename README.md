
# DCHP

This repository is a companion to the paper

    A.H.P. Farha, J.P. Ore, E.N. Pergantis, D. Ziviani, E.A. Groll, and K.J. Kircher. "Laboratory and field testing of a residential heat pump retrofit for a DC solar nanogrid" In Energy Conversion and Management

It runs an analysis of the field and laboratory data to compare the operation of a 4-ton residential split system heat pump on AC and DC power sources.

The file `pv_util.py` is a python script which simulates the annual operation of the heat pump on AC and DC for comparison. Efficiency curves are used to calculate the conversion losses on AC and DC nanogrids.

## Installation

### Install from GitHub

You can install the package directly from the GitHub repository:

```bash
pip install git+https://github.com/yourusername/dchp.git
```

Or install a specific branch or tag:

```bash
pip install git+https://github.com/yourusername/dchp.git@branch-name
```

### Install from Local Source

If you've cloned the repository locally, you can install it in development mode:

```bash
git clone https://github.com/yourusername/dchp.git
cd dchp
pip install -e .
```

Or install it normally:

```bash
pip install .
```

## Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.3.0
- scipy >= 1.7.0
- statsmodels >= 0.12.0

Optional dependencies:
- scikit-learn >= 0.24.0 (for isolation forest outlier detection) 