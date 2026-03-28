# Liouvillian paper code

This folder contains the Python scripts used to generate the main numerical checks and figures for the paper.

## Environment

Tested with Python 3.10+ on Ubuntu.

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Scripts

### 1. `sector_scans_and_plots.py`
Generates:
- broad sector-winner map
- focused wedge map
- line cuts at several fixed detunings
- benchmark table CSV
- raw datasets for the scans

Usage:
```bash
python3 sector_scans_and_plots.py
```

### 2. `full_vs_block_validation.py`
Validates the irreducible-tensor block decomposition against direct diagonalization of the full symmetric-space Liouvillian for small system sizes.

Usage:
```bash
python3 full_vs_block_validation.py
```

## Output

All outputs are written to an `outputs/` subfolder inside this directory.

## Notes

- All scripts assume dimensionless units with `gamma = 1` unless explicitly changed.
- The block construction follows the rank-`k` tridiagonal Liouvillian blocks used in the manuscript.
- The full Liouvillian validation uses the symmetric spin sector of dimension `N+1`.