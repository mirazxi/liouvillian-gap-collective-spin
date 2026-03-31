# Liouvillian paper code

This folder contains the Python scripts used to generate the main numerical checks and figures for the paper.

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

# Low-rank Liouvillian-gap selection: code and datasets

This repository contains the Python scripts, datasets, and figure-generation code used for the paper

**“Low-rank structure and Liouvillian-gap selection in a coherently driven collective spin under collective dephasing.”**

The project studies the Liouvillian spectral gap of a driven collective-dephasing model, with emphasis on:

- exact irreducible-tensor block decomposition,
- low-rank sector selection,
- dipolar–quadrupolar crossover structure,
- higher-rank exclusion via scalar reduction,
- and computer-assisted box checks over the scanned parameter region.

---

## What this repository contains

The repository is organized around four main tasks:

1. **Sector scans and figure generation**  
   Numerical scans of reduced Liouvillian blocks used to generate the main sector-selection maps, line cuts, and benchmark datasets.

2. **Validation of the exact block decomposition**  
   Direct comparison between the full symmetric-space Liouvillian and the reduced rank-resolved block formulation for small system sizes.

3. **Finite-range higher-rank certification**  
   Scripts and datasets used for explicit higher-rank exclusion checks over finite ranges of sector rank.

4. **Large-\(k\) certification chunks**  
   Scripts and CSV outputs for the certified large-rank chunk computations used in the manuscript.

---

## Environment

Tested with:

- Python 3.10+
- Ubuntu / WSL
- standard scientific Python stack

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Optional thread settings for reproducible CPU usage

For more stable CPU usage on WSL / Linux, you may run:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
