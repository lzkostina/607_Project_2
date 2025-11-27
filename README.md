# Unit 3 Project - Simulation Study (High Performance)


### Overview 
This project was created to extend Project 2 simulation study based on article 
"CONTROLLING THE FALSE DISCOVERY RATE VIA KNOCKOFFS"(https://projecteuclid.org/journalArticle). The main goal of this
extension is to transform the original simulation code used for Project 2 into high-performance and reproducible 
simulation pipeline. 

## SETUP DESCRIPTION
All optimizations and their documentation appear in:
`docs/BASELINE.md` – Pre-optimization profiling and complexity. 
`docs/OPTIMIZATION.md` – Implemented improvements.
All outputs are stored in the `results/` directory.
Recreated figures are stored in the `artifacts/` directory.  

### Repository Structure
```
607_Project_2/
├── data/
│ └── simulated/ # cache simulation replicates if needed
├── src/
│ ├── dgps.py # data-generating functions
│ ├── methods.py # statistical methods being evaluated
│ ├── metrics.py # performance measure calculations
│ └── simulation.py # main simulation orchestration
├── configs/  # .json files with parameters  
│ ├── baseline.json
│ └── ....
├── run_experiment.py # main script (runs all simulations)
├── results/
│ ├── raw/ # raw simulation output (*.csv, *.pkl)
│ ├── figures/ # complexity plots, comparison plots, etc.
│ └── profiling
├── scripts/
│ ├── plot_knockoff_pairs.py # script to recreate figure 1 
│ ├── plot_figure_2.py # script to recreate figure 2
│ ├── complexity_baseline.py
│ ├── plot_complexity.py
│ ├── plot_complexity_comparison.py
│ ├── benchmark_runtime.py
│ ├── stability_check.py
│ └── ...
├── tools/
│ └── generate_grid.py # creates .json files with parameter sets needed for particular experiment
├── docs/
│ ├── ADEMP.md # simulation design document 
│ ├── ANALYSIS.md 
│ ├── BASELINE.md # baseline profiling and complexity
│ └── OPTIMIZATION.md # all improvements
├── artifacts/ # contains recreated plots
├── tests/ # simple pytest sanity checks
├── requirements.txt 
├── Makefile
└── README.md # this file
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/lzkostina/607_Project_2
cd 607_Project_2
```
2. Create environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage
All main steps are automated via the Makefile.
#### Run the full simulation pipeline:
```bash
make all
```
### Run only the §3.3.1 Row 1 experiment:
```bash
make simulate RUN_331=1
```
#### Create all plots
```bash
make figures
```
#### You can also run individual figures:
```bash
make fig1 # pairwise knockoff plot
make fig2 # Lasso path visualization
```

#### Profile runtime on a single configurations set
```bash
make profile
```
#### Complexity analysis
```bash
make complexity
```
Produces the following files:
- `results/profiling/complexity_baseline.csv`
- `results/figures/complexity_baseline.png`
- `results/figures/complexity_comparison.png`

#### Sequentila vs parallel timing 
```bash
make benchmark
```
Creates speedup table

#### Run a parallel version 
```bash
make parallel
```
#### Numerical stability check
```bash
make stability-check
```
#### To delete generated files:
```bash
make clean
```

### Testing

This project includes basic tests to ensure pipeline correctness.
+ In this updated version `test_regression.py` file added to ensure that the optimized version of a code 
produces similar results to the original implementation.

To run all tests:
```bash
make test
```