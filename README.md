# Unit 2 Project - Simulation Study


### Overview 
This project was created to implement a simulation study from the article "CONTROLLING THE FALSE DISCOVERY RATE VIA KNOCKOFFS"(https://projecteuclid.org/journalArticle). The main goal is to evaluate statistical methods and write a well-structured, reproducible
pipeline with clear reporting.

## SETUP DESCRIPTION

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
│ └── figures/ 
├── scripts/
│ ├── plot_knockoff_pairs.py # script to recreate figure 1 
│ ├── plot_figure_2.py # script to recreate figure 2
│ └── ...
├── tools/
│ └── generate_grid.py # creates .json files with parameter sets needed for particular experiment
├── artifacts/ # contains recreated plots
├── tests/ # simple pytest sanity checks
├── requirements.txt 
├── Makefile
├── ADEMP.md # simulation design document 
├── ANALYSIS.md #  
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
make fig1
make fig2
```
#### To delete generated files:
```bash
make clean
```

### Testing

This project includes basic tests to ensure pipeline correctness.

To run all tests:
```bash
make test
```