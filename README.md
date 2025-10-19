# Unit 2 Project - Simulation Study


### Overview 
This project was created to implement a simulation study from the article "CONTROLLING THE FALSE DISCOVERY RATE VIA KNOCKOFFS"(https://projecteuclid.org/journalArticle). The main goal is to evaluate statistical methods and write a well-structured, reproducible
pipeline with clear reporting.

## SETUP DESCRIPTION

All outputs are stored in the `results/` directory.

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
├── configs/  
│ ├── baseline.json
│ └── ar1_05.json
├── run_experiment.py # main script (runs all simulations)
├── results/
│ ├── raw/ # raw simulation output (*.csv, *.pkl)
│ └── figures/ # stores generated plots
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
```bash

```

By running the code above you will be able to recreate figures stored in results/figures
### Testing

This project includes basic tests to ensure pipeline correctness.

To run all tests:
```bash
pytest tests/
```