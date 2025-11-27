PY            ?= python
PYTHONPATH    ?= .

CONFIG_DIR    ?= configs
RESULTS_DIR   ?= results
ARTIFACTS_DIR ?= artifacts
SUMM_DIR      ?= $(RESULTS_DIR)/summaries
TABLE_DIR     ?= $(RESULTS_DIR)/raw
FIG_DIR       ?= $(RESULTS_DIR)/figures
PROFILE_DIR      ?= $(RESULTS_DIR)/profiling
PROFILE_CONFIG   ?= $(CONFIG_DIR)/ar1_rho0p5_p1000_n3000_k10_A3p5.json
PROFILE_PROF_OUT ?= $(PROFILE_DIR)/prof_ar1_rho0p5_p1000_n3000_k10_A3p5.prof


# Simulation controls
REPS   ?= 600
SEED0  ?= 1021


# Simulation scripts (from scripts/)
SIM_SCRIPT_ALL ?= src/simulation.py

SIM_SCRIPT_331 ?= scripts/sec331_row1_knockoffplus_equi.py
OUT_331        ?= $(SUMM_DIR)/sec331_row1_knockoffplus_equi.csv

SCRIPT_FIG_1  ?= scripts/plot_knockoff_pairs.py
SCRIPT_FIG_2  ?= scripts/plot_figure_2.py
OUT_FIG_1     ?= $(ARTIFACTS_DIR)/knockoffs/fig1_synthetic.png
OUT_FIG_2     ?= $(ARTIFACTS_DIR)/knockoffs/figure2_lasso_path.png


# --------- Derived variables ----------
CONFIGS    := $(wildcard $(CONFIG_DIR)/*.json)
COND_IDS   := $(notdir $(basename $(CONFIGS)))
SUMMARIES  := $(addsuffix .csv,$(addprefix $(SUMM_DIR)/,$(COND_IDS)))
#AGG_TABLE  := $(TABLE_DIR)/aggregate.csv
AGG_TABLE :=  $(wildcard $(TABLE_DIR)/*.csv)

.PHONY: all simulate simulate331 analyze figures clean test help profile complexity benchmark parallel stability-check

all: simulate analyze figures

# --------- SIMULATE (conditional dispatcher) ----------
simulate: _simulate_dispatch

_simulate_dispatch:
	@mkdir -p $(SUMM_DIR) $(ARTIFACTS_DIR)/params
	@if [ -n "$(ONLY_SCRIPT)" ]; then \
		echo "Running ONLY_SCRIPT=$(ONLY_SCRIPT)"; \
		if $(PY) $(ONLY_SCRIPT) --help >/dev/null 2>&1; then \
			PYTHONPATH=$(PYTHONPATH) $(PY) $(ONLY_SCRIPT) \
				--replicates $(REPS) --seed0 $(SEED0) --out $(OUT_331) || \
			PYTHONPATH=$(PYTHONPATH) $(PY) $(ONLY_SCRIPT); \
		else \
			PYTHONPATH=$(PYTHONPATH) $(PY) $(ONLY_SCRIPT); \
		fi; \
	elif [ "$(RUN_331)" = "1" ] && [ -f "$(SIM_SCRIPT_331)" ]; then \
		echo "Running §3.3.1 Row 1: $(SIM_SCRIPT_331)"; \
		if $(PY) $(SIM_SCRIPT_331) --help >/dev/null 2>&1; then \
			PYTHONPATH=$(PYTHONPATH) $(PY) $(SIM_SCRIPT_331) \
				--replicates $(REPS) --seed0 $(SEED0) --out $(OUT_331); \
			echo "   ↳ wrote $(OUT_331)"; \
		else \
			PYTHONPATH=$(PYTHONPATH) $(PY) $(SIM_SCRIPT_331); \
		fi; \
	elif [ -f "$(SIM_SCRIPT_ALL)" ]; then \
    	echo "Running batch via $(SIM_SCRIPT_ALL)"; \
    	PYTHONPATH=$(PYTHONPATH) $(PY) $(SIM_SCRIPT_ALL) \
       		--simulate $(foreach c,$(CONFIGS),-c $(c)); \
	elif [ -f "$(SIM_SCRIPT_ONE)" ]; then \
		echo "Running per-config via $(SIM_SCRIPT_ONE)"; \
		$(MAKE) -j $$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1) $(SUMMARIES); \
	else \
		echo "No simulation scripts found. Provide $(SIM_SCRIPT_ALL) or $(SIM_SCRIPT_ONE)."; \
		exit 1; \
	fi
	@echo "Simulation complete."

# Handy alias
simulate331:
	@$(MAKE) simulate RUN_331=1


analyze: $(AGG_TABLE)
	@echo "Analysis complete: $(AGG_TABLE)"

figures: fig1 fig2
	@echo "Figures complete."

# --------- Profiling / Complexity / Benchmark / Stability ----------

# Run profiling on a representative simulation (writes .prof file)
profile:
	@mkdir -p $(PROFILE_DIR)
	@echo "Profiling baseline config: $(PROFILE_CONFIG)"
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m cProfile \
		-o $(PROFILE_PROF_OUT) \
		src/simulation.py --simulate -c $(PROFILE_CONFIG)
	@echo "Wrote profile to $(PROFILE_PROF_OUT)"

# Run computational complexity analysis (timing vs n) + log-log plot
complexity:
	@mkdir -p $(PROFILE_DIR) $(FIG_DIR)
	@echo "Running computational complexity experiment (timing vs n)..."
	@PYTHONPATH=$(PYTHONPATH) $(PY) scripts/complexity_baseline.py
	@PYTHONPATH=$(PYTHONPATH) $(PY) scripts/plot_complexity.py
	@PYTHONPATH=$(PYTHONPATH) $(PY) scripts/plot_complexity_comparison.py
	@echo "Complexity results:         $(PROFILE_DIR)/complexity_baseline.csv"
	@echo "Baseline complexity plot:   $(FIG_DIR)/complexity_baseline.png"
	@echo "Comparison complexity plot: $(FIG_DIR)/complexity_comparison.png"


# Run timing comparison: sequential vs parallel (baseline vs optimized)
# (expects you to implement scripts/benchmark_runtime.py)
benchmark:
	@echo "Running runtime benchmark (sequential vs parallel)..."
	@PYTHONPATH=$(PYTHONPATH) $(PY) scripts/benchmark_runtime.py
	@echo "Benchmark complete."

# Run optimized simulation with parallelization enabled
parallel:
	@echo "Running optimized simulation with parallelization..."
	@PYTHONPATH=$(PYTHONPATH) $(PY) scripts/run_parallel.py
	@echo "Parallel simulation complete."


# Check for numerical warnings / convergence issues across conditions
# (expects you to implement scripts/stability_check.py)
stability-check:
	@echo "Running numerical stability / warnings check..."
	@PYTHONPATH=$(PYTHONPATH) $(PY) scripts/stability_check.py
	@echo "Stability check complete."

test:
	@echo "Running tests..."
	@PYTHONPATH=$(PYTHONPATH) pytest -q
	@echo "Tests passed."

clean:
	@echo "Cleaning generated files..."
	@rm -rf $(RESULTS_DIR) $(ARTIFACTS_DIR)/knockoffs $(ARTIFACTS_DIR)/params
	@echo "Cleaned."

help:
	@echo "Targets:"
	@echo "  make all           - Run complete pipeline (simulate + analyze + figures)"
	@echo "  make simulate      - Run simulations for all configs in $(CONFIG_DIR)/"
	@echo "  make analyze       - Process raw results into summary tables"
	@echo "  make figures       - Build all figures (keeps your existing figure scripts)"
	@echo "  make clean         - Remove generated files"
	@echo "  make test          - Run test suite (pytest)"
	@echo "  make profile       - Run cProfile on a representative simulation config"
	@echo "  make complexity    - Run timing vs n and build complexity plot"
	@echo "  make benchmark     - Run sequential vs parallel runtime benchmark"
	@echo "  make parallel      - Run optimized simulation with parallelization"
	@echo "  make stability-check - Run numerical stability / warnings check"
	@echo ""
	@echo "Overridable vars: PY, PYTHONPATH, CONFIG_DIR, RESULTS_DIR, ARTIFACTS_DIR,"
	@echo "                  REPS, SEED0, SCRIPT_FIG_1/2, SCRIPT_FIG_SIM"


# --------- Pattern rules / recipes ----------

# Per-condition rule (used only in per-config mode)
#   input : configs/<name>.json
#   output: results/summaries/<name>.csv
$(SUMM_DIR)/%.csv: $(CONFIG_DIR)/%.json $(SIM_SCRIPT_ONE)
	@test -f "$(SIM_SCRIPT_ONE)" || (echo "Missing $(SIM_SCRIPT_ONE)"; exit 1)
	@mkdir -p $(SUMM_DIR) $(ARTIFACTS_DIR)/params
	@echo "Simulating"
	@PYTHONPATH=$(PYTHONPATH) $(PY) $(SIM_SCRIPT_ONE) \
		--config $< \
		--replicates $(REPS) \
		--seed0 $(SEED0) \
		--out $@ \
		--save-config $(ARTIFACTS_DIR)/params/$*.json
	@echo "   ↳ wrote $@"
# Aggregate analysis (depends on all per-condition summaries)

#$(AGG_TABLE): $(SUMMARIES) scripts/analyze_results.py
#	@mkdir -p $(TABLE_DIR)
#	@test -f "scripts/analyze_results.py" || (echo "Missing scripts/analyze_results.py"; exit 1)
#	@echo "Aggregating summaries into $@"
#	@PYTHONPATH=$(PYTHONPATH) $(PY) scripts/analyze_results.py \
#		--summaries $(SUMM_DIR) \
#		--out-table $@
#	@echo "wrote"

# --------- Figure targets----------
fig1:
	@mkdir -p $(dir $(OUT_FIG_1))
	@test -f "$(SCRIPT_FIG_1)" || (echo "Missing $(SCRIPT_FIG_1)"; exit 1)
	@PYTHONPATH=$(PYTHONPATH) $(PY) $(SCRIPT_FIG_1) \
		 --out $(OUT_FIG_1)
	@echo "Wrote $(OUT_FIG_1)"

fig2:
	@mkdir -p $(dir $(OUT_FIG_2))
	@test -f "$(SCRIPT_FIG_2)" || (echo "Missing $(SCRIPT_FIG_2)"; exit 1)
	@PYTHONPATH=$(PYTHONPATH) $(PY) $(SCRIPT_FIG_2) \
		--seed $(SEED0) --out $(OUT_FIG_2)
	@echo "Wrote $(OUT_FIG_2)"





