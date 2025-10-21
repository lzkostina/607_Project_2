PY            ?= python
PYTHONPATH    ?= .
SCRIPT_FIG_1  ?= scripts/plot_knockoff_pairs.py
SCRIPT_FIG_2  ?= scripts/figure2_lasso_path.py
OUT_FIG_1     ?= artifacts/knockoffs/fig1_synthetic.png
OUT_FIG_2     ?= artifacts/knockoffs/figure2_lasso_path.png
SEED          ?= 1

P    ?= 160
K    ?= 18
T    ?= 1.5
STAT ?= signed_max   # or: diff

.PHONY: all fig1 fig2 clean

all: fig1 fig2

fig1:
	@mkdir -p $(dir $(OUT_FIG_1))
	@test -f "$(SCRIPT_FIG_1)" || (echo "Missing $(SCRIPT_FIG_1)"; exit 1)
	@PYTHONPATH=$(PYTHONPATH) $(PY) $(SCRIPT_FIG_1) --p $(P) --k $(K) --t $(T) --stat $(STAT) --out $(OUT_FIG_1)
	@echo "Wrote $(OUT_FIG_1)"

fig2:
	@mkdir -p $(dir $(OUT_FIG_2))
	@test -f "$(SCRIPT_FIG_2)" || (echo "Missing $(SCRIPT_FIG_2)"; exit 1)
	@PYTHONPATH=$(PYTHONPATH) $(PY) $(SCRIPT_FIG_2) --seed $(SEED) --out $(OUT_FIG_2)
	@echo "Wrote $(OUT_FIG_2)"

clean:
	@rm -rf artifacts/knockoffs
	@echo "Cleaned artifacts/knockoffs"

