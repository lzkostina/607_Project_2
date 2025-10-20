PY      ?= python
SCRIPT  ?= scripts/plot_knockoff_pairs.py
OUT     ?= artifacts/knockoffs/fig1_synthetic.png

# light knobs (optional)
P ?= 160
K ?= 18
T ?= 1.5
STAT ?= signed_max   # or: diff

.PHONY: fig1 clean

fig1:
	@mkdir -p $(dir $(OUT))
	@$(PY) $(SCRIPT) --p $(P) --k $(K) --t $(T) --stat $(STAT) --out $(OUT)
	@echo "Wrote $(OUT)"

clean:
	@rm -rf artifacts/knockoffs
