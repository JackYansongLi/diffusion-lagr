PYTHON ?= python
export MPLCONFIGDIR := repro/.cache/matplotlib
.PHONY: test analyze plots report

test:
	$(PYTHON) -m unittest discover -s repro/tests -v

analyze:
	$(PYTHON) -m repro.validate_demo
	$(PYTHON) -m repro.explore_estimators

plots:
	$(PYTHON) -m repro.plot_audit

report:
	$(PYTHON) -m repro.report_tables
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
