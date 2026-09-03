.PHONY: assets pdf supplement clean
assets:
	python scripts/build_paper_assets.py
pdf: assets
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
supplement:
	cd paper/supplement && latexmk -pdf -interaction=nonstopmode supplement.tex
clean:
	cd paper && latexmk -C || true
	cd paper/supplement && latexmk -C || true
