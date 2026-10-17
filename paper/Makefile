.PHONY: all figures clean

all: main.pdf

main.pdf: main.tex refs.bib $(wildcard sections/*.tex) figures/fig_scores.pdf
	tectonic main.tex

# Figures are drawn from the raw result CSVs in ../results by this script.
figures figures/fig_scores.pdf: figures/make_figures.py
	uv run --frozen --with matplotlib python figures/make_figures.py

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out
