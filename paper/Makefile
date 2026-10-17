SVG := $(wildcard ../results/chart_*.svg ../results/fig_*.svg)
FIGS := $(patsubst ../results/%.svg,figures/%.pdf,$(SVG))

.PHONY: all figures clean

all: main.pdf

main.pdf: main.tex refs.bib $(wildcard sections/*.tex) $(FIGS)
	tectonic main.tex

figures: $(FIGS)

figures/%.pdf: ../results/%.svg
	@mkdir -p figures
	rsvg-convert -f pdf -o $@ $<

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out
