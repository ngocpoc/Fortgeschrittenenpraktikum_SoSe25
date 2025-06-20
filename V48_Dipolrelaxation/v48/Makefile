all: build/v48.pdf

# hier Python-Skripte:
build/Heizraten.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py


# hier weitere Abhängigkeiten für build/vXXX.pdf deklarieren:
build/v48.pdf: build/Heizraten.pdf
build/v48.pdf: build/ersterStrom.pdf
build/v48.pdf: build/zweiterStromVor.pdf
build/v48.pdf: build/ersterUntergrundfrei.pdf
build/v48.pdf: build/zweiterUntergrundfrei.pdf
build/v48.pdf: build/polarisation1.pdf
build/v48.pdf: build/polarisation2.pdf
build/v48.pdf: build/stromdichte1.pdf
build/v48.pdf: build/stromdichte2.pdf
build/v48.pdf: build/relaxation.pdf

build/v48.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	v48.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean