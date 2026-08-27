include ${COSMOSIS_SRC_DIR}/config/compilers.mk

PYTHON ?= python
PYTHONINSTALL = $(PWD)/ee_install

DATA = ee2_bindata.dat
HEADERS = src/cosmo.h src/emulator.h src/units_and_constants.h
SRC = src/euclidemu2.pyx src/cosmo.cxx src/emulator.cxx

all: ee_install/done

ee_install/done: $(SRC) $(HEADERS) $(DATA)
	export CC=$(CC) CXX=$(CXX); export PYTHONPATH=${PYTHONPATH}:$(PYTHONINSTALL);  $(PYTHON) -m pip install  -vv --no-deps   --target=${PYTHONINSTALL} .
	touch ee_install/done

clean:
	rm -rf build dist ee_install

.PHONY: clean all
