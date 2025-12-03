include ${COSMOSIS_SRC_DIR}/config/compilers.mk


VERSION=0031e3604f0c
DEFINES = -D HAS_LAPACK -D LAPACK_CLIK -D NOHEALPIX -D CLIK_LENSING -D 'CLIKSVNVERSION="$(VERSION)"' -D CAMSPEC_V1

PLANCK_DIR=plc-3.0

RPATH_FLAGS=-Wl,-rpath,$(PWD)/$(PLANCK_DIR)/lib -Wl,-rpath,$(PWD)/$(PLANCK_DIR)/buildir
USER_LDFLAGS=-L$(PLANCK_DIR)/lib -lcosmosis $(RPATH_FLAGS) -L$(CFITSIO_LIB) -lcfitsio $(LAPACK_LINK)

PLANCK_LIB=lib/libclik.so
USER_CFLAGS=-I$(PLANCK_DIR)/include  -D_GNU_SOURCE $(DEFINES)


all: planck_interface.so

data/hi_l:
	mkdir -p data
	curl "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R3.00.tar.gz" -o data/COM_Likelihood_Data-baseline_R3.00.tar.gz
	cd data && tar -zxf COM_Likelihood_Data-baseline_R3.00.tar.gz
	mv data/baseline/plc_3.0/* data/
	rm -r data/baseline data/COM_Likelihood_Data-baseline_R3.00.tar.gz

data: data/hi_l

$(PLANCK_DIR)/$(PLANCK_LIB): 
	cd $(PLANCK_DIR) && $(MAKE) install

planck_interface.so: $(PLANCK_DIR)/$(PLANCK_LIB) planck_interface.c
	$(CC) -shared $(CFLAGS) -o $@ planck_interface.c -lclik $(LDFLAGS)

clean:
	rm -rf *.o *.so *.dSYM
	rm -rf $(PLANCK_DIR)/lib
	rm -rf $(PLANCK_DIR)/bin
	rm -rf $(PLANCK_DIR)/include
	rm -rf $(PLANCK_DIR)/share
	cd $(PLANCK_DIR) && $(MAKE) clean

test:
	@echo Alas, planck has no tests

.PHONY: data all clean test
