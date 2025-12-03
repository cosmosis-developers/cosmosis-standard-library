include ${COSMOSIS_SRC_DIR}/config/compilers.mk

USER_CFLAGS= -I ../../shear/limber -I ${GSL_INC}
USER_LDFLAGS= -L../../shear/limber -llimber -lcosmosis -L ${GSL_LIB} -lgsl -lgslcblas 

all: interface.so

interface.so: nla_higher_order.c ../../shear/limber/liblimber.so
	$(CC) $(CFLAGS)  -shared -o interface.so nla_higher_order.c $(LDFLAGS)

clean:
	rm -rf interface.so interface.so.dSYM

test:
	@echo la_model_higher_order has no tests

