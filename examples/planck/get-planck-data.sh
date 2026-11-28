#!/usr/bin/env bash
set -x

pushd likelihood/planck2018
URL="https://portal.nersc.gov/cfs/lsst/zuntz/planck/COM_Likelihood_Data-baseline_R3.00.tar.gz"
FILENAME="COM_Likelihood_Data-baseline_R3.00.tar.gz"
wget -O ${FILENAME} ${URL}
tar -zxvf ${FILENAME}
popd
