#!/usr/bin/env bash

pushd "$(dirname "$0")"
if [ -d data/ACTDR6CMBonly/v1.0 ]
then
    echo ACT DR6 Lite data already downloaded
else
    wget https://lambda.gsfc.nasa.gov/data/act/pspipe/sacc_files/dr6_data_cmbonly.tar.gz
    tar -zxvf dr6_data_cmbonly.tar.gz
    mkdir -p data/ACTDR6CMBonly/v1.0
    mv v1.0/dr6_data_cmbonly.fits data/ACTDR6CMBonly/v1.0
    rm dr6_data_cmbonly.tar.gz
fi
popd