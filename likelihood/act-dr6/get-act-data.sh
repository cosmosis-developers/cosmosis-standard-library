#!/usr/bin/env bash

pushd "$(dirname "$0")"
if [ -d data/ACTDR6MFLike/v1.0 ]
then
    echo ACT DR6 Lensing data already downloaded
else
    wget https://portal.nersc.gov/project/act/dr6_data/dr6_data.tar.gz
    tar -zxvf dr6_data.tar.gz
    mkdir -p data/ACTDR6MFLike/v1.0
    mv v1.0/dr6_data.fits data/ACTDR6MFLike/v1.0
    rm dr6_data.tar.gz
fi
popd
