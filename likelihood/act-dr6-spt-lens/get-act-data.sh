#!/usr/bin/env bash

pushd "$(dirname "$0")"
if [ -d data/v1.2 ]
then
    echo ACT DR6 + SPT Lensing data already downloaded
else
    mkdir -p data
    pushd data
    wget https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/ACT_dr6_likelihood_v1.2.tgz
    tar -zxvf ACT_dr6_likelihood_v1.2.tgz
    rm ACT_dr6_likelihood_v1.2.tgz
    popd
fi
popd
