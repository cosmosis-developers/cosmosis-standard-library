#!/usr/bin/env bash
set -e
set -x

pushd likelihood/wmap9/data

echo Downloading WMAP data. This can take up to 10 minutes from some locations.

wget https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/likelihood/wmap_likelihood_full_v5.tar.gz
tar -zxvf wmap_likelihood_full_v5.tar.gz

# remove the likelihood code itself as we already have a copy
mv wmap_likelihood_v5/data/* .

# clean up
rm -rf wmap_likelihood_v5 wmap_likelihood_full_v5.tar.gz
popd
