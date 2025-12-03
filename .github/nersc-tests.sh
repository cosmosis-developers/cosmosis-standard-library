#!/usr/bin/env bash
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -N 2
#SBATCH --mail-type=FAIL


ENV=../env

rm -rf output/demo5* output/pantheon* output/mn*


source $CFS/des/zuntz/cosmosis-global/setup-cosmosis3 $ENV

set -ex


#emcee tests - general mpi4py
export OMP_NUM_THREADS=2

# Test sampler
cosmosis demos/demo5.ini -p runtime.sampler=test

# Test on a single node
srun -u -N 1 -n 32 cosmosis --mpi demos/demo5.ini -p emcee.nsteps=5 emcee.samples=50
rm -rf output/demo5*

# Test on two nodes
srun -u -N 2 -n 32 cosmosis --mpi demos/demo5.ini -p emcee.nsteps=5 emcee.samples=50
rm -rf output/demo5*



# multinest

# one nodes
srun -u -N 1 -n 32 cosmosis --mpi examples/pantheon.ini -p runtime.sampler=multinest
rm -rf output/pantheon* output/mn*

# two nodes
srun -u -N 2 -n 32 cosmosis --mpi examples/pantheon.ini -p runtime.sampler=multinest
rm -rf output/pantheon* output/mn*



# polychord
mkdir -p clusters

# one nodes
srun -u -N 1 -n 32 cosmosis --mpi examples/pantheon.ini -p runtime.sampler=polychord
rm -rf output/pantheon*

# two nodes
srun -u -N 2 -n 32 cosmosis --mpi examples/pantheon.ini -p runtime.sampler=polychord
rm -rf output/pantheon*



# des-y3

export OMP_NUM_THREADS=8
cosmosis examples/des-y3.ini



# planck
if [ ! -f "likelihood/planck2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik/_mdb" ]; then
    ./examples/get-planck-data.sh
fi

cosmosis examples/planck.ini

echo All tasks complete
