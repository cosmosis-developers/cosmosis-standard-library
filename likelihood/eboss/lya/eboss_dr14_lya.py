from numpy import log, pi, interp, where, loadtxt,dot, append, linalg, inf
import os
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
from scipy.interpolate import bisplrep, bisplev

cosmo = section_names.cosmological_parameters
likes = section_names.likelihoods
dist = section_names.distances

c_km_per_s =  299792.458

dirname = os.path.dirname(os.path.realpath(__file__))


def setup(options):
    section = option_section
    feedback = options.get_bool(section, "feedback", default=False)
    
    # read alphas and chi2 table
    lya_filename = os.path.join(dirname, 'lya_combined_2019_chi2.txt')
    fid_filename = os.path.join(dirname, 'desteagathe_2019_fiducial.txt')

    alpha_parallel, alpha_transverse, chi2_data = loadtxt(lya_filename).T
    z_eff_fid, dh_rd_fid, dm_rd_fid = loadtxt(fid_filename)

    dm_rd_data = alpha_transverse * dm_rd_fid
    dh_rd_data = alpha_parallel * dh_rd_fid

    # The likelihood is not a Gaussian here.  For some reason this is presented
    # in a table of chi^2 values, which are presumably -2log(L) values.
    # We build the interpolator which we use to read into that table here.
    # Outside the table range we use -inf
    chi2_interp = bisplrep(dm_rd_data, dh_rd_data, chi2_data, kx=3, ky=3, s=0)

    return chi2_interp, z_eff_fid, feedback, dm_rd_data, dh_rd_data

def execute(block, config):
    chi2_interp, z_eff_fid, feedback, dm_rd_data, dh_rd_data = config
        
    # theory
    z = block[dist, 'z']
    rd = block[dist, "rs_zdrag"]
    d_m = block[dist, 'd_m']
    h = c_km_per_s*block[dist, 'h']
    d_h = c_km_per_s/h
        
    # likelihood on D_m, D_h not Gaussian
    # read in the chi2 table provided on https://github.com/igmhub/picca/
    # transform alphas table to Dm, Dh

    # Get the predicted values of D_M and D_H at the fiducial redshift
    dm_z = interp(z_eff_fid, z, d_m)
    dh_z = interp(z_eff_fid, z, d_h)
    dm_z_rd_predicted = dm_z / rd
    dh_z_rd_predicted = dh_z / rd

    # Interpolate to get the actual likelihood
    chi2 = bisplev(dm_z_rd_predicted, dh_z_rd_predicted, chi2_interp)

    block[likes,'eboss14_lya_like'] = -chi2 / 2
    
    return 0

def cleanup(config):
    return 0

