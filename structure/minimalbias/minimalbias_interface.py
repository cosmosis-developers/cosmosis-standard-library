"""
Author: Sunao Sugiyama
Last edit: 2023.05.16

This module computes the 2x2pt observables in physical scale. This choice is used in HSC weak lensing analysis.
The observables are the galaxy-galaxy lensign, dSigma(r_p), and the projected correlation function, w_p(r_p).
This module returns the observables after measureemnt corrections:
    1. correction on projected radial bin length r_p -> f * r_p, where f is the ratio 
        of comoving distance at the lens redshift and correct for the difference of cosmologies 
        in measurement and modeling.
    2. correction on projection length of w_p, pimax -> f * pimax, where f is the ratio of 
        Hubble parameter at the lens redshift and correct for the difference of cosmologies 
        in measurement and modeling.
    3. correction on overall dSigma amplitude, dSigma -> f * dSigma, where f is the ratio of 
        Sigma_crit values averaged over all lens-source pairs, and corrects for the differene of 
        cosmologies and photometric redshift in measurement and modeling.
This module also take account of the averaging effect within the finite radial bin.

The radial bins are read in this module at the time of setup, and the correction factors introduced above are all computed in a module of 'shear/meascorr/meascorr_interface.py', and the results are saved in data block.

Code reviewed by Tianqing Zhang in Oct 2023
"""
from cosmosis.datablock import names, option_section
from minimalbias import minimalbias_class
import numpy as np
import os, sys
dirname = os.path.split(__file__)[0]
twopoint_path = os.path.join(dirname,"..","..","likelihood","2pt")
sys.path.insert(0, twopoint_path)
import twopoint
import time

def read_rp(filename, xi_type_2pt, theta_type = 'centers'):
    """
    Read the projected radial bins from data file.
    """
    T = twopoint.TwoPointFile.from_fits(filename)
    xi = T.get_spectrum(xi_type_2pt)
    # get the theta values for a single bin pair.
    # This method currently assumes that the same binning is used for all bin pairs (in a given corr funct).
    # scale cuts are applied later and can be different for each bin pair. 
    pairs = xi.get_bin_pairs()
    indices = xi.get_pair_mask(pairs[0][0], pairs[0][1]) #use the first appearing pair, since all should be the same.
    if theta_type == 'centers':
        thetas = xi.angle[indices]
        return thetas
    elif theta_type == 'edges':
        theta_min = xi.angle_min[indices]  # array of min values
        theta_max = xi.angle_max[indices] # array of max values
        # We currently assume adjacent and non-overlapping bins.
        np.testing.assert_allclose(theta_min[1:], theta_max[:-1], rtol=1e-03,
            err_msg='theta bin min and max values do not match.', verbose=True)
        theta_bins = np.append(theta_min, theta_max[-1])
        return theta_bins
    else:
        raise ValueError("Unknown theta_type in read_theta")
        
        
def setup(options):
    
    # lens redshift: Note that dSigma and wp are independent of source redshift.
    redshifts = options.get_double_array_1d(option_section, 'redshifts')
    
    # Choice whether we average the signal within finite radial bin.
    bin_avg = options.get_bool(option_section, "bin_avg", False)
    
    # radial bins in reference cosmology for dSigma (ds) and wp
    theta_file = options.get_string(option_section, "theta_file", "")
    rp, rp_edges = dict(), dict()
    if bin_avg:
        rp['ds'] = read_rp(theta_file, 'ds', theta_type = 'centers')
        rp['wp'] = read_rp(theta_file, 'wp'    , theta_type = 'centers')
        rp_edges['ds'] = read_rp(theta_file, 'ds', theta_type = 'edges')
        rp_edges['wp'] = read_rp(theta_file, 'wp'    , theta_type = 'edges')
    else:
        rp['ds'] = read_rp(theta_file, 'ds', theta_type = 'centers')
        rp['wp'] = read_rp(theta_file, 'wp'    , theta_type = 'centers')
        rp_edges['ds'] = None
        rp_edges['wp'] = None
    
    # projection length
    pimax = options.get_double(option_section, 'pimax', default=100.0)
    
    # initialize minimal bias module
    mb_config  = {'do_Kaiser': options.get_bool(option_section, 'do_Kaiser', default=True), 
                  'verbose'  : options.get_bool(option_section, 'verbose', default=True)}
    mb = minimalbias_class(mb_config)
    
    
    # Section names
    # These names are used to extract/incept the results from/to the block.
    # These names must be consistently defined in ini file among different modules.
    section_names = {}
    # inputs
    # Name of samples used for lens and source
    lens_sample   = options.get_string(option_section, "lens_sample", "")
    section_names['lens_sample'] = lens_sample
    section_names['nz_lens']   = "NZ_" + lens_sample.upper()
    source_sample = options.get_string(option_section, "source_sample", "")
    section_names['source_sample'] = source_sample
    section_names['nz_source'] = "NZ_" + source_sample.upper()
    # galaxy bias
    section_names['galaxy_bias'] = options.get_string(option_section, "galaxy_bias", "galaxy_bias_parameters")
    # measurement correction
    section_names['f_wp'] = options.get_string(option_section, "meascorr_wp", "f_wp")
    section_names['f_rp'] = options.get_string(option_section, "meascorr_rp", "f_rp")
    section_names['f_ds'] = options.get_string(option_section, "meascorr_ds", "f_ds")
    # outputs
    section_names['wp_out'] = options.get_string(option_section, "wp_out", "galaxy_xi")
    section_names['ds_out'] = options.get_string(option_section, "ds_out", "galaxy_shear_xi")
    # save name
    section_names['wp_save_name'] = options.get_string(option_section, "wp_save_name", "")
    section_names['ds_save_name'] = options.get_string(option_section, "ds_save_name", "")
    
    return redshifts, rp, rp_edges, mb, pimax, bin_avg, section_names
    
    
def execute(block, config):
    time_start = time.time()
    
    # Get configuration
    redshifts, rp, rp_edges, mb, pimax, bin_avg, section_names = config
    # number of redshift bins for lens and source.
    nbin_lens = block[section_names['nz_lens'], 'nbin']
    nbin_srce = block[section_names['nz_source'], 'nbin']
    
    # matter power spectra
    z_lin , k_lin , P_lin  = block.get_grid(names.matter_power_lin, "z", "k_h", "P_k")
    z_nlin, k_nlin, P_nlin = block.get_grid(names.matter_power_nl, "z", "k_h", "P_k")
    mb.set_pk_lin_data(z_lin , k_lin , P_lin)
    mb.set_pk_nlin_data(z_nlin, k_nlin, P_nlin)
    
    # z & chi relation
    # print(block.sections())
    # print(names.growth_parameters)
    z   = block[names.growth_parameters, "z"]
    f   = block[names.growth_parameters, "f_z"]
    
    mb.set_z2f(z, f)
    
    # unpack radial bin
    if bin_avg:
        # We input the lower edge of each radial bin
        rp_wp = rp_edges['wp'][:-1]
        rp_ds = rp_edges['ds'][:-1]
        dlnrp_wp = np.log(rp_wp[1]/rp_wp[0])
        dlnrp_ds = np.log(rp_ds[1]/rp_ds[0])
    else:
        # We input the center each radial bin
        rp_wp = rp['wp']
        rp_ds = rp['ds']
        dlnrp_wp = 0.0
        dlnrp_ds = 0.0
    # iterate over all lens-source bin pair
    for i in range(nbin_lens):
        # Get representative redshift of i-th lens bin
        zl = redshifts[i]
        
        # Linear galayx bias
        b1  = block[section_names['galaxy_bias'], 'b1_{0}'.format(i+1)]
        Omm = block[names.cosmological_parameters, "Omega_m"]
        mb.set_param(Omm, b1)
        
        # correction factors 
        f_rp = block.get_double(section_names['f_rp'], 'bin_{0}'.format(i+1), 1.0)
        f_wp = block.get_double(section_names['f_wp'], 'bin_{0}'.format(i+1), 1.0) # multiplied to pimax
        # update wp
        # print('enter get_wp()', i)
        block[section_names['wp_out'], 'bin_{0}_{0}'.format(i+1)] = mb.get_wp(zl, f_rp*rp_wp, dlnrp_wp, f_wp*pimax)
        # print('exit get_wp()', i)
        
        # print(mb.get_wp(zl, f_rp*rp_wp, dlnrp_wp, f_wp*pimax))
        # Compute the dSigma without the correction of Sigmacrit
        ds = mb.get_ds(zl, f_rp*rp_ds, dlnrp_ds)
        
        # apply correction to dSigma overall amplitude
        for j in range(nbin_srce):
            f_ds = block.get_double(section_names['f_ds'], 'bin_{0}_{1}'.format(i+1,j+1), 1.0)
            block[section_names['ds_out'], 'bin_{0}_{1}'.format(i+1,j+1)] = f_ds * ds
    
    # Write extra output to be used other modules
    for n in ['wp', 'ds']:
        o = section_names['{0}_out'.format(n)]
        block[o, "nbin_a"] = {'wp':nbin_lens, 'ds':nbin_lens}[n]
        block[o, "nbin_b"] = {'wp':nbin_lens, 'ds':nbin_srce}[n]
        block[o, "sample_a"] = {'wp': section_names['lens_sample'], 'ds': section_names['lens_sample']}[n]
        block[o, "sample_b"] = {'wp': section_names['lens_sample'], 'ds': section_names['source_sample']}[n]
        block[o, "is_auto"] = {'wp':True, 'ds':False}[n]
        block[o, "rp"] = rp[n]
        block.put_metadata(o, "rp", "unit", "rad")
        block[o, "sep_name"] = "rp"
        block[o, "save_name"] = section_names['{}_save_name'.format(n)]
        block[o, "bin_avg"] = bin_avg
        if bin_avg:
            block[o, "rp_edges"] = rp_edges[n]
            block.put_metadata(o, "rp_edges", "unit", "rad")
    
    return 0