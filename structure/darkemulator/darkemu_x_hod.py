"""
Author: Sunao Sugiyama
Last edit: 2023.06.07
See Nishimichi et al. (https://arxiv.org/pdf/1811.09504.pdf) for detail.
"""
from cosmosis.datablock import names, option_section
from dark_emulator import model_hod
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
    
    # initialize model_hod bias module
    darkemu_x_hod = model_hod.darkemu_x_hod()

    # Whether we apply thee Kaiser correction or not
    do_Kaiser = options.get_bool(option_section, 'do_Kaiser', default=True)

    # Type of distribution of satellite galaxies 
    sat_dist_type = options.get_string(option_section, 'sat_dist_type', default='NFW')

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
    section_names['hod'] = options.get_string(option_section, "hod", "hod_parameters")
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
    
    return redshifts, rp, rp_edges, darkemu_x_hod, pimax, bin_avg, section_names, do_Kaiser, sat_dist_type
   

    
def execute(block, config):
    time_start = time.time()
    
    # Get configuration
    redshifts, rp, rp_edges, darkemu_x_hod, pimax, bin_avg, section_names, do_Kaiser, sat_dist_type = config
    
    # number of redshift bins for lens and source.
    nbin_lens = block[section_names['nz_lens'], 'nbin']
    nbin_srce = block[section_names['nz_source'], 'nbin']

    # Set cosmological parameter
    pars = names.cosmological_parameters
    params= np.array([block[pars, 'ombh2'], 
                      block[pars, 'omch2'],
                      block[pars, 'omega_lambda'],
                      block[pars, 'log1e10As'],
                      block[pars, 'n_s'],
                      block[pars, 'w']])
    darkemu_x_hod.set_cosmology(params.reshape((1,6)))
    
    # unpack radial bin
    if bin_avg:
        # We input the lower edge of each radial bin
        rp_wp = rp_edges['wp']
        rp_ds = rp_edges['ds']
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
        hod_params = {'logMmin'      : block[section_names['hod'], 'logMmin_{}'.format(i+1)], 
                      'sigma_sq'     : block[section_names['hod'], 'sigma_sq_{}'.format(i+1)], 
                      'logM1'        : block[section_names['hod'], 'logM1_{}'.format(i+1)], 
                      'alpha'        : block[section_names['hod'], 'alpha_{}'.format(i+1)],
                      'kappa'        : block[section_names['hod'], 'kappa_{}'.format(i+1)],
                      'poff'         : block[section_names['hod'], 'poff_{}'.format(i+1)], 
                      'Roff'         : block[section_names['hod'], 'Roff_{}'.format(i+1)],
                      'sat_dist_type': sat_dist_type, 
                      'alpha_inc'    : block[section_names['hod'], 'alpha_inc_{}'.format(i+1)],
                      'logM_inc'     : block[section_names['hod'], 'logM_inc_{}'.format(i+1)]}
        darkemu_x_hod.set_galaxy(hod_params)
        
        # correction factors 
        f_rp = block.get_double(section_names['f_rp'], 'bin_{0}'.format(i+1), 1.0)
        f_wp = block.get_double(section_names['f_wp'], 'bin_{0}'.format(i+1), 1.0) # multiplied to pimax
        
        # update wp
        block[section_names['wp_out'], 'bin_{0}_{0}'.format(i+1)] = darkemu_x_hod.get_wp(f_rp*rp_wp, zl, pimax=f_wp*pimax, dlnrp=dlnrp_wp, rsd=do_Kaiser)
            
        # Compute the dSigma without the correction of Sigmacrit
        ds = darkemu_x_hod.get_ds(f_rp*rp_ds, zl, dlnrp=dlnrp_ds)
        
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
        if n in rp_edges:
            block[o, "rp_edges"] = rp_edges[n]
            block.put_metadata(o, "rp_edges", "unit", "rad")
    
    return 0


