"""
Author: Sunao Sugiyama
Last edit: 2023.05.16

This does almost the same procedure as structure/minimalbias/minimalbias_interface.py: 
here we apply measurement corrections to the magnification bias term of dSigma.

Code reviewed by Tianqing Zhang in Oct 2023
"""
from cosmosis.datablock import names, option_section
from magnification_phys import magnification_class
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
    """
    Defines the setups to calcurate the magnification bias term, and 
    instantiate necessary quantities.
    
    Now this module adds the magnification bias term only to g-g lensing signal, 
    but not to the galaxy clustering wp. (This is because that contribution is 
    small especially given the covariance of HSC weak lensing analysis.) 
    However, I (SS) leave some lines for wp for the future purpose.
    """
    
    # lens redshift: Note that dSigma and wp are independent of source redshift.
    redshifts = options.get_double_array_1d(option_section, 'redshifts')
    
    # Choice whether we average the signal within finite radial bin.
    bin_avg = options.get_bool(option_section, "bin_avg", False)
    save_mag = options.get_bool(option_section, "save_mag", False)
    
    # radial bins in reference cosmology for dSigma (ds) and wp
    # In reality these radial bins are defined in unit of Mpc/h in the file,
    # but for the compatibility with CosmoSIS, we need to rename these observables
    # gammat and wtheta in unit of arcmin.
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
    
    # initialize magnification bias module
    config = {'verbose': options.get_bool(option_section, "verbose", True)}
    mag = magnification_class(config)
    
    
    # Section names
    # These names are used to extract/incept the results from/to the block.
    # These names must be consistently defined in ini file among different modules.
    section_names = {}
    # inputs
    # Name of samples used for lens and source
    name = options.get_string(option_section, "lens_sample", "")
    section_names['nz_lens']   = "NZ_" + name.upper()
    name = options.get_string(option_section, "source_sample", "")
    section_names['nz_source'] = "NZ_" + name.upper()
    # galaxy bias
    section_names['magnification_bias'] = options.get_string(option_section, "magnification_bias", "magnification_bias_parameters")
    # measurement correction
    section_names['f_wp'] = options.get_string(option_section, "f_wp", "f_wp")
    section_names['f_rp'] = options.get_string(option_section, "f_rp", "f_rp")
    section_names['f_ds'] = options.get_string(option_section, "f_ds", "f_ds")
    # outputs
    section_names['wp_out'] = options.get_string(option_section, "wp_out", "galaxy_xi")
    section_names['ds_out'] = options.get_string(option_section, "ds_out", "galaxy_shear_xi")
    
    # print(redshifts, rp, rp_edges, mag, pimax, bin_avg, section_names)
    
    return redshifts, rp, rp_edges, mag, pimax, bin_avg, section_names, save_mag
    
def execute(block, config):
    """
    Name the section of magnification biass parameter as "magnification" 
    so that this modules can read the parameter.
    """
    time_start = time.time()
    
    # Get configuration
    # Get configuration
    redshifts, rp, rp_edges, mag, pimax, bin_avg, section_names, save_mag = config
    
    # number of redshift bins for lens and source.
    nbin_lens = block[section_names['nz_lens'], 'nbin']
    nbin_srce = block[section_names['nz_source'], 'nbin']
    
    # power spectra
    z_nlin, k_nlin, P_nlin = block.get_grid(names.matter_power_nl, "z", "k_h", "P_k")
    mag.set_pk_nlin_data(z_nlin, k_nlin, P_nlin)
    # print("mag.set_pk_nlin_data(z_nlin, k_nlin, P_nlin)", z_nlin, k_nlin, P_nlin)
    # comoving distance
    h0  = block[names.cosmological_parameters, 'h0']
    z   = block[names.distances, "z"]
    # print(names.distances)
    # print(block.keys(section  = 'distances'))
    chi = block[names.distances, "D_C"] * h0
    # chi = block[names.distances, "d_l"] * block[names.distances, "a"] * h0
    # print("mag.set_z2chi(z, chi)", z, chi)
    mag.set_z2chi(z, chi)
    
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
        
        # magnification bias parameter
        alpha  = block[section_names['magnification_bias'], 'alpha_{0}'.format(i+1)]
        Omm = block[names.cosmological_parameters, "Omega_m"]
        mag.set_param(alpha, Omm)
        
        # lens redshift distribution
        mag.set_nz_lens(block[section_names['nz_lens'], "z"], block[section_names['nz_lens'], "bin_{0}".format(i+1)])
        
        # correction factors 
        f_rp = block.get_double(section_names['f_rp'], 'bin_{0}'.format(i+1), 1.0)
        f_wp = block.get_double(section_names['f_wp'], 'bin_{0}'.format(i+1), 1.0) # multiplied to pimax
        
        # update wp
        block[section_names['wp_out']+'_gg', 'bin_{0}_{0}'.format(i+1)]  = block[section_names['wp_out'], 'bin_{0}_{0}'.format(i+1)]
        block[section_names['wp_out']      , 'bin_{0}_{0}'.format(i+1)] += 0.0 #mag.get_wp_mag(z, rp ,dlnrp,f_wp*pimax)
        
        # compute magnification bias and apply correction to dSigma overall amplitude
        for j in range(nbin_srce):
            # print(i,j)
            # print(block[section_names['nz_source'], "z"], block[section_names['nz_source'], "bin_{0}".format(j+1)])
            mag.set_nz_source(block[section_names['nz_source'], "z"], block[section_names['nz_source'], "bin_{0}".format(j+1)])
            f_ds = block.get_double(section_names['f_ds'], 'bin_{0}_{1}'.format(i+1,j+1), 1.0)
            block[section_names['ds_out']+'_gG', 'bin_{0}_{1}'.format(i+1,j+1)]  = block[section_names['ds_out'], 'bin_{0}_{1}'.format(i+1,j+1)]
            block[section_names['ds_out']      , 'bin_{0}_{1}'.format(i+1,j+1)] += f_ds * mag.get_ds_mag(zl, f_rp*rp_ds, dlnrp_ds)
            
            if save_mag:
                block['ds_mag', 'bin_{0}_{1}'.format(i+1,j+1)] = mag.get_ds_mag(zl, f_rp*rp_ds, dlnrp_ds)
    
    return 0