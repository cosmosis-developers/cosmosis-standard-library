# coding:utf-8
import os
import numpy as np
from cosmosis.datablock import names, option_section, BlockError
import re
import sys
import scipy.interpolate as interp

# for timing
from timeit import default_timer as timer
from datetime import timedelta
from typing import Annotated, Any

import pyccl as ccl
cosmo = names.cosmological_parameters
ia = names.intrinsic_alignment_parameters
bias = names.bias_field



def setup(options):
    # General options: define ells, Liber/non-Limber, which Cl's
    # which nl modelling
    # TODO: crazy spline parameters which control the accuracy, all default for now
    
    config = {}
    config['compute_gc'] = True
    config['compute_shear'] = True
    config['compute_cross'] = True
    config['transfer_function'] = 'boltzmann_camb'
    config['matter_power_spectrum'] = 'camb'
    config['extra_parameters'] = {"camb": {"halofit_version": "mead2020" }}
    config['baryonic_effects'] = None
    config['mg_parametrization'] = None


    ell = np.array([])
    ell_min_logspaced = options.get_double(option_section, "ell_min_logspaced", -1.)
    ell_max_logspaced = options.get_double(option_section, "ell_max_logspaced", -1.)
    n_ell_logspaced = options.get_int(option_section, "n_ell_logspaced", -1)
    if n_ell_logspaced>0:
        assert ell_min_logspaced>0.
        assert ell_max_logspaced>0.
        ell = np.logspace(np.log10(ell_min_logspaced), np.log10(ell_max_logspaced), n_ell_logspaced)
    config['ell'] = ell
    config['n_ell'] = n_ell_logspaced
    return config
        



def call_ccl(block, config):
    ccl_args = {
            "Omega_c": block.get_double(cosmo, "omega_c"),
            "Omega_b": block.get_double(cosmo, "omega_b"),
            "h": block.get_double(cosmo, "h0"),
            "n_s": block.get_double(cosmo, 'n_s'),
            "Omega_k": block.get_double(cosmo, "omega_k", default=0.),
            "Neff": block.get_double(cosmo, "nnu", default=3.044),
            "w0": block.get_double(cosmo, "w", default=-1.),
            "wa": block.get_double(cosmo, 'wa', default=0.0),
            "T_CMB": block.get_double(cosmo, 'TCMB'),
            "m_nu": block.get_double(cosmo, 'mnu'),
            #"mass_split": 'normal' #'single', 'equal', 'normal', 'inverted'

            "transfer_function": config['transfer_function'], # 'bbks' / 'boltmann_class'/ 'boltzmann_camb'/ 'boltzmann_isitgr' or EmulatorPk
            "matter_power_spectrum": config['matter_power_spectrum'], # 'linear',  'halofit', 'camb' or EmulatorPk (e.g., ccl.BaccoemuNonlinear())
            "extra_parameters": config['extra_parameters'],    #None or {"camb": {"halofit_version": "mead2020_feedback" or "mead2020",
                                        #"HMCode_A_baryon": 3.13,
                                        #"HMCode_eta_baryon": 0.603,
                                        #"HMCode_logT_AGN": 7.8,
                                        #"kmax": 100,
                                        #"lmax": 5000,
                                        #"dark_energy_model": "ppf"}}
            "baryonic_effects": config['baryonic_effects'], #None or Baryon-object
            "mg_parametrization": config['mg_parametrization'], #None or ModifiedGravity-object (e.g.,  MuSigmaModel())
        }
    
    if block.has_value(cosmo, "a_s"):
        ccl_args["A_s"] = block.get_double(cosmo, "a_s")
    if block.has_value(cosmo, "sigma_8"):
        ccl_args["sigma8"] = block.get_double(cosmo, "sigma_8")
    assert ("A_s" in ccl_args) or ("sigma8" in ccl_args)
    

    cosmo_ccl = ccl.Cosmology(**ccl_args)
    #print(ccl.boltzmann.get_camb_pk_lin(cosmo_ccl, nonlin=True ) )

    print('cosmo_ccl initialised')
    if config['compute_shear']:
        section_name = "nz_source"
        nbin_source = block[section_name, "nbin"]
        z = block[section_name, "z"]
        bias_ia = block.get_double(ia, "a1") * ((1. + z)/(1. + block.get_double(ia, "z_piv"))) ** block.get_double(ia, "alpha1")
        sources = [ccl.WeakLensingTracer(cosmo_ccl, dndz=(z, block[section_name, "bin_%d"%i]),
                                    ia_bias=(z, bias_ia)) for i in range(1, nbin_source+1)]

    print('wl tracers computed')    
    if config['compute_gc']:
        section_name = "nz_lens"
        nbin_lens = block[section_name, "nbin"]
        z = block[section_name, "z"]
        lenses = [ccl.NumberCountsTracer(cosmo_ccl, has_rsd=False, dndz=(z, block[section_name, "bin_%d"%i]), 
                                            bias=(z, block["bin_bias", "b%d"%i]*np.ones(len(z)) )) for i in range(1, nbin_lens+1)]

    print('gc tracers computed')
    ell = config['ell']
    n_ell = config['n_ell']
    cl_ll_ccl_nl = np.zeros((nbin_source, nbin_source, n_ell))
    cl_gg_ccl_nl = np.zeros((nbin_lens, nbin_lens, n_ell))
    cl_xc_ccl_nl = np.zeros((nbin_lens, nbin_source, n_ell))

    if config['compute_gc']:
        for i in range(nbin_lens):
            for j in range(nbin_lens):
                cl_gg_ccl_nl[i,j] = ccl.angular_cl(cosmo_ccl, lenses[i], lenses[j], ell) 
                block['galaxy_cl', f'bin_{i+1}_{j+1}'] = cl_gg_ccl_nl[i,j]
    if config['compute_cross']:
        for i in range(nbin_lens):
            for j in range(nbin_source):
                cl_xc_ccl_nl[i,j] = ccl.angular_cl(cosmo_ccl, lenses[i], sources[j], ell) 
                block['galaxy_shear_cl', f'bin_{i+1}_{j+1}'] = cl_xc_ccl_nl[i,j]
    if config['compute_shear']:
        for i in range(nbin_source):
            for j in range(nbin_source):
                cl_ll_ccl_nl[i,j] = ccl.angular_cl(cosmo_ccl, sources[i], sources[j], ell) 
                block['shear_cl', f'bin_{i+1}_{j+1}'] = cl_ll_ccl_nl[i,j]
    print('Cls computed')
    block['shear_cl', 'ell'] = block['galaxy_shear_cl', 'ell'] = block['galaxy_cl', 'ell'] = ell
    block['shear_cl', 'save_name'] = 'shear_cl'
    block['galaxy_shear_cl', 'save_name'] = 'galaxy_shear_cl'
    block['galaxy_cl', 'save_name'] = 'galaxy_cl'
    block['shear_cl', 'is_auto'] = block['galaxy_shear_cl', 'is_auto'] = block['galaxy_cl', 'is_auto'] = False
    block['shear_cl', 'nbin'] = block['shear_cl', 'nbin_a'] = block['shear_cl', 'nbin_b'] = nbin_source
    block['galaxy_cl', 'nbin'] = block['galaxy_cl', 'nbin_a'] = block['galaxy_cl', 'nbin_b'] = nbin_lens
    block['galaxy_shear_cl', 'nbin_a'] = nbin_lens
    block['galaxy_shear_cl', 'nbin_b'] = nbin_source
    block['shear_cl', 'sep_name'] = block['galaxy_shear_cl', 'sep_name'] = block['galaxy_cl', 'sep_name'] = "ell"
    return 0


def execute(block, config):
    try:
        call_ccl(block, config)
    except ValueError as error:
        return 1
    return 0    

if __name__=="__main__":
    print("Executing example case")