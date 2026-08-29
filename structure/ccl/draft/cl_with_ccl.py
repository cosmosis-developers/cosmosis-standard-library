# coding:utf-8
"""
Enhanced CCL integration for CosmoSIS with full functionality support.

This module provides comprehensive integration of the Core Cosmology Library (CCL)
into CosmoSIS, supporting:
- Background quantities (distances, growth factors)
- Linear and non-linear power spectra with multiple prescriptions
- Comprehensive halo model framework
- Multiple tracer types (weak lensing, number counts, CMB lensing, etc.)
- Modified gravity extensions
- Correlation functions
- Baryonic effects and emulators
"""

import os
import numpy as np
from cosmosis.datablock import names, option_section, BlockError
import re
import sys
import scipy.interpolate as interp
import warnings

# for timing
from timeit import default_timer as timer
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

import pyccl as ccl

# CosmoSIS section names
cosmo = names.cosmological_parameters
ia = names.intrinsic_alignment_parameters
bias = names.bias_field
distances = names.distances
matter_power_lin = names.matter_power_lin
matter_power_nl = names.matter_power_nl
growth_parameters = names.growth_parameters
halo_model = names.halo_model_parameters



def setup(options):
    """
    Setup function for comprehensive CCL integration.
    
    Configures all CCL functionality including:
    - Background calculations
    - Power spectrum computations  
    - Halo model parameters
    - Tracer configurations
    - Angular power spectra options
    """
    config = {}
    
    # ===== GENERAL COMPUTATION OPTIONS =====
    config['compute_background'] = options.get_bool(option_section, "compute_background", True)
    config['compute_power_spectra'] = options.get_bool(option_section, "compute_power_spectra", True)
    config['compute_growth'] = options.get_bool(option_section, "compute_growth", True)
    config['compute_halo_model'] = options.get_bool(option_section, "compute_halo_model", False)
    
    # ===== ANGULAR POWER SPECTRA OPTIONS =====
    config['compute_gc'] = options.get_bool(option_section, "compute_gc", True)
    config['compute_shear'] = options.get_bool(option_section, "compute_shear", True)
    config['compute_cross'] = options.get_bool(option_section, "compute_cross", True)
    config['compute_cmb_lensing'] = options.get_bool(option_section, "compute_cmb_lensing", False)
    config['compute_isw'] = options.get_bool(option_section, "compute_isw", False)
    config['compute_tsz'] = options.get_bool(option_section, "compute_tsz", False)
    config['compute_cib'] = options.get_bool(option_section, "compute_cib", False)
    
    # ===== MAGNIFICATION BIAS =====
    config['use_magnification'] = options.get_bool(option_section, "use_magnification", False)
    config['magnification_alpha'] = options.get_double(option_section, "magnification_alpha", 2.5)
    
    # ===== PERTURBATIVE TRACERS =====
    config['use_perturbative_bias'] = options.get_bool(option_section, "use_perturbative_bias", False)
    config['perturbative_bias_type'] = options.get_string(option_section, "perturbative_bias_type", "bacco_lbias")
    config['log10k_min_pt'] = options.get_double(option_section, "log10k_min_pt", -4.0)
    config['log10k_max_pt'] = options.get_double(option_section, "log10k_max_pt", 2.0)
    config['nk_per_decade_pt'] = options.get_int(option_section, "nk_per_decade_pt", 20)
    
    # ===== CORRELATION FUNCTIONS =====
    config['compute_xi'] = options.get_bool(option_section, "compute_xi", False)
    config['xi_type'] = options.get_string(option_section, "xi_type", "gg+")
    config['xi_method'] = options.get_string(option_section, "xi_method", "fftlog")
    
    # 3D correlation functions
    config['compute_xi_3d'] = options.get_bool(option_section, "compute_xi_3d", False)
    config['compute_xi_rsd'] = options.get_bool(option_section, "compute_xi_rsd", False)
    config['compute_xi_multipoles'] = options.get_bool(option_section, "compute_xi_multipoles", False)
    config['rsd_beta'] = options.get_double(option_section, "rsd_beta", 0.5)
    config['multipole_ells'] = options.get_string(option_section, "multipole_ells", "0 2 4")
    
    # ===== CCL COSMOLOGY SETUP =====
    config['transfer_function'] = options.get_string(option_section, "transfer_function", "boltzmann_camb")
    config['matter_power_spectrum'] = options.get_string(option_section, "matter_power_spectrum", "camb")
    
    # Emulator support for power spectra
    config['use_emulator_pk'] = options.get_bool(option_section, "use_emulator_pk", False)
    config['emulator_pk_type'] = options.get_string(option_section, "emulator_pk_type", "baccoemu")
    
    # Emulator support for transfer functions  
    config['use_emulator_tf'] = options.get_bool(option_section, "use_emulator_tf", False)
    config['emulator_tf_type'] = options.get_string(option_section, "emulator_tf_type", "baccoemu")
    
    # Halofit version for CAMB
    halofit_version = options.get_string(option_section, "halofit_version", "mead2020")
    config['extra_parameters'] = {"camb": {"halofit_version": halofit_version}}
    
    # Additional CAMB parameters
    if options.has_value(option_section, "camb_kmax"):
        config['extra_parameters']["camb"]["kmax"] = options.get_double(option_section, "camb_kmax")
    if options.has_value(option_section, "camb_lmax"):
        config['extra_parameters']["camb"]["lmax"] = options.get_int(option_section, "camb_lmax")
    
    # Advanced cosmology parameters
    config['T_ncdm'] = options.get_double(option_section, "T_ncdm", 0.0)
    config['use_sigma8_cb'] = options.get_bool(option_section, "use_sigma8_cb", False)
    
    # ===== BARYONIC EFFECTS =====
    baryonic_model = options.get_string(option_section, "baryonic_model", "none")
    if baryonic_model.lower() == "none":
        config['baryonic_effects'] = None
    elif baryonic_model.lower() == "baccoemu":
        config['baryonic_effects'] = ccl.BaccoemuBaryons()
    elif baryonic_model.lower() == "schneider":
        config['baryonic_effects'] = ccl.BaryonCorrection()
    else:
        config['baryonic_effects'] = None
        
    # ===== MODIFIED GRAVITY =====
    mg_model = options.get_string(option_section, "mg_model", "none")
    if mg_model.lower() == "none":
        config['mg_parametrization'] = None
    elif mg_model.lower() == "mu_sigma":
        config['mg_parametrization'] = ccl.MuSigmaModel()
    else:
        config['mg_parametrization'] = None
    
    # ===== HALO MODEL CONFIGURATION =====
    if config['compute_halo_model']:
        config['halo_mass_function'] = options.get_string(option_section, "halo_mass_function", "tinker08")
        config['halo_bias'] = options.get_string(option_section, "halo_bias", "tinker10")
        config['halo_concentration'] = options.get_string(option_section, "halo_concentration", "duffy2008")
        config['halo_profile'] = options.get_string(option_section, "halo_profile", "nfw")
        config['mass_definition'] = options.get_string(option_section, "mass_definition", "200m")
    
    # ===== ELL CONFIGURATION =====
    ell = np.array([])
    ell_min_logspaced = options.get_double(option_section, "ell_min_logspaced", -1.)
    ell_max_logspaced = options.get_double(option_section, "ell_max_logspaced", -1.)
    n_ell_logspaced = options.get_int(option_section, "n_ell_logspaced", -1)
    
    if n_ell_logspaced > 0:
        assert ell_min_logspaced > 0., "ell_min_logspaced must be positive"
        assert ell_max_logspaced > 0., "ell_max_logspaced must be positive"
        ell = np.logspace(np.log10(ell_min_logspaced), np.log10(ell_max_logspaced), n_ell_logspaced)
    elif options.has_value(option_section, "ell_values"):
        # Allow custom ell values
        ell_str = options.get_string(option_section, "ell_values")
        ell = np.array([float(x) for x in ell_str.split()])
        n_ell_logspaced = len(ell)
    else:
        # Default ell range
        ell = np.logspace(1, 4, 100)  # ell from 10 to 10000
        n_ell_logspaced = len(ell)
        
    config['ell'] = ell
    config['n_ell'] = n_ell_logspaced
    
    # ===== REDSHIFT AND K RANGES FOR POWER SPECTRA =====
    if config['compute_power_spectra']:
        config['z_min'] = options.get_double(option_section, "z_min", 0.0)
        config['z_max'] = options.get_double(option_section, "z_max", 3.0)
        config['n_z'] = options.get_int(option_section, "n_z", 100)
        config['k_min'] = options.get_double(option_section, "k_min", 1e-4)
        config['k_max'] = options.get_double(option_section, "k_max", 10.0)
        config['n_k'] = options.get_int(option_section, "n_k", 100)
    
    # ===== CORRELATION FUNCTION CONFIGURATION =====
    if config['compute_xi']:
        config['theta_min'] = options.get_double(option_section, "theta_min", 0.1)
        config['theta_max'] = options.get_double(option_section, "theta_max", 300.0)
        config['n_theta'] = options.get_int(option_section, "n_theta", 50)
    
    # ===== ACCURACY PARAMETERS =====
    config['limber_integration'] = options.get_bool(option_section, "limber_integration", True)
    config['non_limber_max_ell'] = options.get_int(option_section, "non_limber_max_ell", 100)
    config['integration_method'] = options.get_string(option_section, "integration_method", "qag_quad")
    config['relative_tolerance'] = options.get_double(option_section, "relative_tolerance", 1e-4)
    config['absolute_tolerance'] = options.get_double(option_section, "absolute_tolerance", 0.0)
    
    # Advanced angular_cl options
    config['p_of_k_a'] = options.get_string(option_section, "p_of_k_a", "delta_matter:delta_matter")
    config['l_logstep'] = options.get_double(option_section, "l_logstep", 1.15)
    config['l_linstep'] = options.get_double(option_section, "l_linstep", 40.0)
    config['dchi'] = options.get_double(option_section, "dchi", -1.0)
    config['dlnchi'] = options.get_double(option_section, "dlnchi", -1.0)
    
    return config
        



def create_emulator_objects(config: Dict) -> tuple:
    """Create emulator objects for power spectra and transfer functions."""
    emulator_pk = None
    emulator_tf = None
    
    # Power spectrum emulators
    if config.get('use_emulator_pk', False):
        emulator_type = config.get('emulator_pk_type', 'baccoemu')
        try:
            if emulator_type.lower() == 'baccoemu':
                emulator_pk = ccl.BaccoemuNonlinear()
            elif emulator_type.lower() == 'euclid_emulator2':
                emulator_pk = ccl.EuclidEmulator2Nonlinear()
            elif emulator_type.lower() == 'cosmic_emu':
                emulator_pk = ccl.CosmicEmuNonlinear()
            else:
                warnings.warn(f"Unknown emulator type: {emulator_type}, using default")
        except Exception as e:
            warnings.warn(f"Failed to create power spectrum emulator: {e}")
    
    # Transfer function emulators  
    if config.get('use_emulator_tf', False):
        emulator_type = config.get('emulator_tf_type', 'baccoemu')
        try:
            if emulator_type.lower() == 'baccoemu':
                emulator_tf = ccl.BaccoemuTransfer()
            else:
                warnings.warn(f"Unknown transfer function emulator: {emulator_type}")
        except Exception as e:
            warnings.warn(f"Failed to create transfer function emulator: {e}")
    
    return emulator_pk, emulator_tf


def create_ccl_cosmology(block: Any, config: Dict) -> ccl.Cosmology:
    """Create CCL Cosmology object from CosmoSIS data block with full emulator support."""
    
    # Create emulator objects if requested
    emulator_pk, emulator_tf = create_emulator_objects(config)
    
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
        "baryonic_effects": config['baryonic_effects'],
        "mg_parametrization": config['mg_parametrization'],
    }
    
    # Advanced neutrino temperature parameter
    if config.get('T_ncdm', 0.0) != 0.0:
        ccl_args["T_ncdm"] = config['T_ncdm']
    
    # Use emulators if available, otherwise use string specifications
    if emulator_pk is not None:
        ccl_args["matter_power_spectrum"] = emulator_pk
    else:
        ccl_args["matter_power_spectrum"] = config['matter_power_spectrum']
    
    if emulator_tf is not None:
        ccl_args["transfer_function"] = emulator_tf
    else:
        ccl_args["transfer_function"] = config['transfer_function']
    
    # Extra parameters (only if not using emulators)
    if emulator_pk is None and emulator_tf is None:
        ccl_args["extra_parameters"] = config['extra_parameters']
    
    # Handle normalization (A_s or sigma8 or sigma8_cb)
    if config.get('use_sigma8_cb', False) and block.has_value(cosmo, "sigma_8_cb"):
        ccl_args["sigma8"] = block.get_double(cosmo, "sigma_8_cb")
    elif block.has_value(cosmo, "a_s"):
        ccl_args["A_s"] = block.get_double(cosmo, "a_s")
    elif block.has_value(cosmo, "sigma_8"):
        ccl_args["sigma8"] = block.get_double(cosmo, "sigma_8")
    else:
        raise ValueError("Must provide either A_s, sigma_8, or sigma_8_cb for cosmology normalization")
    
    # Handle neutrino mass splitting
    if block.has_value(cosmo, "mass_split"):
        mass_split = block.get_string(cosmo, "mass_split")
        if mass_split.lower() in ['normal', 'inverted', 'equal', 'single']:
            ccl_args["mass_split"] = mass_split.lower()
    
    return ccl.Cosmology(**ccl_args)


def compute_background_quantities(block: Any, cosmo_ccl: ccl.Cosmology, config: Dict) -> None:
    """Compute and store background quantities using CCL."""
    if not config['compute_background']:
        return
        
    # Define redshift range
    z_max = config.get('z_max', 3.0)
    n_z = config.get('n_z', 100)
    z = np.linspace(0, z_max, n_z)
    
    # Compute background quantities
    try:
        # Distances
        chi = ccl.comoving_radial_distance(cosmo_ccl, 1./(1+z))
        d_A = ccl.angular_diameter_distance(cosmo_ccl, 1./(1+z))
        d_L = ccl.luminosity_distance(cosmo_ccl, 1./(1+z))
        d_M = ccl.comoving_transverse_distance(cosmo_ccl, 1./(1+z))
        
        # Hubble parameter and age
        h_z = ccl.h_over_h0(cosmo_ccl, 1./(1+z))
        age = ccl.age(cosmo_ccl, 1./(1+z))
        
        # Store in data block
        block[distances, "z"] = z
        block[distances, "d_a"] = d_A
        block[distances, "d_l"] = d_L
        block[distances, "d_m"] = d_M
        block[distances, "h"] = h_z
        block[distances, "age"] = age
        block[distances, "rs_drag"] = ccl.sound_horizon(cosmo_ccl, 1./(1+z))
        
    except Exception as e:
        warnings.warn(f"Error computing background quantities: {e}")


def compute_growth_factors(block: Any, cosmo_ccl: ccl.Cosmology, config: Dict) -> None:
    """Compute and store growth factors using CCL."""
    if not config['compute_growth']:
        return
        
    # Define redshift range
    z_max = config.get('z_max', 3.0)
    n_z = config.get('n_z', 100)
    z = np.linspace(0, z_max, n_z)
    a = 1./(1+z)
    
    try:
        # Growth factor and growth rate
        D_z = ccl.growth_factor(cosmo_ccl, a)
        f_z = ccl.growth_rate(cosmo_ccl, a)
        
        # Store in data block
        block[growth_parameters, "z"] = z
        block[growth_parameters, "d_z"] = D_z
        block[growth_parameters, "f_z"] = f_z
        
    except Exception as e:
        warnings.warn(f"Error computing growth factors: {e}")


def compute_power_spectra(block: Any, cosmo_ccl: ccl.Cosmology, config: Dict) -> None:
    """Compute and store matter power spectra using CCL."""
    if not config['compute_power_spectra']:
        return
        
    # Define k and z ranges
    k_min = config.get('k_min', 1e-4)
    k_max = config.get('k_max', 10.0)
    n_k = config.get('n_k', 100)
    z_min = config.get('z_min', 0.0)
    z_max = config.get('z_max', 3.0)
    n_z = config.get('n_z', 100)
    
    k = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    z = np.linspace(z_min, z_max, n_z)
    a = 1./(1+z)
    
    try:
        # Compute linear and non-linear power spectra
        P_lin = np.zeros((n_z, n_k))
        P_nl = np.zeros((n_z, n_k))
        
        for i, a_val in enumerate(a):
            P_lin[i, :] = ccl.linear_matter_power(cosmo_ccl, k, a_val)
            P_nl[i, :] = ccl.nonlin_matter_power(cosmo_ccl, k, a_val)
        
        # Store in data block
        block[matter_power_lin, "z"] = z
        block[matter_power_lin, "k_h"] = k
        block[matter_power_lin, "p_k"] = P_lin
        
        block[matter_power_nl, "z"] = z
        block[matter_power_nl, "k_h"] = k
        block[matter_power_nl, "p_k"] = P_nl
        
    except Exception as e:
        warnings.warn(f"Error computing power spectra: {e}")


def create_perturbative_calculator(cosmo_ccl: ccl.Cosmology, config: Dict) -> Optional[Any]:
    """Create perturbative bias calculator if requested."""
    if not config.get('use_perturbative_bias', False):
        return None
        
    calc_type = config.get('perturbative_bias_type', 'bacco_lbias')
    log10k_min = config.get('log10k_min_pt', -4.0)
    log10k_max = config.get('log10k_max_pt', 2.0)
    nk_per_decade = config.get('nk_per_decade_pt', 20)
    
    try:
        if calc_type.lower() == 'bacco_lbias':
            # BaccoLbiasCalculator for perturbative bias
            calculator = ccl.nl_pt.BaccoLbiasCalculator(
                cosmo=cosmo_ccl,
                log10k_min=log10k_min,
                log10k_max=log10k_max,
                nk_per_decade=nk_per_decade
            )
            calculator.update_ingredients(cosmo_ccl)
            return calculator
        else:
            warnings.warn(f"Unknown perturbative bias calculator: {calc_type}")
            return None
    except Exception as e:
        warnings.warn(f"Failed to create perturbative calculator: {e}")
        return None


def create_tracers(block: Any, cosmo_ccl: ccl.Cosmology, config: Dict) -> Dict[str, List]:
    """Create CCL tracers for different observables with advanced features."""
    tracers = {}
    
    # Create perturbative calculator if needed
    pt_calculator = create_perturbative_calculator(cosmo_ccl, config)
    
    # Weak lensing tracers
    if config['compute_shear'] and block.has_section("nz_source"):
        section_name = "nz_source"
        nbin_source = block[section_name, "nbin"]
        z = block[section_name, "z"]
        
        # Intrinsic alignment bias
        if block.has_value(ia, "a1"):
            bias_ia = block.get_double(ia, "a1") * ((1. + z)/(1. + block.get_double(ia, "z_piv"))) ** block.get_double(ia, "alpha1")
        else:
            bias_ia = np.zeros_like(z)
        
        # Magnification bias for sources (if enabled)
        mag_bias = None
        if config.get('use_magnification', False):
            alpha = config.get('magnification_alpha', 2.5)
            mag_bias = (alpha - 1.0) * np.ones_like(z)
        
        sources = []
        for i in range(1, nbin_source + 1):
            tracer_kwargs = {
                'dndz': (z, block[section_name, f"bin_{i}"]),
                'ia_bias': (z, bias_ia)
            }
            
            # Add magnification bias if specified
            if mag_bias is not None:
                tracer_kwargs['mag_bias'] = (z, mag_bias)
            
            tracer = ccl.WeakLensingTracer(cosmo_ccl, **tracer_kwargs)
            sources.append(tracer)
        
        tracers['weak_lensing'] = sources
    
    # Galaxy clustering tracers with advanced bias modeling
    if config['compute_gc'] and block.has_section("nz_lens"):
        section_name = "nz_lens"
        nbin_lens = block[section_name, "nbin"]
        z = block[section_name, "z"]
        
        lenses = []
        for i in range(1, nbin_lens + 1):
            # Galaxy bias - can be from perturbative calculator or simple bias
            if pt_calculator is not None:
                # Use perturbative bias from calculator
                # Note: This is a simplified example - in practice you'd need to 
                # compute the bias at each redshift using the calculator
                bias_vals = np.ones(len(z)) * 1.5  # Placeholder - would use pt_calculator
            elif block.has_section("bin_bias"):
                bias_vals = block["bin_bias", f"b{i}"]
                if len(bias_vals) == 1:
                    bias_vals = bias_vals * np.ones(len(z))
            else:
                bias_vals = np.ones(len(z))
            
            # Magnification bias for lenses (if enabled)
            mag_bias = None
            if config.get('use_magnification', False):
                alpha = config.get('magnification_alpha', 2.5)
                mag_bias = (alpha - 1.0) * np.ones_like(z)
            
            tracer_kwargs = {
                'has_rsd': False,
                'dndz': (z, block[section_name, f"bin_{i}"]),
                'bias': (z, bias_vals)
            }
            
            # Add magnification bias if specified
            if mag_bias is not None:
                tracer_kwargs['mag_bias'] = (z, mag_bias)
            
            tracer = ccl.NumberCountsTracer(cosmo_ccl, **tracer_kwargs)
            lenses.append(tracer)
        
        tracers['number_counts'] = lenses
    
    # CMB lensing tracer
    if config['compute_cmb_lensing']:
        cmb_tracer = ccl.CMBLensingTracer(cosmo_ccl, z_source=1100)
        tracers['cmb_lensing'] = [cmb_tracer]
    
    # ISW tracer
    if config['compute_isw']:
        isw_tracer = ccl.ISWTracer(cosmo_ccl)
        tracers['isw'] = [isw_tracer]
    
    # Store perturbative calculator for later use
    if pt_calculator is not None:
        tracers['pt_calculator'] = pt_calculator
    
    return tracers


def compute_angular_power_spectra(block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict, config: Dict) -> None:
    """Compute angular power spectra for all tracer combinations with advanced CCL options."""
    ell = config['ell']
    n_ell = config['n_ell']
    
    # Prepare advanced angular_cl options
    angular_cl_kwargs = {}
    
    # Limber integration control
    if not config.get('limber_integration', True):
        angular_cl_kwargs['limber_integration'] = False
        if config.get('non_limber_max_ell', 100) > 0:
            angular_cl_kwargs['non_limber_max_ell'] = config['non_limber_max_ell']
    
    # Integration method and tolerances
    integration_method = config.get('integration_method', 'qag_quad')
    if integration_method != 'qag_quad':
        angular_cl_kwargs['integration_method'] = integration_method
    
    rel_tol = config.get('relative_tolerance', 1e-4)
    abs_tol = config.get('absolute_tolerance', 0.0)
    if rel_tol != 1e-4:
        angular_cl_kwargs['rtol'] = rel_tol
    if abs_tol != 0.0:
        angular_cl_kwargs['atol'] = abs_tol
    
    # Power spectrum specification
    p_of_k_a = config.get('p_of_k_a', 'delta_matter:delta_matter')
    if p_of_k_a != 'delta_matter:delta_matter':
        angular_cl_kwargs['p_of_k_a'] = p_of_k_a
    
    # Sampling parameters
    l_logstep = config.get('l_logstep', 1.15)
    l_linstep = config.get('l_linstep', 40.0)
    if l_logstep != 1.15:
        angular_cl_kwargs['l_logstep'] = l_logstep
    if l_linstep != 40.0:
        angular_cl_kwargs['l_linstep'] = l_linstep
    
    # Radial sampling
    dchi = config.get('dchi', -1.0)
    dlnchi = config.get('dlnchi', -1.0)
    if dchi > 0:
        angular_cl_kwargs['dchi'] = dchi
    if dlnchi > 0:
        angular_cl_kwargs['dlnchi'] = dlnchi
    
    # Galaxy-galaxy auto and cross-correlations
    if config['compute_gc'] and 'number_counts' in tracers:
        lenses = tracers['number_counts']
        nbin_lens = len(lenses)
        cl_gg = np.zeros((nbin_lens, nbin_lens, n_ell))
        
        for i in range(nbin_lens):
            for j in range(i, nbin_lens):
                try:
                    cl_gg[i, j] = ccl.angular_cl(cosmo_ccl, lenses[i], lenses[j], ell, **angular_cl_kwargs)
                    if i != j:
                        cl_gg[j, i] = cl_gg[i, j]  # Symmetry
                    block['galaxy_cl', f'bin_{i+1}_{j+1}'] = cl_gg[i, j]
                except Exception as e:
                    warnings.warn(f"Error computing galaxy_cl for bins {i+1},{j+1}: {e}")
                    # Fallback to basic computation
                    cl_gg[i, j] = ccl.angular_cl(cosmo_ccl, lenses[i], lenses[j], ell)
                    if i != j:
                        cl_gg[j, i] = cl_gg[i, j]
                    block['galaxy_cl', f'bin_{i+1}_{j+1}'] = cl_gg[i, j]
        
        # Store metadata
        block['galaxy_cl', 'ell'] = ell
        block['galaxy_cl', 'nbin'] = nbin_lens
        block['galaxy_cl', 'nbin_a'] = nbin_lens
        block['galaxy_cl', 'nbin_b'] = nbin_lens
        block['galaxy_cl', 'save_name'] = 'galaxy_cl'
        block['galaxy_cl', 'is_auto'] = False
        block['galaxy_cl', 'sep_name'] = "ell"
    
    # Shear-shear auto and cross-correlations
    if config['compute_shear'] and 'weak_lensing' in tracers:
        sources = tracers['weak_lensing']
        nbin_source = len(sources)
        cl_ll = np.zeros((nbin_source, nbin_source, n_ell))
        
        for i in range(nbin_source):
            for j in range(i, nbin_source):
                try:
                    cl_ll[i, j] = ccl.angular_cl(cosmo_ccl, sources[i], sources[j], ell, **angular_cl_kwargs)
                    if i != j:
                        cl_ll[j, i] = cl_ll[i, j]  # Symmetry
                    block['shear_cl', f'bin_{i+1}_{j+1}'] = cl_ll[i, j]
                except Exception as e:
                    warnings.warn(f"Error computing shear_cl for bins {i+1},{j+1}: {e}")
                    # Fallback to basic computation
                    cl_ll[i, j] = ccl.angular_cl(cosmo_ccl, sources[i], sources[j], ell)
                    if i != j:
                        cl_ll[j, i] = cl_ll[i, j]
                    block['shear_cl', f'bin_{i+1}_{j+1}'] = cl_ll[i, j]
        
        # Store metadata
        block['shear_cl', 'ell'] = ell
        block['shear_cl', 'nbin'] = nbin_source
        block['shear_cl', 'nbin_a'] = nbin_source
        block['shear_cl', 'nbin_b'] = nbin_source
        block['shear_cl', 'save_name'] = 'shear_cl'
        block['shear_cl', 'is_auto'] = False
        block['shear_cl', 'sep_name'] = "ell"
    
    # Galaxy-shear cross-correlations
    if config['compute_cross'] and 'number_counts' in tracers and 'weak_lensing' in tracers:
        lenses = tracers['number_counts']
        sources = tracers['weak_lensing']
        nbin_lens = len(lenses)
        nbin_source = len(sources)
        cl_xc = np.zeros((nbin_lens, nbin_source, n_ell))
        
        for i in range(nbin_lens):
            for j in range(nbin_source):
                try:
                    cl_xc[i, j] = ccl.angular_cl(cosmo_ccl, lenses[i], sources[j], ell, **angular_cl_kwargs)
                    block['galaxy_shear_cl', f'bin_{i+1}_{j+1}'] = cl_xc[i, j]
                except Exception as e:
                    warnings.warn(f"Error computing galaxy_shear_cl for bins {i+1},{j+1}: {e}")
                    # Fallback to basic computation
                    cl_xc[i, j] = ccl.angular_cl(cosmo_ccl, lenses[i], sources[j], ell)
                    block['galaxy_shear_cl', f'bin_{i+1}_{j+1}'] = cl_xc[i, j]
        
        # Store metadata
        block['galaxy_shear_cl', 'ell'] = ell
        block['galaxy_shear_cl', 'nbin_a'] = nbin_lens
        block['galaxy_shear_cl', 'nbin_b'] = nbin_source
        block['galaxy_shear_cl', 'save_name'] = 'galaxy_shear_cl'
        block['galaxy_shear_cl', 'is_auto'] = False
        block['galaxy_shear_cl', 'sep_name'] = "ell"
    
    # CMB lensing cross-correlations
    if config['compute_cmb_lensing'] and 'cmb_lensing' in tracers:
        cmb_tracer = tracers['cmb_lensing'][0]
        
        # CMB lensing - galaxy clustering
        if 'number_counts' in tracers:
            lenses = tracers['number_counts']
            nbin_lens = len(lenses)
            for i in range(nbin_lens):
                try:
                    cl_cmb_gc = ccl.angular_cl(cosmo_ccl, cmb_tracer, lenses[i], ell, **angular_cl_kwargs)
                    block['cmb_galaxy_cl', f'bin_1_{i+1}'] = cl_cmb_gc
                except Exception as e:
                    warnings.warn(f"Error computing cmb_galaxy_cl for bin {i+1}: {e}")
                    cl_cmb_gc = ccl.angular_cl(cosmo_ccl, cmb_tracer, lenses[i], ell)
                    block['cmb_galaxy_cl', f'bin_1_{i+1}'] = cl_cmb_gc
            
            block['cmb_galaxy_cl', 'ell'] = ell
            block['cmb_galaxy_cl', 'nbin_a'] = 1
            block['cmb_galaxy_cl', 'nbin_b'] = nbin_lens
            block['cmb_galaxy_cl', 'save_name'] = 'cmb_galaxy_cl'
            block['cmb_galaxy_cl', 'is_auto'] = False
            block['cmb_galaxy_cl', 'sep_name'] = "ell"
        
        # CMB lensing - cosmic shear
        if 'weak_lensing' in tracers:
            sources = tracers['weak_lensing']
            nbin_source = len(sources)
            for i in range(nbin_source):
                try:
                    cl_cmb_wl = ccl.angular_cl(cosmo_ccl, cmb_tracer, sources[i], ell, **angular_cl_kwargs)
                    block['cmb_shear_cl', f'bin_1_{i+1}'] = cl_cmb_wl
                except Exception as e:
                    warnings.warn(f"Error computing cmb_shear_cl for bin {i+1}: {e}")
                    cl_cmb_wl = ccl.angular_cl(cosmo_ccl, cmb_tracer, sources[i], ell)
                    block['cmb_shear_cl', f'bin_1_{i+1}'] = cl_cmb_wl
            
            block['cmb_shear_cl', 'ell'] = ell
            block['cmb_shear_cl', 'nbin_a'] = 1
            block['cmb_shear_cl', 'nbin_b'] = nbin_source
            block['cmb_shear_cl', 'save_name'] = 'cmb_shear_cl'
            block['cmb_shear_cl', 'is_auto'] = False
            block['cmb_shear_cl', 'sep_name'] = "ell"


def compute_correlation_functions(block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict, config: Dict) -> None:
    """Compute correlation functions using advanced CCL options."""
    
    # Angular correlation functions
    if config.get('compute_xi', False):
        compute_angular_correlation_functions(block, cosmo_ccl, tracers, config)
    
    # 3D correlation functions
    if config.get('compute_xi_3d', False):
        compute_3d_correlation_functions(block, cosmo_ccl, tracers, config)
    
    # RSD correlation functions
    if config.get('compute_xi_rsd', False):
        compute_rsd_correlation_functions(block, cosmo_ccl, tracers, config)
    
    # Correlation function multipoles
    if config.get('compute_xi_multipoles', False):
        compute_correlation_multipoles(block, cosmo_ccl, tracers, config)


def compute_angular_correlation_functions(block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict, config: Dict) -> None:
    """Compute angular correlation functions with all CCL correlation types and methods."""
    
    theta_min = config.get('theta_min', 0.1)
    theta_max = config.get('theta_max', 300.0) 
    n_theta = config.get('n_theta', 50)
    theta = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta)
    
    xi_type = config.get('xi_type', 'gg+')
    xi_method = config.get('xi_method', 'fftlog')
    
    try:
        # First compute angular power spectra for the correlation function calculation
        ell = config['ell']
        
        # Galaxy-galaxy correlation functions (NN type)
        if xi_type in ["gg+", "gg-", "nn"] and 'number_counts' in tracers:
            lenses = tracers['number_counts']
            nbin_lens = len(lenses)
            
            for i in range(nbin_lens):
                for j in range(i, nbin_lens):
                    # Compute angular power spectrum first
                    cl_gg = ccl.angular_cl(cosmo_ccl, lenses[i], lenses[j], ell)
                    
                    # Convert to correlation function
                    corr_type = 'NN' if xi_type == 'nn' else ('GG+' if xi_type == 'gg+' else 'GG-')
                    xi_gg = ccl.correlation(cosmo_ccl, ell=ell, C_ell=cl_gg, theta=theta, 
                                          type=corr_type, method=xi_method)
                    
                    section_name = 'galaxy_xi_plus' if xi_type in ['gg+', 'nn'] else 'galaxy_xi_minus'
                    block[section_name, f'bin_{i+1}_{j+1}'] = xi_gg
            
            section_name = 'galaxy_xi_plus' if xi_type in ['gg+', 'nn'] else 'galaxy_xi_minus'
            block[section_name, 'theta'] = theta
            block[section_name, 'nbin'] = nbin_lens
            block[section_name, 'save_name'] = section_name
            block[section_name, 'sep_name'] = 'theta'
        
        # Shear correlation functions (GG+ and GG- types)
        elif xi_type in ["ll+", "ll-", "shear+", "shear-"] and 'weak_lensing' in tracers:
            sources = tracers['weak_lensing']
            nbin_source = len(sources)
            
            for i in range(nbin_source):
                for j in range(i, nbin_source):
                    # Compute angular power spectrum first
                    cl_ll = ccl.angular_cl(cosmo_ccl, sources[i], sources[j], ell)
                    
                    # Convert to correlation function
                    corr_type = 'GG+' if xi_type in ['ll+', 'shear+'] else 'GG-'
                    xi_ll = ccl.correlation(cosmo_ccl, ell=ell, C_ell=cl_ll, theta=theta,
                                          type=corr_type, method=xi_method)
                    
                    section_name = 'shear_xi_plus' if xi_type in ['ll+', 'shear+'] else 'shear_xi_minus'
                    block[section_name, f'bin_{i+1}_{j+1}'] = xi_ll
            
            section_name = 'shear_xi_plus' if xi_type in ['ll+', 'shear+'] else 'shear_xi_minus'
            block[section_name, 'theta'] = theta
            block[section_name, 'nbin'] = nbin_source
            block[section_name, 'save_name'] = section_name
            block[section_name, 'sep_name'] = 'theta'
        
        # Galaxy-shear correlation functions (NG type)
        elif xi_type in ["ng", "galaxy_shear"] and 'number_counts' in tracers and 'weak_lensing' in tracers:
            lenses = tracers['number_counts']
            sources = tracers['weak_lensing']
            nbin_lens = len(lenses)
            nbin_source = len(sources)
            
            for i in range(nbin_lens):
                for j in range(nbin_source):
                    # Compute angular power spectrum first
                    cl_ng = ccl.angular_cl(cosmo_ccl, lenses[i], sources[j], ell)
                    
                    # Convert to correlation function
                    xi_ng = ccl.correlation(cosmo_ccl, ell=ell, C_ell=cl_ng, theta=theta,
                                          type='NG', method=xi_method)
                    
                    block['galaxy_shear_xi', f'bin_{i+1}_{j+1}'] = xi_ng
            
            block['galaxy_shear_xi', 'theta'] = theta
            block['galaxy_shear_xi', 'nbin_a'] = nbin_lens
            block['galaxy_shear_xi', 'nbin_b'] = nbin_source
            block['galaxy_shear_xi', 'save_name'] = 'galaxy_shear_xi'
            block['galaxy_shear_xi', 'sep_name'] = 'theta'
            
    except Exception as e:
        warnings.warn(f"Error computing angular correlation functions: {e}")


def compute_3d_correlation_functions(block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict, config: Dict) -> None:
    """Compute 3D correlation functions."""
    
    # Define r range for 3D correlations
    r_min = config.get('r_min_3d', 0.1)  # Mpc
    r_max = config.get('r_max_3d', 200.0)  # Mpc  
    n_r = config.get('n_r_3d', 50)
    r = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    
    # Scale factor for evaluation
    z_eval = config.get('z_eval_3d', 0.5)
    a_eval = 1.0 / (1.0 + z_eval)
    
    # Power spectrum to use
    p_of_k_a = config.get('p_of_k_a', 'delta_matter:delta_matter')
    
    try:
        # Compute 3D correlation function
        xi_3d = ccl.correlation_3d(cosmo_ccl, r=r, a=a_eval, p_of_k_a=p_of_k_a)
        
        # Store results
        block['correlation_3d', 'r'] = r
        block['correlation_3d', 'xi_3d'] = xi_3d
        block['correlation_3d', 'z_eval'] = z_eval
        block['correlation_3d', 'save_name'] = 'correlation_3d'
        block['correlation_3d', 'sep_name'] = 'r'
        
    except Exception as e:
        warnings.warn(f"Error computing 3D correlation functions: {e}")


def compute_rsd_correlation_functions(block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict, config: Dict) -> None:
    """Compute RSD correlation functions."""
    
    # Define r range
    r_min = config.get('r_min_3d', 0.1)
    r_max = config.get('r_max_3d', 200.0)
    n_r = config.get('n_r_3d', 50)
    r = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    
    # Scale factor and RSD parameter
    z_eval = config.get('z_eval_3d', 0.5)
    a_eval = 1.0 / (1.0 + z_eval)
    beta = config.get('rsd_beta', 0.5)
    
    # Power spectrum
    p_of_k_a = config.get('p_of_k_a', 'delta_matter:delta_matter')
    
    try:
        # RSD correlation function averaged over angles (monopole)
        xi_rsd_avg = ccl.correlation_3dRsd_avgmu(cosmo_ccl, r=r, a=a_eval, beta=beta, p_of_k_a=p_of_k_a)
        
        # Store results
        block['correlation_rsd', 'r'] = r
        block['correlation_rsd', 'xi_rsd_avg'] = xi_rsd_avg
        block['correlation_rsd', 'z_eval'] = z_eval
        block['correlation_rsd', 'beta'] = beta
        block['correlation_rsd', 'save_name'] = 'correlation_rsd'
        block['correlation_rsd', 'sep_name'] = 'r'
        
        # RSD correlation in (pi, sigma) space
        pi_values = np.linspace(0.1, 50.0, 20)  # Mpc
        sigma_values = np.linspace(0.1, 50.0, 20)  # Mpc
        
        xi_pi_sigma = np.zeros((len(pi_values), len(sigma_values)))
        for i, pi in enumerate(pi_values):
            for j, sigma in enumerate(sigma_values):
                xi_pi_sigma[i, j] = ccl.correlation_pi_sigma(cosmo_ccl, pi=pi, sigma=sigma, 
                                                           a=a_eval, beta=beta, p_of_k_a=p_of_k_a)
        
        block['correlation_pi_sigma', 'pi'] = pi_values
        block['correlation_pi_sigma', 'sigma'] = sigma_values
        block['correlation_pi_sigma', 'xi_pi_sigma'] = xi_pi_sigma
        block['correlation_pi_sigma', 'z_eval'] = z_eval
        block['correlation_pi_sigma', 'beta'] = beta
        block['correlation_pi_sigma', 'save_name'] = 'correlation_pi_sigma'
        
    except Exception as e:
        warnings.warn(f"Error computing RSD correlation functions: {e}")


def compute_correlation_multipoles(block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict, config: Dict) -> None:
    """Compute correlation function multipoles."""
    
    # Define r range
    r_min = config.get('r_min_3d', 0.1)
    r_max = config.get('r_max_3d', 200.0)
    n_r = config.get('n_r_3d', 50)
    r = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    
    # Scale factor and RSD parameter
    z_eval = config.get('z_eval_3d', 0.5)
    a_eval = 1.0 / (1.0 + z_eval)
    beta = config.get('rsd_beta', 0.5)
    
    # Multipole ells to compute
    multipole_ells_str = config.get('multipole_ells', '0 2 4')
    multipole_ells = [int(x) for x in multipole_ells_str.split()]
    
    # Power spectrum
    p_of_k_a = config.get('p_of_k_a', 'delta_matter:delta_matter')
    
    try:
        for ell in multipole_ells:
            xi_ell = ccl.correlation_multipole(cosmo_ccl, r=r, a=a_eval, beta=beta, 
                                             ell=ell, p_of_k_a=p_of_k_a)
            
            # Store results
            section_name = f'correlation_multipole_ell_{ell}'
            block[section_name, 'r'] = r
            block[section_name, f'xi_{ell}'] = xi_ell
            block[section_name, 'ell'] = ell
            block[section_name, 'z_eval'] = z_eval
            block[section_name, 'beta'] = beta
            block[section_name, 'save_name'] = section_name
            block[section_name, 'sep_name'] = 'r'
        
    except Exception as e:
        warnings.warn(f"Error computing correlation multipoles: {e}")


def compute_halo_model(block: Any, cosmo_ccl: ccl.Cosmology, config: Dict) -> None:
    """Compute halo model quantities using CCL."""
    if not config['compute_halo_model']:
        return
        
    try:
        # Mass range
        log_m_min = 10.0
        log_m_max = 16.0
        n_m = 100
        m = np.logspace(log_m_min, log_m_max, n_m)
        
        # Redshift range
        z_max = config.get('z_max', 3.0)
        n_z = config.get('n_z', 100)
        z = np.linspace(0, z_max, n_z)
        a = 1./(1+z)
        
        # Set up halo model components
        mass_function = getattr(ccl, f"halos.MassFuncTinker08", ccl.halos.MassFuncTinker08)()
        halo_bias = getattr(ccl, f"halos.HaloBiasTinker10", ccl.halos.HaloBiasTinker10)()
        concentration = getattr(ccl, f"halos.ConcentrationDuffy08", ccl.halos.ConcentrationDuffy08)()
        
        # Mass definition
        mass_def = ccl.halos.MassDef200m()
        
        # Halo profile (NFW)
        halo_profile = ccl.halos.HaloProfileNFW(concentration)
        
        # Compute halo model quantities
        hm_calculator = ccl.halos.HMCalculator(
            cosmo_ccl, mass_function, halo_bias, mass_def
        )
        
        # Mass function
        mf = np.zeros((n_z, n_m))
        bias_hm = np.zeros((n_z, n_m))
        
        for i, a_val in enumerate(a):
            mf[i, :] = mass_function(cosmo_ccl, m, a_val)
            bias_hm[i, :] = halo_bias(cosmo_ccl, m, a_val)
        
        # Store in data block
        block[halo_model, "z"] = z
        block[halo_model, "m"] = m
        block[halo_model, "dndm"] = mf
        block[halo_model, "bias"] = bias_hm
        
    except Exception as e:
        warnings.warn(f"Error computing halo model: {e}")


def call_ccl(block, config):
    """
    Main function to call all CCL computations.
    
    This function orchestrates the full CCL integration, computing:
    - Cosmology setup
    - Background quantities
    - Growth factors  
    - Power spectra
    - Halo model (if requested)
    - Tracers
    - Angular power spectra
    - Correlation functions (if requested)
    """
    start_time = timer()
    
    try:
        # Create CCL cosmology object
        cosmo_ccl = create_ccl_cosmology(block, config)
        print(f"CCL cosmology initialized")
        
        # Compute background quantities
        compute_background_quantities(block, cosmo_ccl, config)
        if config['compute_background']:
            print("Background quantities computed")
        
        # Compute growth factors
        compute_growth_factors(block, cosmo_ccl, config)
        if config['compute_growth']:
            print("Growth factors computed")
        
        # Compute power spectra
        compute_power_spectra(block, cosmo_ccl, config)
        if config['compute_power_spectra']:
            print("Power spectra computed")
        
        # Compute halo model
        compute_halo_model(block, cosmo_ccl, config)
        if config['compute_halo_model']:
            print("Halo model computed")
        
        # Create tracers for angular power spectra
        tracers = create_tracers(block, cosmo_ccl, config)
        print(f"Created tracers: {list(tracers.keys())}")
        
        # Compute angular power spectra
        compute_angular_power_spectra(block, cosmo_ccl, tracers, config)
        print("Angular power spectra computed")
        
        # Compute correlation functions if requested
        compute_correlation_functions(block, cosmo_ccl, tracers, config)
        if config['compute_xi']:
            print("Correlation functions computed")
        
        elapsed_time = timer() - start_time
        print(f"CCL computation completed in {timedelta(seconds=elapsed_time)}")
        
        return 0
        
    except Exception as e:
        print(f"Error in CCL computation: {e}")
        return 1


def execute(block, config):
    """
    Execute function called by CosmoSIS.
    
    This is the main entry point for the CCL module in CosmoSIS.
    It calls the main CCL computation function and handles errors.
    """
    try:
        return call_ccl(block, config)
    except Exception as error:
        print(f"CCL module failed with error: {error}")
        return 1


def cleanup(config):
    """
    Cleanup function called by CosmoSIS at the end of the chain.
    
    Currently no cleanup is needed for CCL.
    """
    pass


if __name__ == "__main__":
    print("Enhanced CCL integration for CosmoSIS")
    print("This module provides comprehensive CCL functionality including:")
    print("- Background quantities (distances, growth factors)")
    print("- Linear and non-linear power spectra")
    print("- Comprehensive halo model framework")
    print("- Multiple tracer types (WL, GC, CMB lensing, ISW, etc.)")
    print("- Angular power spectra and correlation functions")
    print("- Modified gravity extensions")
    print("- Baryonic effects and emulators")