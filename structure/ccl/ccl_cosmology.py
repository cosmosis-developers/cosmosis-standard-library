"""
CCL Cosmology Manager

Handles CCL cosmology creation, emulators, and basic cosmological calculations.
"""

import numpy as np
import warnings
from typing import Dict, Any, Tuple, Optional

import pyccl as ccl
from cosmosis.datablock import names

# CosmoSIS section names
cosmo = names.cosmological_parameters
distances = names.distances
matter_power_lin = names.matter_power_lin
matter_power_nl = names.matter_power_nl
growth_parameters = names.growth_parameters
halo_model = names.halo_model_parameters


class CCLCosmologyManager:
    """Manages CCL cosmology creation and basic calculations."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
        self.cosmo_ccl = None
    
    def create_emulator_objects(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Create emulator objects for power spectra and transfer functions."""
        emulator_pk = None
        emulator_tf = None
        
        # Power spectrum emulators
        if self.config.get('use_emulator_pk', False):
            emulator_type = self.config.get('emulator_pk_type', 'baccoemu')
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
        if self.config.get('use_emulator_tf', False):
            emulator_type = self.config.get('emulator_tf_type', 'baccoemu')
            try:
                if emulator_type.lower() == 'baccoemu':
                    emulator_tf = ccl.BaccoemuTransfer()
                else:
                    warnings.warn(f"Unknown transfer function emulator: {emulator_type}")
            except Exception as e:
                warnings.warn(f"Failed to create transfer function emulator: {e}")
        
        return emulator_pk, emulator_tf
    
    def create_cosmology(self, block: Any) -> ccl.Cosmology:
        """Create CCL Cosmology object from CosmoSIS data block."""
        
        # Create emulator objects if requested
        emulator_pk, emulator_tf = self.create_emulator_objects()
        
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
        }
        
        # Advanced neutrino temperature parameter
        if self.config.get('T_ncdm', 0.0) != 0.0:
            ccl_args["T_ncdm"] = self.config['T_ncdm']
        
        # Use emulators if available, otherwise use string specifications
        if emulator_pk is not None:
            ccl_args["matter_power_spectrum"] = emulator_pk
        else:
            ccl_args["matter_power_spectrum"] = self.config['matter_power_spectrum']
        
        if emulator_tf is not None:
            ccl_args["transfer_function"] = emulator_tf
        else:
            ccl_args["transfer_function"] = self.config['transfer_function']
        
        # Extra parameters (only if not using emulators)
        if emulator_pk is None and emulator_tf is None:
            ccl_args["extra_parameters"] = self.config['extra_parameters']
        
        # Baryonic effects
        baryonic_model = self.config.get('baryonic_effects')
        if baryonic_model and baryonic_model != 'none':
            if baryonic_model.lower() == 'baccoemu':
                ccl_args["baryonic_effects"] = ccl.BaccoemuBaryons()
            elif baryonic_model.lower() == 'schneider':
                ccl_args["baryonic_effects"] = ccl.BaryonCorrection()
        
        # Modified gravity
        mg_model = self.config.get('mg_parametrization')
        if mg_model and mg_model != 'none':
            if mg_model.lower() == 'mu_sigma':
                ccl_args["mg_parametrization"] = ccl.MuSigmaModel()
        
        # Handle normalization (A_s or sigma8 or sigma8_cb)
        if self.config.get('use_sigma8_cb', False) and block.has_value(cosmo, "sigma_8_cb"):
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
        
        self.cosmo_ccl = ccl.Cosmology(**ccl_args)
        return self.cosmo_ccl
    
    def compute_background_quantities(self, block: Any) -> None:
        """Compute and store background quantities using CCL."""
        if not self.config['compute_background'] or self.cosmo_ccl is None:
            return
            
        # Define redshift range
        z_max = self.config.get('z_max', 3.0)
        n_z = self.config.get('n_z', 100)
        z = np.linspace(0, z_max, n_z)
        
        try:
            # Distances
            d_A = ccl.angular_diameter_distance(self.cosmo_ccl, 1./(1+z))
            d_L = ccl.luminosity_distance(self.cosmo_ccl, 1./(1+z))
            d_M = ccl.comoving_angular_distance(self.cosmo_ccl, 1./(1+z))
            
            # Hubble parameter and age
            h_z = ccl.h_over_h0(self.cosmo_ccl, 1./(1+z))
            age = ccl.age(self.cosmo_ccl, 1./(1+z))
            
            # Store in data block
            block[distances, "z"] = z
            block[distances, "d_a"] = d_A
            block[distances, "d_l"] = d_L
            block[distances, "d_m"] = d_M
            block[distances, "h"] = h_z
            block[distances, "age"] = age
            block[distances, "rs_drag"] = ccl.sound_horizon(self.cosmo_ccl, 1./(1+z))
            
        except Exception as e:
            warnings.warn(f"Error computing background quantities: {e}")
    
    def compute_growth_factors(self, block: Any) -> None:
        """Compute and store growth factors using CCL."""
        if not self.config['compute_growth'] or self.cosmo_ccl is None:
            return
            
        # Define redshift range
        z_max = self.config.get('z_max', 3.0)
        n_z = self.config.get('n_z', 100)
        z = np.linspace(0, z_max, n_z)
        a = 1./(1+z)
        
        try:
            # Growth factor and growth rate
            D_z = ccl.growth_factor(self.cosmo_ccl, a)
            f_z = ccl.growth_rate(self.cosmo_ccl, a)
            
            # Store in data block
            block[growth_parameters, "z"] = z
            block[growth_parameters, "d_z"] = D_z
            block[growth_parameters, "f_z"] = f_z
            
        except Exception as e:
            warnings.warn(f"Error computing growth factors: {e}")
    
    def compute_power_spectra(self, block: Any) -> None:
        """Compute and store matter power spectra using CCL."""
        if not self.config['compute_power_spectra'] or self.cosmo_ccl is None:
            return
            
        # Define k and z ranges
        k_min = self.config.get('k_min', 1e-4)
        k_max = self.config.get('k_max', 10.0)
        n_k = self.config.get('n_k', 100)
        z_min = self.config.get('z_min', 0.0)
        z_max = self.config.get('z_max', 3.0)
        n_z = self.config.get('n_z', 100)
        
        k = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
        z = np.linspace(z_min, z_max, n_z)
        a = 1./(1+z)
        
        try:
            # Compute linear and non-linear power spectra
            P_lin = np.zeros((n_z, n_k))
            P_nl = np.zeros((n_z, n_k))
            
            for i, a_val in enumerate(a):
                P_lin[i, :] = ccl.linear_matter_power(self.cosmo_ccl, k, a_val)
                P_nl[i, :] = ccl.nonlin_matter_power(self.cosmo_ccl, k, a_val)
            
            # Store in data block
            block[matter_power_lin, "z"] = z
            block[matter_power_lin, "k_h"] = k
            block[matter_power_lin, "p_k"] = P_lin
            
            block[matter_power_nl, "z"] = z
            block[matter_power_nl, "k_h"] = k
            block[matter_power_nl, "p_k"] = P_nl
            
        except Exception as e:
            warnings.warn(f"Error computing power spectra: {e}")
    
    def compute_halo_model(self, block: Any) -> None:
        """Compute halo model quantities using CCL."""
        if not self.config['compute_halo_model'] or self.cosmo_ccl is None:
            return
            
        try:
            # Mass range
            log_m_min = 10.0
            log_m_max = 16.0
            n_m = 100
            m = np.logspace(log_m_min, log_m_max, n_m)
            
            # Redshift range
            z_max = self.config.get('z_max', 3.0)
            n_z = self.config.get('n_z', 100)
            z = np.linspace(0, z_max, n_z)
            a = 1./(1+z)
            
            # Set up halo model components
            mass_function = getattr(ccl, f"halos.MassFuncTinker08", ccl.halos.MassFuncTinker08)()
            halo_bias = getattr(ccl, f"halos.HaloBiasTinker10", ccl.halos.HaloBiasTinker10)()
            
            # Mass function
            mf = np.zeros((n_z, n_m))
            bias_hm = np.zeros((n_z, n_m))
            
            for i, a_val in enumerate(a):
                mf[i, :] = mass_function(self.cosmo_ccl, m, a_val)
                bias_hm[i, :] = halo_bias(self.cosmo_ccl, m, a_val)
            
            # Store in data block
            block[halo_model, "z"] = z
            block[halo_model, "m"] = m
            block[halo_model, "dndm"] = mf
            block[halo_model, "bias"] = bias_hm
            
        except Exception as e:
            warnings.warn(f"Error computing halo model: {e}")
