"""
CCL Configuration Manager

Handles all configuration parsing and setup for the CCL integration.
"""

import numpy as np
from typing import Dict, Any
from cosmosis.datablock import option_section


class CCLConfig:
    """Configuration manager for CCL integration."""
    
    def __init__(self, options):
        """Initialize configuration from CosmoSIS options."""
        self.options = options
        self.config = {}
        self._parse_all_options()
    
    def _parse_all_options(self):
        """Parse all configuration options."""
        self._parse_general_options()
        self._parse_angular_cl_options()
        self._parse_correlation_options()
        self._parse_cosmology_options()
        self._parse_advanced_options()
        self._parse_sampling_options()
    
    def _parse_general_options(self):
        """Parse general computation control options."""
        self.config.update({
            'compute_background': self.options.get_bool(option_section, "compute_background", True),
            'compute_power_spectra': self.options.get_bool(option_section, "compute_power_spectra", True),
            'compute_growth': self.options.get_bool(option_section, "compute_growth", True),
            'compute_halo_model': self.options.get_bool(option_section, "compute_halo_model", False),
        })
    
    def _parse_angular_cl_options(self):
        """Parse angular power spectra options."""
        self.config.update({
            'compute_gc': self.options.get_bool(option_section, "compute_gc", True),
            'compute_shear': self.options.get_bool(option_section, "compute_shear", True),
            'compute_cross': self.options.get_bool(option_section, "compute_cross", True),
            'compute_cmb_lensing': self.options.get_bool(option_section, "compute_cmb_lensing", False),
            'compute_isw': self.options.get_bool(option_section, "compute_isw", False),
            'compute_tsz': self.options.get_bool(option_section, "compute_tsz", False),
            'compute_cib': self.options.get_bool(option_section, "compute_cib", False),
        })
    
    def _parse_correlation_options(self):
        """Parse correlation function options."""
        self.config.update({
            # Angular correlations
            'compute_xi': self.options.get_bool(option_section, "compute_xi", False),
            'xi_type': self.options.get_string(option_section, "xi_type", "gg+"),
            'xi_method': self.options.get_string(option_section, "xi_method", "fftlog"),
            'theta_min': self.options.get_double(option_section, "theta_min", 0.1),
            'theta_max': self.options.get_double(option_section, "theta_max", 300.0),
            'n_theta': self.options.get_int(option_section, "n_theta", 50),
            
            # 3D correlations
            'compute_xi_3d': self.options.get_bool(option_section, "compute_xi_3d", False),
            'compute_xi_rsd': self.options.get_bool(option_section, "compute_xi_rsd", False),
            'compute_xi_multipoles': self.options.get_bool(option_section, "compute_xi_multipoles", False),
            'r_min_3d': self.options.get_double(option_section, "r_min_3d", 0.1),
            'r_max_3d': self.options.get_double(option_section, "r_max_3d", 200.0),
            'n_r_3d': self.options.get_int(option_section, "n_r_3d", 50),
            'z_eval_3d': self.options.get_double(option_section, "z_eval_3d", 0.5),
            'rsd_beta': self.options.get_double(option_section, "rsd_beta", 0.5),
            'multipole_ells': self.options.get_string(option_section, "multipole_ells", "0 2 4"),
        })
    
    def _parse_cosmology_options(self):
        """Parse CCL cosmology setup options."""
        self.config.update({
            'transfer_function': self.options.get_string(option_section, "transfer_function", "boltzmann_camb"),
            'matter_power_spectrum': self.options.get_string(option_section, "matter_power_spectrum", "camb"),
            
            # Emulator support
            'use_emulator_pk': self.options.get_bool(option_section, "use_emulator_pk", False),
            'emulator_pk_type': self.options.get_string(option_section, "emulator_pk_type", "baccoemu"),
            'use_emulator_tf': self.options.get_bool(option_section, "use_emulator_tf", False),
            'emulator_tf_type': self.options.get_string(option_section, "emulator_tf_type", "baccoemu"),
            
            # Advanced cosmology
            'T_ncdm': self.options.get_double(option_section, "T_ncdm", 0.0),
            'use_sigma8_cb': self.options.get_bool(option_section, "use_sigma8_cb", False),
            
            # CAMB parameters
            'halofit_version': self.options.get_string(option_section, "halofit_version", "mead2020"),
        })
        
        # Extra CAMB parameters
        self.config['extra_parameters'] = {"camb": {"halofit_version": self.config['halofit_version']}}
        if self.options.has_value(option_section, "camb_kmax"):
            self.config['extra_parameters']["camb"]["kmax"] = self.options.get_double(option_section, "camb_kmax")
        if self.options.has_value(option_section, "camb_lmax"):
            self.config['extra_parameters']["camb"]["lmax"] = self.options.get_int(option_section, "camb_lmax")
    
    def _parse_advanced_options(self):
        """Parse advanced physics options."""
        # Baryonic effects
        baryonic_model = self.options.get_string(option_section, "baryonic_model", "none")
        if baryonic_model.lower() == "none":
            self.config['baryonic_effects'] = None
        else:
            self.config['baryonic_effects'] = baryonic_model
        
        # Modified gravity
        mg_model = self.options.get_string(option_section, "mg_model", "none")
        self.config['mg_parametrization'] = None if mg_model.lower() == "none" else mg_model
        
        # Magnification bias
        self.config.update({
            'use_magnification': self.options.get_bool(option_section, "use_magnification", False),
            'magnification_alpha': self.options.get_double(option_section, "magnification_alpha", 2.5),
        })
        
        # Perturbative tracers
        self.config.update({
            'use_perturbative_bias': self.options.get_bool(option_section, "use_perturbative_bias", False),
            'perturbative_bias_type': self.options.get_string(option_section, "perturbative_bias_type", "bacco_lbias"),
            'log10k_min_pt': self.options.get_double(option_section, "log10k_min_pt", -4.0),
            'log10k_max_pt': self.options.get_double(option_section, "log10k_max_pt", 2.0),
            'nk_per_decade_pt': self.options.get_int(option_section, "nk_per_decade_pt", 20),
        })
        
        # Halo model
        if self.config['compute_halo_model']:
            self.config.update({
                'halo_mass_function': self.options.get_string(option_section, "halo_mass_function", "tinker08"),
                'halo_bias': self.options.get_string(option_section, "halo_bias", "tinker10"),
                'halo_concentration': self.options.get_string(option_section, "halo_concentration", "duffy2008"),
                'halo_profile': self.options.get_string(option_section, "halo_profile", "nfw"),
                'mass_definition': self.options.get_string(option_section, "mass_definition", "200m"),
            })
    
    def _parse_sampling_options(self):
        """Parse sampling and accuracy options."""
        # Ell configuration
        ell_min_logspaced = self.options.get_double(option_section, "ell_min_logspaced", -1.)
        ell_max_logspaced = self.options.get_double(option_section, "ell_max_logspaced", -1.)
        n_ell_logspaced = self.options.get_int(option_section, "n_ell_logspaced", -1)
        
        if n_ell_logspaced > 0:
            assert ell_min_logspaced > 0., "ell_min_logspaced must be positive"
            assert ell_max_logspaced > 0., "ell_max_logspaced must be positive"
            ell = np.logspace(np.log10(ell_min_logspaced), np.log10(ell_max_logspaced), n_ell_logspaced)
        elif self.options.has_value(option_section, "ell_values"):
            ell_str = self.options.get_string(option_section, "ell_values")
            ell = np.array([float(x) for x in ell_str.split()])
            n_ell_logspaced = len(ell)
        else:
            ell = np.logspace(1, 4, 100)
            n_ell_logspaced = len(ell)
        
        self.config.update({
            'ell': ell,
            'n_ell': n_ell_logspaced,
        })

        if self.config["compute_background"]:
            self.config.update({
                'z_min': self.options.get_double(option_section, "z_min", 0.0),
                'z_max': self.options.get_double(option_section, "z_max", 3.0),
                'n_z': self.options.get_int(option_section, "n_z", 100),
            })
        
        # Power spectra sampling
        if self.config['compute_power_spectra']:
            self.config.update({
                'z_min': self.options.get_double(option_section, "z_min", 0.0),
                'z_max': self.options.get_double(option_section, "z_max", 3.0),
                'n_z': self.options.get_int(option_section, "n_z", 100),
                'k_min': self.options.get_double(option_section, "k_min", 1e-4),
                'k_max': self.options.get_double(option_section, "k_max", 10.0),
                'n_k': self.options.get_int(option_section, "n_k", 100),
            })
        
        # Accuracy parameters
        self.config.update({
            'limber_integration': self.options.get_bool(option_section, "limber_integration", True),
            'non_limber_max_ell': self.options.get_int(option_section, "non_limber_max_ell", 100),
            'integration_method': self.options.get_string(option_section, "integration_method", "qag_quad"),
            'relative_tolerance': self.options.get_double(option_section, "relative_tolerance", 1e-4),
            'absolute_tolerance': self.options.get_double(option_section, "absolute_tolerance", 0.0),
            'p_of_k_a': self.options.get_string(option_section, "p_of_k_a", "delta_matter:delta_matter"),
            'l_logstep': self.options.get_double(option_section, "l_logstep", 1.15),
            'l_linstep': self.options.get_double(option_section, "l_linstep", 40.0),
            'dchi': self.options.get_double(option_section, "dchi", -1.0),
            'dlnchi': self.options.get_double(option_section, "dlnchi", -1.0),
        })
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str):
        """Get configuration value with dict-like access."""
        return self.config[key]
    
    def __contains__(self, key: str):
        """Check if key exists in configuration."""
        return key in self.config
