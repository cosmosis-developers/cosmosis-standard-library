"""
CCL Correlation Functions Calculator

Handles computation of all types of correlation functions with advanced CCL options.
"""

import numpy as np
import warnings
from typing import Dict, List, Any

import pyccl as ccl


class CCLCorrelationFunctions:
    """Computes correlation functions with advanced CCL options."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
    
    def compute_angular_correlation_functions(self, block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict) -> None:
        """Compute angular correlation functions with all CCL correlation types and methods."""
        if not self.config.get('compute_xi', False):
            return
        
        theta_min = self.config.get('theta_min', 0.1)
        theta_max = self.config.get('theta_max', 300.0) 
        n_theta = self.config.get('n_theta', 50)
        theta = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta)
        
        xi_type = self.config.get('xi_type', 'gg+')
        xi_method = self.config.get('xi_method', 'fftlog')
        ell = self.config['ell']
        
        try:
            # Galaxy-galaxy correlation functions (NN type)
            if xi_type in ["gg+", "gg-", "nn"] and 'number_counts' in tracers:
                self._compute_galaxy_correlations(block, cosmo_ccl, tracers, ell, theta, xi_type, xi_method)
            
            # Shear correlation functions (GG+ and GG- types)
            elif xi_type in ["ll+", "ll-", "shear+", "shear-"] and 'weak_lensing' in tracers:
                self._compute_shear_correlations(block, cosmo_ccl, tracers, ell, theta, xi_type, xi_method)
            
            # Galaxy-shear correlation functions (NG type)
            elif xi_type in ["ng", "galaxy_shear"] and 'number_counts' in tracers and 'weak_lensing' in tracers:
                self._compute_galaxy_shear_correlations(block, cosmo_ccl, tracers, ell, theta, xi_method)
                
        except Exception as e:
            warnings.warn(f"Error computing angular correlation functions: {e}")
    
    def _compute_galaxy_correlations(self, block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict,
                                   ell: np.ndarray, theta: np.ndarray, xi_type: str, xi_method: str) -> None:
        """Compute galaxy-galaxy correlation functions."""
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
        self._store_xi_metadata(block, section_name, theta, nbin_lens, nbin_lens)
    
    def _compute_shear_correlations(self, block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict,
                                  ell: np.ndarray, theta: np.ndarray, xi_type: str, xi_method: str) -> None:
        """Compute shear correlation functions."""
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
        self._store_xi_metadata(block, section_name, theta, nbin_source, nbin_source)
    
    def _compute_galaxy_shear_correlations(self, block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict,
                                         ell: np.ndarray, theta: np.ndarray, xi_method: str) -> None:
        """Compute galaxy-shear correlation functions."""
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
        
        self._store_cross_xi_metadata(block, 'galaxy_shear_xi', theta, nbin_lens, nbin_source)
    
    def compute_3d_correlation_functions(self, block: Any, cosmo_ccl: ccl.Cosmology) -> None:
        """Compute 3D correlation functions."""
        if not self.config.get('compute_xi_3d', False):
            return
        
        # Define r range for 3D correlations
        r_min = self.config.get('r_min_3d', 0.1)
        r_max = self.config.get('r_max_3d', 200.0)
        n_r = self.config.get('n_r_3d', 50)
        r = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
        
        # Scale factor for evaluation
        z_eval = self.config.get('z_eval_3d', 0.5)
        a_eval = 1.0 / (1.0 + z_eval)
        
        # Power spectrum to use
        p_of_k_a = self.config.get('p_of_k_a', 'delta_matter:delta_matter')
        
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
    
    def compute_rsd_correlation_functions(self, block: Any, cosmo_ccl: ccl.Cosmology) -> None:
        """Compute RSD correlation functions."""
        if not self.config.get('compute_xi_rsd', False):
            return
        
        # Define r range
        r_min = self.config.get('r_min_3d', 0.1)
        r_max = self.config.get('r_max_3d', 200.0)
        n_r = self.config.get('n_r_3d', 50)
        r = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
        
        # Scale factor and RSD parameter
        z_eval = self.config.get('z_eval_3d', 0.5)
        a_eval = 1.0 / (1.0 + z_eval)
        beta = self.config.get('rsd_beta', 0.5)
        
        # Power spectrum
        p_of_k_a = self.config.get('p_of_k_a', 'delta_matter:delta_matter')
        
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
            self._compute_pi_sigma_correlations(block, cosmo_ccl, a_eval, beta, p_of_k_a, z_eval)
            
        except Exception as e:
            warnings.warn(f"Error computing RSD correlation functions: {e}")
    
    def _compute_pi_sigma_correlations(self, block: Any, cosmo_ccl: ccl.Cosmology, 
                                     a_eval: float, beta: float, p_of_k_a: str, z_eval: float) -> None:
        """Compute RSD correlation in (pi, sigma) space."""
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
    
    def compute_correlation_multipoles(self, block: Any, cosmo_ccl: ccl.Cosmology) -> None:
        """Compute correlation function multipoles."""
        if not self.config.get('compute_xi_multipoles', False):
            return
        
        # Define r range
        r_min = self.config.get('r_min_3d', 0.1)
        r_max = self.config.get('r_max_3d', 200.0)
        n_r = self.config.get('n_r_3d', 50)
        r = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
        
        # Scale factor and RSD parameter
        z_eval = self.config.get('z_eval_3d', 0.5)
        a_eval = 1.0 / (1.0 + z_eval)
        beta = self.config.get('rsd_beta', 0.5)
        
        # Multipole ells to compute
        multipole_ells_str = self.config.get('multipole_ells', '0 2 4')
        multipole_ells = [int(x) for x in multipole_ells_str.split()]
        
        # Power spectrum
        p_of_k_a = self.config.get('p_of_k_a', 'delta_matter:delta_matter')
        
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
    
    def _store_xi_metadata(self, block: Any, section_name: str, theta: np.ndarray, 
                          nbin_a: int, nbin_b: int) -> None:
        """Store metadata for auto-correlation functions."""
        block[section_name, 'theta'] = theta
        block[section_name, 'nbin'] = nbin_a
        block[section_name, 'save_name'] = section_name
        block[section_name, 'sep_name'] = 'theta'
    
    def _store_cross_xi_metadata(self, block: Any, section_name: str, theta: np.ndarray, 
                                nbin_a: int, nbin_b: int) -> None:
        """Store metadata for cross-correlation functions."""
        block[section_name, 'theta'] = theta
        block[section_name, 'nbin_a'] = nbin_a
        block[section_name, 'nbin_b'] = nbin_b
        block[section_name, 'save_name'] = section_name
        block[section_name, 'sep_name'] = 'theta'
    
    def compute_all_correlations(self, block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict) -> None:
        """Compute all correlation functions."""
        # Angular correlation functions
        self.compute_angular_correlation_functions(block, cosmo_ccl, tracers)
        
        # 3D correlation functions
        self.compute_3d_correlation_functions(block, cosmo_ccl)
        
        # RSD correlation functions
        self.compute_rsd_correlation_functions(block, cosmo_ccl)
        
        # Correlation function multipoles
        self.compute_correlation_multipoles(block, cosmo_ccl)
