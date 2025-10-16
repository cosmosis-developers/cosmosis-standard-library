"""
CCL Angular Power Spectra Calculator

Handles computation of angular power spectra with all advanced CCL options.
"""

import numpy as np
import warnings
from typing import Dict, List, Any

import pyccl as ccl


class CCLAngularPowerSpectra:
    """Computes angular power spectra with advanced CCL options."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
    
    def _prepare_angular_cl_kwargs(self) -> Dict:
        """Prepare advanced angular_cl keyword arguments."""
        angular_cl_kwargs = {}
        
        # Limber integration control      
        limber_max_error = self.config.get('limber_max_error', 0.01)
        if limber_max_error != 0.01:
            angular_cl_kwargs['limber_max_error'] = limber_max_error
        
        # Integration methods
        limber_integration_method = self.config.get('limber_integration_method', 'qag_quad')
        if limber_integration_method != 'qag_quad':
            angular_cl_kwargs['limber_integration_method'] = limber_integration_method
        
        non_limber_integration_method = self.config.get('non_limber_integration_method', 'FKEM')
        if non_limber_integration_method != 'FKEM':
            angular_cl_kwargs['non_limber_integration_method'] = non_limber_integration_method
        
        # FKEM parameters
        fkem_chi_min = self.config.get('fkem_chi_min', None)
        if fkem_chi_min is not None:
            angular_cl_kwargs['fkem_chi_min'] = fkem_chi_min
        
        fkem_Nchi = self.config.get('fkem_Nchi', None)
        if fkem_Nchi is not None:
            angular_cl_kwargs['fkem_Nchi'] = fkem_Nchi
        
        # Power spectrum specification
        p_of_k_a = self.config.get('p_of_k_a', 'delta_matter:delta_matter')
        if p_of_k_a != 'delta_matter:delta_matter':
            angular_cl_kwargs['p_of_k_a'] = p_of_k_a
        
        p_of_k_a_lin = self.config.get('p_of_k_a_lin', 'delta_matter:delta_matter')
        if p_of_k_a_lin != 'delta_matter:delta_matter':
            angular_cl_kwargs['p_of_k_a_lin'] = p_of_k_a_lin
        
        # Return metadata flag
        return_meta = self.config.get('return_meta', False)
        if return_meta:
            angular_cl_kwargs['return_meta'] = return_meta
        
        return angular_cl_kwargs
    
    def _compute_cl_safely(self, cosmo_ccl: ccl.Cosmology, tracer1: Any, tracer2: Any, 
                          ell: np.ndarray, angular_cl_kwargs: Dict) -> np.ndarray:
        """Compute angular power spectrum with fallback to basic computation."""
        try:
            return ccl.angular_cl(cosmo_ccl, tracer1, tracer2, ell, **angular_cl_kwargs)
        except Exception as e:
            warnings.warn(f"Error with advanced angular_cl options: {e}. Falling back to basic computation.")
            return ccl.angular_cl(cosmo_ccl, tracer1, tracer2, ell)
    
    def compute_galaxy_galaxy_cl(self, block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict) -> None:
        """Compute galaxy-galaxy angular power spectra."""
        if not self.config['compute_gc'] or 'number_counts' not in tracers:
            return
        
        lenses = tracers['number_counts']
        nbin_lens = len(lenses)
        ell = self.config['ell']
        n_ell = self.config['n_ell']
        angular_cl_kwargs = self._prepare_angular_cl_kwargs()
        angular_cl_kwargs['l_limber'] = self.config.get('l_limber_gc', -1) 
        #print('angular_cl_kwargs (galaxy) ', angular_cl_kwargs)
        
        cl_gg = np.zeros((nbin_lens, nbin_lens, n_ell))
        
        for i in range(nbin_lens):
            for j in range(nbin_lens):
                cl_gg[i,j] = self._compute_cl_safely(cosmo_ccl, lenses[i], lenses[j], ell, angular_cl_kwargs) 
                block['galaxy_cl', f'bin_{i+1}_{j+1}'] = cl_gg[i,j]

        # Store metadata
        self._store_cl_metadata(block, 'galaxy_cl', ell, nbin_lens, nbin_lens)
    
    def compute_shear_shear_cl(self, block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict) -> None:
        """Compute shear-shear angular power spectra."""
        if not self.config['compute_shear'] or 'weak_lensing' not in tracers:
            return
        
        sources = tracers['weak_lensing']
        nbin_source = len(sources)
        ell = self.config['ell']
        n_ell = self.config['n_ell']
        angular_cl_kwargs = self._prepare_angular_cl_kwargs()
        angular_cl_kwargs['l_limber'] = self.config.get('l_limber_shear', -1) 
        #print('angular_cl_kwargs (shear) ', angular_cl_kwargs)
        
        cl_ll = np.zeros((nbin_source, nbin_source, n_ell))
        
        for i in range(nbin_source):
            for j in range(i, nbin_source):
                cl_ll[i, j] = self._compute_cl_safely(cosmo_ccl, sources[i], sources[j], ell, angular_cl_kwargs) # Symmetry
                block['shear_cl', f'bin_{i+1}_{j+1}'] = cl_ll[i, j]
                if i != j:
                    #cl_ll[j, i] = cl_ll[i, j] 
                    block['shear_cl', f'bin_{j+1}_{i+1}'] = cl_ll[i, j]

        # Store metadata
        self._store_cl_metadata(block, 'shear_cl', ell, nbin_source, nbin_source)
    
    def compute_galaxy_shear_cl(self, block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict) -> None:
        """Compute galaxy-shear cross angular power spectra."""
        if not self.config['compute_cross'] or 'number_counts' not in tracers or 'weak_lensing' not in tracers:
            return
        
        lenses = tracers['number_counts']
        sources = tracers['weak_lensing']
        nbin_lens = len(lenses)
        nbin_source = len(sources)
        ell = self.config['ell']
        angular_cl_kwargs = self._prepare_angular_cl_kwargs()
        angular_cl_kwargs['l_limber'] = self.config.get('l_limber_cross', -1) 
        #print('angular_cl_kwargs (galaxy-shear) ', angular_cl_kwargs)
        
        for i in range(nbin_lens):
            for j in range(nbin_source):
                cl_xc = self._compute_cl_safely(cosmo_ccl, lenses[i], sources[j], ell, angular_cl_kwargs)
                block['galaxy_shear_cl', f'bin_{i+1}_{j+1}'] = cl_xc
        
        # Store metadata
        self._store_cross_cl_metadata(block, 'galaxy_shear_cl', ell, nbin_lens, nbin_source)
    
    def compute_cmb_cross_cl(self, block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict) -> None:
        """Compute CMB lensing cross-correlations."""
        if not self.config['compute_cmb_lensing'] or 'cmb_lensing' not in tracers:
            return
        
        cmb_tracer = tracers['cmb_lensing'][0]
        ell = self.config['ell']
        angular_cl_kwargs = self._prepare_angular_cl_kwargs()
        
        # CMB lensing - galaxy clustering
        if 'number_counts' in tracers:
            lenses = tracers['number_counts']
            nbin_lens = len(lenses)
            for i in range(nbin_lens):
                cl_cmb_gc = self._compute_cl_safely(cosmo_ccl, cmb_tracer, lenses[i], ell, angular_cl_kwargs)
                block['cmb_galaxy_cl', f'bin_1_{i+1}'] = cl_cmb_gc
            
            self._store_cross_cl_metadata(block, 'cmb_galaxy_cl', ell, 1, nbin_lens)
        
        # CMB lensing - cosmic shear
        if 'weak_lensing' in tracers:
            sources = tracers['weak_lensing']
            nbin_source = len(sources)
            for i in range(nbin_source):
                cl_cmb_wl = self._compute_cl_safely(cosmo_ccl, cmb_tracer, sources[i], ell, angular_cl_kwargs)
                block['cmb_shear_cl', f'bin_1_{i+1}'] = cl_cmb_wl
            
            self._store_cross_cl_metadata(block, 'cmb_shear_cl', ell, 1, nbin_source)
    
    def _store_cl_metadata(self, block: Any, section_name: str, ell: np.ndarray, 
                          nbin_a: int, nbin_b: int) -> None:
        """Store metadata for auto-correlation power spectra."""
        block[section_name, 'ell'] = ell
        block[section_name, 'nbin'] = nbin_a
        block[section_name, 'nbin_a'] = nbin_a
        block[section_name, 'nbin_b'] = nbin_b
        block[section_name, 'save_name'] = section_name
        block[section_name, 'is_auto'] = False
        block[section_name, 'sep_name'] = "ell"

        if section_name == 'galaxy_cl':
            block[section_name, 'sample_a'] = "lens"
            block[section_name, 'sample_b'] = "lens"
        elif section_name == 'shear_cl':
            block[section_name, 'sample_a'] = "source"
            block[section_name, 'sample_b'] = "source"
    
    
    def _store_cross_cl_metadata(self, block: Any, section_name: str, ell: np.ndarray, 
                                nbin_a: int, nbin_b: int) -> None:
        """Store metadata for cross-correlation power spectra."""
        block[section_name, 'ell'] = ell
        block[section_name, 'nbin_a'] = nbin_a
        block[section_name, 'nbin_b'] = nbin_b
        block[section_name, 'save_name'] = section_name
        block[section_name, 'is_auto'] = False
        block[section_name, 'sep_name'] = "ell"
        if section_name == 'galaxy_shear_cl':
            block[section_name, 'sample_a'] = "lens"
            block[section_name, 'sample_b'] = "source"
        elif section_name == 'cmb_galaxy_cl':
            block[section_name, 'sample_a'] = "cmb_lensing"
            block[section_name, 'sample_b'] = "lens"
    
    def compute_all_angular_cl(self, block: Any, cosmo_ccl: ccl.Cosmology, tracers: Dict) -> None:
        """Compute all angular power spectra."""
        self.compute_galaxy_galaxy_cl(block, cosmo_ccl, tracers)
        self.compute_shear_shear_cl(block, cosmo_ccl, tracers)
        self.compute_galaxy_shear_cl(block, cosmo_ccl, tracers)
        self.compute_cmb_cross_cl(block, cosmo_ccl, tracers)
