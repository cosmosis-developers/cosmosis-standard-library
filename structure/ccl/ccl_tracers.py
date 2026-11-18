"""
CCL Tracer Manager

Handles creation and management of CCL tracers including perturbative calculators.
"""

import numpy as np
import warnings
from typing import Dict, List, Any, Optional

import pyccl as ccl
from cosmosis.datablock import names

# CosmoSIS section names
ia = names.intrinsic_alignment_parameters


class CCLTracerManager:
    """Manages CCL tracer creation and perturbative calculators."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
        self.tracers = {}
        self.pt_calculator = None
    
    def create_perturbative_calculator(self, cosmo_ccl: ccl.Cosmology) -> Optional[Any]:
        """Create perturbative bias calculator if requested."""
        if not self.config.get('use_perturbative_bias', False):
            return None
            
        calc_type = self.config.get('perturbative_bias_type', 'bacco_lbias')
        log10k_min = self.config.get('log10k_min_pt', -4.0)
        log10k_max = self.config.get('log10k_max_pt', 2.0)
        nk_per_decade = self.config.get('nk_per_decade_pt', 20)
        
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
    
    def create_weak_lensing_tracers(self, block: Any, cosmo_ccl: ccl.Cosmology) -> List:
        """Create weak lensing tracers."""
        if not self.config['compute_shear'] or not block.has_section("nz_source"):
            return []
        
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
        if self.config.get('use_magnification', False):
            alpha = self.config.get('magnification_alpha', 2.5)
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
        
        return sources
    
    def create_number_counts_tracers(self, block: Any, cosmo_ccl: ccl.Cosmology) -> List:
        """Create galaxy clustering (number counts) tracers."""
        if not self.config['compute_gc'] or not block.has_section("nz_lens"):
            return []
        
        section_name = "nz_lens"
        nbin_lens = block[section_name, "nbin"]
        z = block[section_name, "z"]
        
        lenses = []
        for i in range(1, nbin_lens + 1):
            # Galaxy bias - can be from perturbative calculator or simple bias
            if self.pt_calculator is not None:
                # Use perturbative bias from calculator
                # Note: This is a simplified example - in practice you'd need to 
                # compute the bias at each redshift using the calculator
                bias_vals = np.ones(len(z)) * 1.5  # Placeholder - would use pt_calculator
            elif block.has_section("bin_bias"):
                bias_vals = block["bin_bias", f"b{i}"] * np.ones(len(z))
            else:
                bias_vals = np.ones(len(z))
            
            # Magnification bias for lenses (if enabled)
            mag_bias = None
            if self.config.get('use_magnification', False):
                alpha = self.config.get('magnification_alpha', 2.5)
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
        
        return lenses
    
    def create_cmb_lensing_tracers(self, cosmo_ccl: ccl.Cosmology) -> List:
        """Create CMB lensing tracers."""
        if not self.config['compute_cmb_lensing']:
            return []
        
        cmb_tracer = ccl.CMBLensingTracer(cosmo_ccl, z_source=1100)
        return [cmb_tracer]
    
    def create_isw_tracers(self, cosmo_ccl: ccl.Cosmology) -> List:
        """Create ISW tracers."""
        if not self.config['compute_isw']:
            return []
        
        isw_tracer = ccl.ISWTracer(cosmo_ccl)
        return [isw_tracer]
    
    def create_all_tracers(self, block: Any, cosmo_ccl: ccl.Cosmology) -> Dict[str, List]:
        """Create all CCL tracers for different observables."""
        self.tracers = {}
        
        # Create perturbative calculator if needed
        self.pt_calculator = self.create_perturbative_calculator(cosmo_ccl)
        
        # Create all tracer types
        weak_lensing = self.create_weak_lensing_tracers(block, cosmo_ccl)
        if weak_lensing:
            self.tracers['weak_lensing'] = weak_lensing
        
        number_counts = self.create_number_counts_tracers(block, cosmo_ccl)
        if number_counts:
            self.tracers['number_counts'] = number_counts
        
        cmb_lensing = self.create_cmb_lensing_tracers(cosmo_ccl)
        if cmb_lensing:
            self.tracers['cmb_lensing'] = cmb_lensing
        
        isw = self.create_isw_tracers(cosmo_ccl)
        if isw:
            self.tracers['isw'] = isw
        
        # Store perturbative calculator for later use
        if self.pt_calculator is not None:
            self.tracers['pt_calculator'] = self.pt_calculator
        
        return self.tracers
    
    def get_tracers(self) -> Dict[str, List]:
        """Get all created tracers."""
        return self.tracers
