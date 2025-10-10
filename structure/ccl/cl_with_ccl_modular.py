"""
Enhanced CCL Integration for CosmoSIS - Modular Version

This is the main interface file that orchestrates the modular CCL integration.
All functionality has been split into focused, manageable modules.

Main Components:
- CCLConfig: Configuration parsing and management
- CCLCosmologyManager: Cosmology creation and basic calculations
- CCLTracerManager: Tracer creation and management
- CCLAngularPowerSpectra: Angular power spectra calculations
- CCLCorrelationFunctions: All correlation function calculations
"""

import warnings
from timeit import default_timer as timer
from datetime import timedelta
from typing import Any, Dict

# Import our modular components
from ccl_config import CCLConfig
from ccl_cosmology import CCLCosmologyManager
from ccl_tracers import CCLTracerManager
from ccl_angular_cl import CCLAngularPowerSpectra
from ccl_correlations import CCLCorrelationFunctions


class CCLIntegration:
    """
    Main CCL integration class that orchestrates all computations.
    
    This class provides a clean interface to the full CCL functionality
    while keeping the implementation modular and maintainable.
    """
    
    def __init__(self, config: CCLConfig):
        """Initialize with configuration."""
        self.config = config
        self.cosmology_manager = CCLCosmologyManager(config.config)
        self.tracer_manager = CCLTracerManager(config.config)
        self.angular_cl_calculator = CCLAngularPowerSpectra(config.config)
        self.correlation_calculator = CCLCorrelationFunctions(config.config)
        
        self.cosmo_ccl = None
        self.tracers = {}
    
    def run_full_calculation(self, block: Any) -> int:
        """
        Run the complete CCL calculation pipeline.
        
        Returns:
            0 for success, 1 for failure
        """
        start_time = timer()
        
        # Step 1: Create CCL cosmology
        self.cosmo_ccl = self.cosmology_manager.create_cosmology(block)
        print("✓ CCL cosmology initialized")
        
        # Step 2: Compute basic cosmological quantities
        self._compute_basic_quantities(block)
        
        # Step 3: Create tracers
        self.tracers = self.tracer_manager.create_all_tracers(block, self.cosmo_ccl)
        print(f"✓ Created tracers: {list(self.tracers.keys())}")
        
        # Step 4: Compute angular power spectra
        self.angular_cl_calculator.compute_all_angular_cl(block, self.cosmo_ccl, self.tracers)
        print("✓ Angular power spectra computed")
        
        # Step 5: Compute correlation functions
        self.correlation_calculator.compute_all_correlations(block, self.cosmo_ccl, self.tracers)
        if self.config.get('compute_xi', False) or self.config.get('compute_xi_3d', False):
            print("✓ Correlation functions computed")
        
        elapsed_time = timer() - start_time
        print(f"✓ CCL computation completed in {timedelta(seconds=elapsed_time)}")
        
        return 0
            
    
    def _compute_basic_quantities(self, block: Any) -> None:
        """Compute basic cosmological quantities."""
        quantities_computed = []
        
        if self.config.get('compute_background', False):
            self.cosmology_manager.compute_background_quantities(block)
            quantities_computed.append("background")
        
        if self.config.get('compute_growth', False):
            self.cosmology_manager.compute_growth_factors(block)
            quantities_computed.append("growth")
        
        if self.config.get('compute_power_spectra', False):
            self.cosmology_manager.compute_power_spectra(block)
            quantities_computed.append("power spectra")
        
        if self.config.get('compute_halo_model', False):
            self.cosmology_manager.compute_halo_model(block)
            quantities_computed.append("halo model")
        
        if quantities_computed:
            print(f"✓ Computed: {', '.join(quantities_computed)}")


# ===== CosmoSIS Interface Functions =====

def setup(options):
    """
    Setup function called by CosmoSIS.
    
    Returns:
        Configuration object for use in execute()
    """
    try:
        config = CCLConfig(options)
        print("✓ CCL configuration loaded successfully")
        print(f"  - Angular power spectra: {config.get('compute_gc', False) or config.get('compute_shear', False)}")
        print(f"  - Correlation functions: {config.get('compute_xi', False)}")
        print(f"  - 3D correlations: {config.get('compute_xi_3d', False)}")
        print(f"  - Advanced features: emulators={config.get('use_emulator_pk', False)}, "
              f"magnification={config.get('use_magnification', False)}")
        return config
    except Exception as e:
        print(f"✗ Error in CCL setup: {e}")
        raise


def execute(block, config):
    """
    Execute function called by CosmoSIS.
    
    This is the main entry point that runs the CCL calculations.
    
    Returns:
        0 for success, 1 for failure
    """
    ccl_integration = CCLIntegration(config)
    return ccl_integration.run_full_calculation(block)


def cleanup(config):
    """
    Cleanup function called by CosmoSIS at the end of the chain.
    
    Currently no cleanup is needed for CCL.
    """
    pass


# ===== Module Information =====

if __name__ == "__main__":
    print("=" * 60)
    print("CCL Integration for CosmoSIS")
    print("=" * 60)
    print()
    print("This modular implementation provides:")
    print()
    print("🏗️  ARCHITECTURE:")
    print("   • CCLConfig: Configuration management")
    print("   • CCLCosmologyManager: Cosmology & basic calculations")
    print("   • CCLTracerManager: Tracer creation & management")
    print("   • CCLAngularPowerSpectra: Angular power spectra")
    print("   • CCLCorrelationFunctions: All correlation functions")
    print()
    print("🚀 FEATURES:")
    print("   • Full CCL angular_cl advanced options")
    print("   • All CCL correlation types (NN, NG, GG+, GG-)")
    print("   • 3D correlations with RSD and multipoles")
    print("   • Emulator support (BaccoEmu, etc.)")
    print("   • Magnification bias and perturbative tracers")
    print("   • CMB lensing cross-correlations")
    print()
    print("📁 FILES:")
    print("   • cl_with_ccl_modular.py - Main interface (this file)")
    print("   • ccl_config.py - Configuration management")
    print("   • ccl_cosmology.py - Cosmology and basic calculations")
    print("   • ccl_tracers.py - Tracer creation and management")
    print("   • ccl_angular_cl.py - Angular power spectra")
    print("   • ccl_correlations.py - Correlation functions")
    print()
    print("✨ BENEFITS:")
    print("   • Modular, maintainable code structure")
    print("   • Clear separation of concerns")
    print("   • Easy to extend and modify")
    print("   • Comprehensive error handling")
    print("   • Full CCL feature coverage")
    print("=" * 60)
