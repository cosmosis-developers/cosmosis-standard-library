# Enhanced CCL Integration for CosmoSIS

This directory contains a comprehensive integration of the [Core Cosmology Library (CCL)](https://github.com/LSSTDESC/CCL) into CosmoSIS, providing access to the full range of CCL functionality.

## Overview

The Core Cosmology Library (CCL) is a standardized library of routines to calculate basic observables used in cosmology, developed by the LSST Dark Energy Science Collaboration (DESC). This integration brings all major CCL features into CosmoSIS.

## Features

### Background Cosmology
- **Distances**: Angular diameter, luminosity, comoving radial, and comoving transverse distances
- **Hubble Parameter**: H(z)/H₀ evolution
- **Age**: Age of the universe as a function of redshift
- **Sound Horizon**: Sound horizon at drag epoch

### Growth of Structure
- **Growth Factor**: Linear growth factor D(z)
- **Growth Rate**: Linear growth rate f(z) = d ln D/d ln a

### Power Spectra
- **Linear Matter Power Spectrum**: P_lin(k,z) from multiple methods
- **Non-linear Matter Power Spectrum**: P_nl(k,z) with various prescriptions
- **Transfer Functions**: Support for CAMB, CLASS, BBKS, and ISITGR
- **Emulator Support**: BaccoemuNonlinear, EuclidEmulator2, CosmicEmu, BaccoemuTransfer
- **Baryonic Effects**: BaccoEmu and Schneider correction models
- **Modified Gravity**: μ-Σ parametrization support
- **Advanced Normalization**: Support for σ₈ and σ₈,cb (cold+baryon) normalization

### Halo Model Framework
- **Mass Functions**: Tinker08, Tinker10, Watson13, Bocquet16
- **Halo Bias**: Tinker10, Sheth99, Bhattacharya11
- **Concentration Relations**: Duffy2008, Bhattacharya13, Prada12
- **Halo Profiles**: NFW, Einasto, Hernquist
- **Mass Definitions**: 200m, 200c, virial, 500c

### Tracers and Observables
- **Weak Lensing**: Cosmic shear with intrinsic alignment and magnification bias
- **Galaxy Clustering**: Number counts with bias modeling, RSD, and magnification bias
- **Perturbative Bias**: BaccoLbiasCalculator for advanced bias modeling
- **CMB Lensing**: CMB convergence field
- **ISW Effect**: Integrated Sachs-Wolfe effect
- **Magnification Bias**: Full support for lensing magnification effects
- **Custom Tracers**: Framework for user-defined tracers

### Angular Power Spectra
- **Auto-correlations**: Galaxy-galaxy, shear-shear
- **Cross-correlations**: Galaxy-shear, CMB lensing cross-correlations
- **Advanced Integration**: Limber and non-Limber options with configurable thresholds
- **Multiple Methods**: QAG quadrature, spline, and Riemann integration
- **High Accuracy**: Configurable tolerance parameters and sampling control
- **Advanced Sampling**: Logarithmic and linear ℓ-space sampling with custom step sizes

### Correlation Functions
- **Angular Correlations**: All CCL correlation types (NN, NG, GG+, GG-) with multiple methods
- **3D Correlations**: Real-space 3D correlation functions ξ(r)
- **RSD Correlations**: Redshift-space distortions with ξ(r,μ) and ξ(π,σ)
- **Multipoles**: Correlation function multipoles ξₗ(r) for ℓ = 0, 2, 4, ...
- **Advanced Methods**: FFTLog, Bessel function, and Legendre polynomial methods
- **Flexible Binning**: Configurable angular and radial ranges

## Installation Requirements

1. **pyccl**: Install the Python CCL library
   ```bash
   conda install -c conda-forge pyccl
   # or
   pip install pyccl
   ```

2. **CosmoSIS**: Ensure CosmoSIS is properly installed and configured

## Usage

### Basic Configuration

Add the CCL module to your CosmoSIS pipeline:

```ini
[pipeline]
modules = ... ccl_comprehensive ...

[ccl_comprehensive]
file = cosmosis-standard-library/structure/ccl/cl_with_ccl_modular.py

; Enable basic functionality
compute_background = T
compute_power_spectra = T
compute_gc = T
compute_shear = T
compute_cross = T

; Configure ell range
ell_min_logspaced = 10.0
ell_max_logspaced = 10000.0
n_ell_logspaced = 100
```

### Advanced Configuration

For comprehensive functionality, see `example_ccl.ini` for a full configuration example, or `example_advanced_ccl.ini` for advanced emulator and perturbative features.

### Advanced Features Example

To replicate the advanced CCL usage pattern you described:

```python
# Your Python code:
bemu_nl = ccl.BaccoemuNonlinear()
cosmo_nl = ccl.Cosmology(Omega_c=..., sigma8=params['sigma8_cb'], 
                         matter_power_spectrum=bemu_nl, ...)
heft = ccl.nl_pt.BaccoLbiasCalculator(cosmo=cosmo_nl, ...)
```

**CosmoSIS Configuration:**
```ini
[ccl_comprehensive]
file = cosmosis-standard-library/structure/ccl/cl_with_ccl_modular.py

# Use BaccoemuNonlinear emulator
use_emulator_pk = T
emulator_pk_type = "baccoemu"

# Use sigma8_cb normalization
use_sigma8_cb = T

# Enable perturbative bias calculator
use_perturbative_bias = T
perturbative_bias_type = "bacco_lbias"
log10k_min_pt = -4.0
log10k_max_pt = 2.0
nk_per_decade_pt = 20

# Enable magnification bias
use_magnification = T
magnification_alpha = 2.5
```

This configuration automatically handles:
- Creating `BaccoemuNonlinear()` emulator object
- Setting up `BaccoLbiasCalculator` with specified k-range
- Adding magnification bias to tracers
- Using σ₈,cb normalization instead of total matter σ₈

### Advanced CCL Features

The enhanced integration now supports all advanced CCL options:

#### **Angular Power Spectra (`ccl.angular_cl`)**
```ini
# Advanced integration control
limber_integration = F          # Disable Limber for high accuracy
non_limber_max_ell = 200       # Exact integration up to ℓ=200
integration_method = "spline"   # Use spline integration
relative_tolerance = 1.0e-6     # High precision
l_logstep = 1.05               # Fine ℓ sampling
dchi = 1.0                     # Custom radial sampling
```

#### **Correlation Functions (`ccl.correlation`)**
```ini
# Angular correlation functions with all CCL types
compute_xi = T
xi_type = "gg+"                # NN, NG, GG+, GG- types supported
xi_method = "bessel"           # FFTLog, Bessel, Legendre methods

# 3D correlation functions
compute_xi_3d = T              # ξ(r) real-space
compute_xi_rsd = T             # ξ(r,μ) and ξ(π,σ) with RSD
compute_xi_multipoles = T      # ξₗ(r) multipoles
rsd_beta = 0.4                 # f/b parameter
multipole_ells = "0 2 4 6"     # Custom multipoles
```

#### **Available Correlation Types**
Based on CCL documentation:
- **`NN`**: Galaxy-galaxy (0×0 spin)
- **`NG`**: Galaxy-shear (0×2 spin) 
- **`GG+`**: Shear ξ₊ (2×2 spin)
- **`GG-`**: Shear ξ₋ (2×2 spin)

#### **Integration Methods**
- **`fftlog`**: Fast FFT-based (default, fastest)
- **`bessel`**: Direct Bessel function integration (most accurate)
- **`legendre`**: Legendre polynomial sum (alternative method)

### Input Requirements

The module expects certain sections to be present in the data block:

#### Required
- `[cosmological_parameters]`: Basic cosmological parameters
  - `omega_c`, `omega_b`, `h0`, `n_s`, `TCMB`, `mnu`
  - Either `a_s` or `sigma_8` for normalization

#### Optional (depending on enabled features)
- `[nz_source]`: Source redshift distributions for weak lensing
- `[nz_lens]`: Lens redshift distributions for galaxy clustering  
- `[bin_bias]`: Galaxy bias values for each lens bin
- `[intrinsic_alignment_parameters]`: IA model parameters

## Output Data Products

### Background Quantities
- `[distances]`: All distance measures, Hubble parameter, age, sound horizon

### Growth Parameters  
- `[growth_parameters]`: Growth factor D(z) and growth rate f(z)

### Power Spectra
- `[matter_power_lin]`: Linear matter power spectrum P_lin(k,z)
- `[matter_power_nl]`: Non-linear matter power spectrum P_nl(k,z)

### Angular Power Spectra
- `[galaxy_cl]`: Galaxy clustering C_ℓ^ij
- `[shear_cl]`: Weak lensing shear C_ℓ^ij  
- `[galaxy_shear_cl]`: Galaxy-shear cross C_ℓ^ij

### Correlation Functions (if enabled)
- `[galaxy_xi_plus]`: Galaxy clustering ξ₊(θ)
- `[shear_xi_plus]`: Shear correlation ξ₊(θ)
- `[shear_xi_minus]`: Shear correlation ξ₋(θ)

### Halo Model (if enabled)
- `[halo_model_parameters]`: Mass function dn/dM and halo bias b(M,z)

## Configuration Parameters

### Computation Control
- `compute_background`: Enable background calculations
- `compute_power_spectra`: Enable power spectrum calculations
- `compute_growth`: Enable growth factor calculations
- `compute_halo_model`: Enable halo model calculations

### Observable Selection
- `compute_gc`: Galaxy clustering angular power spectra
- `compute_shear`: Weak lensing angular power spectra
- `compute_cross`: Galaxy-shear cross angular power spectra
- `compute_cmb_lensing`: CMB lensing cross-correlations
- `compute_xi`: Real-space correlation functions

### CCL Configuration
- `transfer_function`: Transfer function method
- `matter_power_spectrum`: Non-linear power spectrum method
- `baryonic_model`: Baryonic feedback model
- `mg_model`: Modified gravity model

### Accuracy Parameters
- `limber_integration`: Use Limber approximation
- `non_limber_max_ell`: Maximum ℓ for exact integration
- Redshift and k-space sampling parameters

## Performance Notes

- **Memory Usage**: Large ℓ ranges and many redshift bins increase memory requirements
- **Computation Time**: Non-Limber integration is slower but more accurate at low ℓ
- **Halo Model**: Enabling halo model calculations adds significant computation time
- **Baryonic Effects**: BaccoEmu emulator adds initialization overhead but is fast for predictions

## Examples and Validation

The module has been tested against:
- Standard CCL examples and benchmarks
- Comparison with other CosmoSIS modules (CAMB, CLASS)
- Cross-validation of angular power spectra and correlation functions

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure pyccl is installed and accessible
2. **Memory Issues**: Reduce ell range or redshift sampling
3. **Convergence Issues**: Check cosmological parameter ranges
4. **Missing Data**: Verify required input sections are present

### Performance Optimization

1. **Limber Approximation**: Use for ℓ > 100 to improve speed
2. **Reduced Sampling**: Lower n_ell, n_z, n_k for faster computation
3. **Selective Computation**: Disable unused features
4. **Parallel Processing**: CCL uses OpenMP when available

## Citation

If you use this CCL integration in your research, please cite:

1. **CCL Paper**: Chisari et al. (2019), arXiv:1812.05995
2. **CCL Code**: https://github.com/LSSTDESC/CCL
3. **CosmoSIS**: Zuntz et al. (2015), Astronomy and Computing, 12, 45-59

## Contributing

Contributions to improve this integration are welcome. Please:
1. Follow existing code style and documentation standards
2. Add appropriate tests for new functionality
3. Update this README for significant changes
4. Consider backwards compatibility

## Contact

For issues specific to this CCL integration, please open an issue in the CosmoSIS repository. For CCL-specific questions, refer to the [CCL documentation](https://ccl.readthedocs.io/) and [GitHub repository](https://github.com/LSSTDESC/CCL).
