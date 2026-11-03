"""
A wrapper for the HMcode2020Emu emulator, model_v2 (06-2024), written by
Felipe Andrade-Oliveira. See the README for installation.

Known issues: the emulator was trained with a single massive neutrino species.
"""
from cosmosis.datablock import names, option_section
import warnings
import numpy as np

cosmo = names.cosmological_parameters
hmpar = names.halo_model_parameters

# Limits of the range where the emulator is valid
Z_EMU_MAX = 3.0
K_EMU_MIN = 1e-4
K_EMU_MAX = 49.99999
# The emulator only gives a non-linear power spectrum above this k
K_NL_MIN = 0.01


def make_params(block, z, k):
    """Build the dictionary of parameters that HMcode2020Emu needs, for
    several redshifts but one set of cosmological and astrophysical parameters."""
    n_z = len(z)

    def repeated(value):
        return [value] * n_z

    return {
        'As': repeated(block.get_double(cosmo, "a_s")),
        'hubble': repeated(block.get_double(cosmo, 'h0')),
        'log10TAGN': repeated(block.get_double(hmpar, "logt_agn")),
        'neutrino_mass': repeated(block.get_double(cosmo, 'mnu')),
        'ns': repeated(block.get_double(cosmo, 'n_s')),
        'omega_baryon': repeated(block.get_double(cosmo, "omega_b")),
        'omega_cdm': repeated(block.get_double(cosmo, "omega_c")),
        'w0': repeated(block.get_double(cosmo, 'w')),
        'wa': repeated(block.get_double(cosmo, 'wa', default=0.0)),
        'z': z,
        'k': k,
    }


def setup(options):
    try:
        import HMcode2020Emu
    except ImportError:
        raise RuntimeError("The HMcode2020Emu python module is required for this module. "
                           "See the README for how to install it.")

    config = {}
    config['emulator'] = HMcode2020Emu.Matter_powerspectrum()
    config['zmin'] = options.get_double(option_section, 'zmin', default=0.0)
    config['zmax'] = options.get_double(option_section, 'zmax', default=3.01)
    config['nz'] = options.get_int(option_section, 'nz', default=150)
    config['kmin'] = options.get_double(option_section, 'kmin', default=K_EMU_MIN)
    config['kmax'] = options.get_double(option_section, 'kmax', default=K_EMU_MAX)
    config['nk'] = options.get_int(option_section, 'nk', default=701)

    if config['zmax'] > Z_EMU_MAX + 0.01:
        raise ValueError(f"zmax={config['zmax']} is out of the emulator range (z<{Z_EMU_MAX})")
    if config['kmin'] < K_EMU_MIN:
        raise ValueError(f"kmin={config['kmin']} is out of the emulator range (k>{K_EMU_MIN})")
    if config['kmax'] > K_EMU_MAX + 1:
        raise ValueError(f"kmax={config['kmax']} is out of the emulator range (k<{K_EMU_MAX})")
    if options.has_value(option_section, 'kmax_extrapolate'):
        warnings.warn("kmax_extrapolate is not implemented in this module, and is ignored. "
                      "Use an extrapolation module afterwards instead.")

    warnings.warn("hmcode2020emu: do not use sigma_12, which is not computed. "
                  "Sample sigma_8 or S_8 instead if you need to.")

    return config


def execute(block, config):
    emulator = config['emulator']

    k = np.logspace(np.log10(config['kmin']), np.log10(config['kmax']), config['nk'])
    if k.max() > K_EMU_MAX:
        warnings.warn(f"Maximum k={k.max()} is above the emulator maximum of {K_EMU_MAX}")

    # The emulator is only valid for z < 3 so we cut the grid there
    z = np.linspace(config['zmin'], config['zmax'], config['nz'])
    z = z[z < Z_EMU_MAX]

    params = make_params(block, z, k)
    params_nl = make_params(block, z, k[k >= K_NL_MIN])

    k_lin, p_lin = emulator.get_linear_pk(**params, no_nu=False)
    k_nl, p_nl = emulator.get_nonlinear_pk(baryonic_boost=True, **params_nl)

    # Use the linear power at low k, where the emulator does not give the
    # non-linear one, and the emulated non-linear power above that.
    low_k = k_lin < k_nl[0]
    p_nl = np.concatenate((p_lin[:, low_k], p_nl), axis=1)
    k_h = np.concatenate((k_lin[low_k], k_nl))

    sigma_8, fsigma_8 = emulator.get_sigma8(**params)
    block[names.growth_parameters, "sigma_8"] = sigma_8
    block[names.growth_parameters, "fsigma_8"] = fsigma_8
    block[cosmo, "sigma_8"] = sigma_8[0]
    block[cosmo, "S_8"] = sigma_8[0] * np.sqrt(block[cosmo, "omega_m"] / 0.3)

    block.put_grid("matter_power_lin", "z", z, "k_h", k_lin, "P_k", p_lin)
    block.put_grid("matter_power_nl", "z", z, "k_h", k_h, "P_k", p_nl)

    return 0


def cleanup(config):
    pass
