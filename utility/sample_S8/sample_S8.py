"""
Sample S_8 = sigma_8 * (Omega_m / 0.3)**alpha instead of sigma_8, converting
it to a sigma_8 value.

This is a copy of a module from the KiDS KCAP repository.
Contributors: Tilman Troester and Marika Asgari
"""
import numpy as np

from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section

cosmo = section_names.cosmological_parameters


def setup(options):
    alpha = options.get_double(option_section, "alpha", default=0.5)
    read_s8_squared = options.get_bool(option_section, "read_s8_squared", default=False)
    return alpha, read_s8_squared


def execute(block, config):
    alpha, read_s8_squared = config

    if read_s8_squared:
        S8 = np.sqrt(block[cosmo, "S8_squared"])
    else:
        S8 = block[cosmo, "S_8"]

    if block.has_value(cosmo, "omega_m"):
        # Use Omega_m if available
        Omega_m = block[cosmo, "omega_m"]
    else:
        # Otherwise use (omch2+ombh2)/h**2
        omch2 = block[cosmo, "omch2"]
        ombh2 = block[cosmo, "ombh2"]
        h = block[cosmo, "h0"]
        Omega_m = (omch2 + ombh2) / h**2

    block[cosmo, "sigma_8"] = S8 / (Omega_m / 0.3) ** alpha

    return 0


def cleanup(config):
    pass
