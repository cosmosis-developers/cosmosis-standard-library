"""
Sample S_8 = sigma_8 * (Omega_m / 0.3)**alpha instead of sigma_8, converting
it to a sigma_8 value for use by other modules (for example sigma8_rescale).

This is a copy of a module from the KiDS KCAP repository.
Contributors: Tilman Troester and Marika Asgari
"""
import numpy as np

from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section

cosmo = section_names.cosmological_parameters


def setup(options):
    alpha = options.get_double(option_section, "alpha", default=0.5)
    S8_squared = options.get_bool(option_section, "S8_squared", default=False)
    S8_input_name = options.get_string(option_section, "S8_name", default="S_8_input")
    sigma8_output_name = options.get_string(option_section, "sigma8_name", default="sigma_8_input")
    return alpha, S8_squared, S8_input_name, sigma8_output_name


def execute(block, config):
    alpha, S8_squared, S8_input_name, sigma8_output_name = config

    S8_input = block[cosmo, S8_input_name]
    if S8_squared:
        S8_input = np.sqrt(S8_input)

    if block.has_value(cosmo, "omega_m"):
        # Use Omega_m if available
        Omega_m = block[cosmo, "omega_m"]
    else:
        # Otherwise use (omch2+ombh2)/h**2
        omch2 = block[cosmo, "omch2"]
        ombh2 = block[cosmo, "ombh2"]
        h = block[cosmo, "h0"]
        Omega_m = (omch2 + ombh2) / h**2

    block[cosmo, sigma8_output_name] = S8_input / (Omega_m / 0.3) ** alpha

    return 0


def cleanup(config):
    pass
