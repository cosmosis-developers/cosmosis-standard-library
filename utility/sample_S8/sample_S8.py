################
#THIS MODULE IS A COPY FROM THE KIDS KCAP repository
#Contributors: Tilman Troester and Marika Asgari
#Need to be tested and chceked for our purpose
################

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
    # Get parameters from sampler and CAMB output
    S8_input = block[cosmo, S8_input_name]
    if S8_squared:
        S8_input = np.sqrt(S8_input)
    if (cosmo, "omega_m") in block:
        # Use Omega_m if available
        Omega_m = block[cosmo, "omega_m"]
    else:
        # Otherwise use (omch2+ombh2)/h**2
        omch2 = block[cosmo, "omch2"]
        ombh2 = block[cosmo, "ombh2"]
        h = block[cosmo, "h0"]
        Omega_m=(omch2+ombh2)/h**2

    sigma8_input = S8_input/(Omega_m/0.3)**alpha
    block[cosmo, sigma8_output_name] = sigma8_input
    # signal that everything went fine
    return 0


def cleanup(config):
    # nothing to do here!  We just include this
    # for completeness
    return 0
