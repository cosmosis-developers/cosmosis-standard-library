from __future__ import print_function
from cosmosis.datablock import option_section, names
import numpy as np

# This short module is designed to stitch together the outputs of some combination
# of other IA codes to estimate the IA signal in a galaxy sample with an
# arbitrary mixture of red and blue galaxies.
# It is assumed that power spectra for BOTH red and blue galaxies have been
# separately saved to the datablock prior to this module being called.


def setup(options):
    catalogue = options[option_section, "catalogue"]
    return catalogue


def execute(block, config):
    catalogue = config

    section_II = names.ia_spectrum_ii
    section_GI = names.ia_spectrum_gi
    section_ia = names.intrinsic_alignment_parameters

    # Load nuisance parameters for the two models
    f_red = block[catalogue, 'red_fraction']
    f_blue = 1. - f_red
    A_red = block[section_ia, 'A_red']
    A_blue = block[section_ia, 'A_blue']
    alpha_red = block[section_ia, 'alpha_red']
    alpha_blue = block[section_ia, 'alpha_blue']

    # Define scaling grid
    _, z_grid = np.meshgrid(k, z)

    # Get relevant power spectra
    z, k, P_II_red = block.get_grid(section_ia,  "z", "k_h", "P_II_red")
    z, k, P_II_blue = block.get_grid(section_ia,  "z", "k_h", "P_II_blue")
    z, k, P_GI_red = block.get_grid(section_ia,  "z", "k_h", "P_GI_red")
    z, k, P_GI_blue = block.get_grid(section_ia,  "z", "k_h", "P_GI_blue")

    # Geometric mean of the red and blue II components
    P_II_red_blue = np.sqrt(P_II_red * P_II_blue)

    # Combine red, blue and cross terms
    P_II = f_red * f_red * A_red * A_red * \
        P_II_red * (1 + z_grid)**(2. * alpha_red)
    P_II += f_blue * f_blue * A_blue * A_blue * \
        P_II_blue * (1 + z_grid)**(2. * alpha_blue)
    P_II += f_red * f_blue * A_red * A_blue * P_II_red_blue * \
        (1 + z_grid)**(alpha_blue) * (1 + z_grid)**(alpha_red)

    P_GI = f_red * A_red * P_GI_red * (1 + z_grid)**(alpha_red)
    P_GI += f_blue * A_blue * P_GI_blue * (1 + z_grid)**(alpha_blue)

    block.put_grid(section_II, "z", z, "k_h", k, "P_II", P_II)
    print("Saved II spectrum for %s" % catalogue)
    block.put_grid(section_GI, "z", z, "k_h", k, "P_GI", P_GI)
    print("Saved GI spectrum for %s" % catalogue)

    return 0


def cleanup(config):
    pass
