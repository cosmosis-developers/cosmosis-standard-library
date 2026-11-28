"""
Apply the xlens scaling factor to the galaxy-galaxy lensing (lens-source)
two-point functions.
"""
from cosmosis.datablock import option_section


def setup(options):
    x_section = options.get_string(option_section, "xlens", "xlens")
    galaxy_shear_section = options.get_string(option_section, "galaxy_shear_section", default="galaxy_shear_xi")
    return x_section, galaxy_shear_section


def execute(block, config):
    x_section, galaxy_shear_section = config

    xlens_val = block[x_section, "xlens_all"]

    # In the default setup a is lens and b is source, but this could change.
    # The sample_a and sample_b values in the block contain this information.
    n_a, n_b = get_nbins(block, galaxy_shear_section)

    for i in range(1, n_a + 1):
        for j in range(1, n_b + 1):
            label = f"bin_{i}_{j}"
            block[galaxy_shear_section, label] = xlens_val * block[galaxy_shear_section, label]

    return 0


def cleanup(config):
    pass


def get_nbins(block, section):
    if block.has_value(section, "nbin_a"):
        return block[section, "nbin_a"], block[section, "nbin_b"]
    n = block[section, "nbin"]
    return n, n
