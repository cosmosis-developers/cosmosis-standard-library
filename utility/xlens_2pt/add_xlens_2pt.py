from cosmosis.datablock import names, option_section
import numpy as np

def setup(options):
	# Get the name of the xlens section
    x_section = options.get_string(option_section,"xlens", "xlens")

    galaxy_shear_section = options.get_string(option_section, "galaxy_shear_section", default="galaxy_shear_xi")
    
    return x_section, galaxy_shear_section


def execute(block, config):

    x_section, galaxy_shear_section = config

    xlens_val = block[x_section, 'xlens_all']

    n_a, n_b = get_nbins(block, galaxy_shear_section)
    # I default setup, a is lens, b is source, but this could change
    # sample_a and sample_b values will contain this info in the block

    for i in range(1,n_a+1):
        for j in range(1,n_b+1):
            ggl_bin_label = 'bin_'+str(i)+'_'+str(j)
            ggl = block[galaxy_shear_section,ggl_bin_label]
            ggl_xlens = xlens_val*ggl
            block[galaxy_shear_section, ggl_bin_label] = ggl_xlens

    return 0


def get_nbins(block, section):
    if block.has_value(section, "nbin_a"):
        n_a = block[section, "nbin_a"]
        n_b = block[section, "nbin_b"]
    else:
        n_a = block[section, "nbin"]
        n_b = n_a
    return n_a, n_b
