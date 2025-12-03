from cosmosis.datablock import option_section, names
import numpy as np
import scipy.integrate

def setup(options):
    z = options[option_section, 'z']
    section = options.get_string(
        option_section, 'output_section', names.wl_number_density
    )
    zmin = options.get_double(option_section, 'zmin', 0.0)
    zmax = options.get_double(option_section, 'zmax', 3.0)
    dz = options.get_double(option_section, 'dz', 0.01)

    # In case there is just one bin, convert to arrays
    if np.isscalar(z):
        z = [z]

    return z, section, zmin, zmax, dz

def delta(x, mu):
    dx = x[1] - x[0]
    idx = (np.abs(x - mu)).argmin()
    ddf = np.zeros_like(x)
    ddf[idx] = 1.0 / dx
    return ddf

def execute(block, config):
    mu, section, zmin, zmax, dz = config
    nbin = len(mu)

    # Set up z array. Make sure ZMAX is inclusive
    z = np.arange(zmin, zmax + dz, dz)

    # Save Z array and count info
    block[section, 'Z'] = z
    block[section, 'NZ'] = len(z)
    block[section, 'NBIN'] = nbin

    for i in range(1, nbin + 1):
        # Generate simple delta function window
        nz_bin = delta(z, mu[i - 1])
        # Save n(z)
        block[section, f'BIN_{i}'] = nz_bin

    return 0

def cleanup(config):
    # nothing to do here! We just include this
    # for completeness
    return 0
