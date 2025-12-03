from cosmosis.datablock import option_section
from cosmosis.datablock import names
import numpy as np
try:
    from scipy.integrate import trapezoid
except ImportError:
    from scipy.integrate import trapz as trapezoid

def setup(options):
    z = options[option_section, "z"]
    sigma = options[option_section, "sigma"]
    section = options.get_string(
        option_section, "output_section", names.wl_number_density)
    zmin = options.get_double(option_section, "zmin", 0.)
    zmax = options.get_double(option_section, "zmax", 3.)
    dz = options.get_double(option_section, "dz", 0.01)
    # in case just one bin, convert to arrays:
    if np.isscalar(z):
        z = [z]
    if np.isscalar(sigma):
        sigma = [sigma]
    assert len(z) == len(
        sigma), "sigma and z had different sizes in the ini file section for gaussian_window"

    return (z, sigma, section, zmin, zmax, dz)


def gaussian(x, mu, sigma):
    norm = np.sqrt(2 * np.pi) * sigma
    return np.exp(-0.5 * (x - mu)**2 / sigma**2) / norm


def execute(block, config):
    (mu, sigma, section, zmin, zmax, dz) = config
    nbin = len(mu)
    # Set up z array.  Make sure ZMAX is inclusive
    z = np.arange(zmin, zmax+dz, dz)

    # Save Z array and count info
    block[section, "Z"] = z
    block[section, "NZ"] = len(z)
    block[section, "NBIN"] = nbin

    for i in range(1, nbin + 1):
        # generate simple gaussian window
        nz_bin = gaussian(z, mu[i - 1], sigma[i - 1])
        # the bin may not quite go to zero before we get to the
        # edges so normalize it
        norm = trapezoid(nz_bin, z)
        nz_bin /= norm
        # Save n(z)
        block[section, "BIN_%d" % i] = nz_bin
    return 0


def cleanup(config):
    # nothing to do here!  We just include this
    # for completeness
    return 0
