from __future__ import print_function
from cosmosis.datablock import names, option_section
import numpy as np
import scipy.interpolate
try:
    import astropy.io.fits as pyfits
except ImportError:
    try:
        import pyfits
    except ImportError:
        raise RuntimeError(
            "You need astropy installed to use the module load_nz_fits; try running: pip install astropy.")


def ensure_starts_at_zero(z, nz):
    nbin = nz.shape[0]
    if z[0] > 0.00000001:
        z_new = np.zeros(len(z) + 1)
        z_new[1:] = z
        nz_new = np.zeros((nbin, len(z) + 1))
        nz_new[:, 1:] = nz
        print("        Putting n(0) = 0 at the start of the n(z)")
    else:
        z_new = z
        nz_new = nz

    return z_new, nz_new


def load_histogram_form(ext, upsampling):
    # Load the various z columns.
    # The cosmosis code is expecting something it can spline
    # so  we need to give it more points than this - we will
    # give it the intermediate z values (which just look like a step
    # function)
    zlow = ext.data['Z_LOW']
    zhigh = ext.data['Z_HIGH']

    # First bin.
    i = 1
    bin_name = 'BIN{0}'.format(i)
    nz = []

    if upsampling == 1:
        z = ext.data['Z_MID']
        # First bin.
        i = 1
        bin_name = 'BIN{0}'.format(i)

        # Load the n(z) columns, bin1, bin2, ...
        while bin_name in ext.data.names:
            col = ext.data[bin_name]
            nz.append(col)
            i += 1
            bin_name = 'BIN{0}'.format(i)

    else:

        z = np.linspace(0.0, zhigh[-1], len(zlow) * upsampling)
        sample_bin = np.digitize(z, zlow) - 1

        # Load the n(z) columns, bin1, bin2, ...
        while bin_name in ext.data.names:
            col = ext.data[bin_name][sample_bin]
            nz.append(col)
            i += 1
            bin_name = 'BIN{0}'.format(i)

    nbin = len(nz)
    print("        Found {0} bins".format(nbin))
    nz = np.array(nz)
    z, nz = ensure_starts_at_zero(z, nz)
    for col in nz:
        norm = np.trapz(col, z)
        col /= norm

    return z, nz


def setup(options):
    nz_file = options.get_string(option_section, "nz_file")
    data_sets = options.get_string(option_section, "data_sets")
    data_sets_output = options.get_string(option_section, "data_sets_output", data_sets) #Allow user to add new output name, but default to regular name
    upsampling = options.get_int(option_section, "upsampling", 1)
    prefix_extension = options.get_bool(
        option_section, "prefix_extension", True)
    prefix_section = options.get_bool(option_section, "prefix_section", True)
    
    data_sets = data_sets.split()
    data_sets_output = data_sets_output.split()
    
    if not data_sets:
        raise RuntimeError(
            "Option data_sets empty; please set the option data_sets=name1 name2 etc and I will search the fits file for nz_name2, nz_name2, etc.")

    print("Loading number density data from {0}:".format(nz_file))
    if upsampling > 1:
        print("I will up-sample - increase the density of n(z) points by a factor {}".format(upsampling))
        print("to make a spline look more like a histogram. Set upsampling=1")
        print("if you do not want this.")
    F = pyfits.open(nz_file)
    data = {}
    for data_set, data_set_output in zip(data_sets, data_sets_output):
        try:
            name = "NZ_" + data_set.upper() if prefix_extension else data_set.upper()
            print("    Looking at FITS extension {0}:".format(name))
            ext = F[name]
        except KeyError:
            name = "nz_" + data_set.lower() if prefix_extension else data_set.lower()
            print("    Looking at FITS extension {0}:".format(name))
            ext = F[name]

        section = "NZ_" + data_set.upper() if prefix_section else data_set.upper()
        section = section.replace(data_set.upper(), data_set_output.upper()) #Replace original name with new name when writing NZ into the cosmosis datablock/section
        z, nz = load_histogram_form(ext, upsampling)
        data[section] = (z, nz)
    return data


def execute(block, config):
    data_sets = config
    for name, data in list(config.items()):
        z, nz = data
        nbin = len(nz)
        ns = len(z)
        block[name, "nbin"] = nbin
        block[name, "nz"] = ns
        block[name, "z"] = z
        for i, n in enumerate(nz):
            block[name, "bin_{0}".format(i + 1)] = n
    return 0
