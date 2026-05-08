from cosmosis.datablock import option_section, names
import numpy as np
import warnings

def get_nbins(block, section):
    if block.has_value(section, "nbin_a"):
        n_a=block[section,"nbin_a"]
        n_b=block[section,"nbin_b"]
    else:
        n_a=block[section,"nbin"]
        n_b=n_a
    return n_a, n_b


def apply_lcut(block, section, Lmin, Lmax, output_section = "", save_name = ""):
    n_a, n_b = get_nbins(block, section)
    # n_b is 1 for CMB kappa cross-correlations, or n_sources for galaxy-kappa
    # cross-correlations (e.g. galaxy_shear_cl used as the lensing ratio denominator).
    # The ell filter is the same for all bin pairs so we apply it uniformly.

    ell = block[section, 'ell']
    w   = np.ones(ell.shape[0])
    w[ell<Lmin]=0
    w[ell>Lmax+1]=0

    for bini in range(0,n_a):
        for binj in range(0,n_b):
            cl_bin = 'bin_{}_{}'.format(bini+1, binj+1)
            if not block.has_value(section, cl_bin):
                continue
            C_ell_orig = block[section, cl_bin]
            filtered_C_ell = C_ell_orig*w
            if (output_section == ""):
                block[section, cl_bin] = filtered_C_ell
            else:
                block[output_section, cl_bin] = filtered_C_ell

    # If output_section is specified and different from section we need
    # to copy over info from section into output_section
    if  (section != output_section and output_section != ""):
        if (block.has_value(section,"nbin")):
            block[output_section, "nbin"] = block[section, "nbin"]
        if (block.has_value(section,"nbin_a")):
            block[output_section, "nbin_a"] = block[section, "nbin_a"]
        if (block.has_value(section,"nbin_b")):
            block[output_section, "nbin_b"] = block[section, "nbin_b"]
        block[output_section, "save_name"] =  block[section, "save_name"]
        block[output_section, "ell"] = block[section, "ell"]
        block[output_section, "sep_name"] = block[section, "sep_name"]
        block[output_section, "sample_a"] = block[section, "sample_a"]
        block[output_section, "sample_b"] = block[section, "sample_b"]
        block[output_section, "is_auto"] = block[section, "is_auto"]
        if (save_name != ""):
            block[output_section, "save_name"] = save_name

def setup(options):
    shearkappa_section = options.get_string(option_section, "shearkappa_section", default="shear_cmbkappa_cl")
    galkappa_section = options.get_string(option_section, "galkappa_section", default="galaxy_cmbkappa_cl")
    kappakappa_section = options.get_string(option_section, "kappakappa_section", default="cmbkappa_cl")
    shearkappa_output_section = options.get_string(option_section, "shearkappa_output_section", default="")
    galkappa_output_section = options.get_string(option_section, "galkappa_output_section", default="")
    kappakappa_output_section = options.get_string(option_section, "kappakappa_output_section", default="")
    try:
        save_name = options.get_string(option_section, "save_name", default="")
    except:
        save_name = ""

    if options.has_value(option_section,'Lmin'):
        Lmin = options[option_section, 'Lmin']
    else:
        warnings.warn("No Lmin supplied! Setting Lmin=0.")
        Lmin = 0
    if options.has_value(option_section,'Lmax'):
        Lmax = options[option_section, 'Lmax']
    else:
        warnings.warn("No Lmax supplied! Setting Lmax=999999.")
        Lmax = 999999

    return {"shearkappa_section":shearkappa_section, "galkappa_section":galkappa_section, "kappakappa_section":kappakappa_section, \
            "shearkappa_output_section":shearkappa_output_section, "galkappa_output_section":galkappa_output_section, "kappakappa_output_section":kappakappa_output_section,
            "Lmin":Lmin, "Lmax":Lmax, "save_name":save_name}

def execute(block, config):
    shearkappa_section = config['shearkappa_section']
    galkappa_section = config['galkappa_section']
    kappakappa_section = config['kappakappa_section']
    shearkappa_output_section = config['shearkappa_output_section']
    galkappa_output_section = config['galkappa_output_section']
    kappakappa_output_section = config['kappakappa_output_section']
    save_name = config['save_name']

    Lmin = config['Lmin']
    Lmax = config['Lmax']

    if block.has_section(galkappa_section):
        apply_lcut(block, galkappa_section, Lmin, Lmax, output_section = galkappa_output_section, save_name = save_name)

    if block.has_section(shearkappa_section):
        apply_lcut(block, shearkappa_section, Lmin, Lmax,  output_section = shearkappa_output_section, save_name = save_name)

    if block.has_section(kappakappa_section):
        apply_lcut(block, kappakappa_section, Lmin, Lmax, output_section = kappakappa_output_section, save_name = save_name)
        
    return 0

def cleanup(config):
    pass
