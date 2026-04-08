from cosmosis.datablock import option_section, names
import numpy as np

def get_nbins(block, section):
    if block.has_value(section, "nbin_a"):
        n_a=block[section,"nbin_a"]
        n_b=block[section,"nbin_b"]
    else:
        n_a=block[section,"nbin"]
        n_b=n_a
    return n_a, n_b
          
def apply_beam(block, section, beam_sigma, is_auto=False, output_section = None, save_name = None):
    n_a, n_b = get_nbins(block, section)
    # n_b is 1 for CMB kappa cross-correlations, or n_sources for galaxy-kappa
    # cross-correlations (e.g. galaxy_shear_cl used as the lensing ratio denominator).
    # The beam is the same for all bin pairs so we apply it uniformly.

    ell = block[section, 'ell']
    B_ell = np.exp(-0.5*ell*(ell+1.)*beam_sigma**2.)
    for bini in range(0,n_a):
        for binj in range(0,n_b):
            cl_bin = 'bin_{}_{}'.format(bini+1, binj+1)
            if not block.has_value(section, cl_bin):
                continue
            C_ell_orig = block[section, cl_bin]
            if (is_auto):
                beamed_C_ell = C_ell_orig*(B_ell**2.)
            else:
                beamed_C_ell = C_ell_orig*B_ell
            if (output_section is None):
                block[section, cl_bin] = beamed_C_ell
            else:
                block[output_section, cl_bin] = beamed_C_ell

    if  (section != output_section and (not output_section is None)):
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
        if (save_name is not None):
            block[output_section, "save_name"] = save_name
  
def setup(options):
    shearkappa_section = options.get_string(option_section, "shearkappa_section", default="shear_cmbkappa_cl")
    galkappa_section = options.get_string(option_section, "galkappa_section", default="galaxy_cmbkappa_cl")
    kappakappa_section = options.get_string(option_section, "kappakappa_section", default="cmbkappa_cl")
    shearkappa_output_section = options.get_string(option_section, "shearkappa_output_section", default=None)
    galkappa_output_section = options.get_string(option_section, "galkappa_output_section", default=None)
    kappakappa_output_section = options.get_string(option_section, "kappakappa_output_section", default=None)
    save_name = options.get_string(option_section, "save_name", default=None)

    if options.has_value(option_section, 'beam_sigma_arcmin'):
        beam_sigma_arcmin = options[option_section, "beam_sigma_arcmin"]
        # radians
        beam_sigma = np.radians(beam_sigma_arcmin/60.)   
    elif options.has_value(option_section, 'beam_fwhm_arcmin'):
        beam_fwhm_arcmin = options[option_section, "beam_fwhm_arcmin"]
        # radians
        beam_sigma = (1./np.sqrt(8.*np.log(2.)))*np.radians(beam_fwhm_arcmin/60.)
    else:
        raise ValueError("No beam_sigma_arcmin or beam_fwhm_armcin supplied!")


    return {"shearkappa_section":shearkappa_section, "galkappa_section":galkappa_section, "kappakappa_section":kappakappa_section,\
            "shearkappa_output_section":shearkappa_output_section, "galkappa_output_section":galkappa_output_section, "kappakappa_output_section":kappakappa_output_section,\
            "beam_sigma":beam_sigma, "save_name":save_name}

def execute(block, config):
    shearkappa_section = config['shearkappa_section']
    galkappa_section = config['galkappa_section']
    kappakappa_section = config['kappakappa_section']
    shearkappa_output_section = config['shearkappa_output_section']
    galkappa_output_section = config['galkappa_output_section']
    kappakappa_output_section = config['kappakappa_output_section']
    beam_sigma = config['beam_sigma']
    save_name = config['save_name']

    if block.has_section(galkappa_section):
        apply_beam(block, galkappa_section, beam_sigma, is_auto = False, output_section  = galkappa_output_section, save_name = save_name)

    if block.has_section(shearkappa_section):
        apply_beam(block, shearkappa_section, beam_sigma, is_auto = False, output_section = shearkappa_output_section, save_name = save_name)

    if block.has_section(kappakappa_section):
        apply_beam(block, kappakappa_section, beam_sigma, is_auto = True, output_section = kappakappa_output_section, save_name = save_name)
        
    return 0

def cleanup(config):
    pass
