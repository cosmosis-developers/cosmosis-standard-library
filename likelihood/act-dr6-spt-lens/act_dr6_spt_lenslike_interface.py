try:
    import act_dr6_spt_lenslike
except ImportError:
    raise RuntimeError('The act_dr6_spt_lenslike python module is required for the act_dr6_spt_lenslike likelihood. This can be obtained from https://github.com/qujia7/spt_act_likelihood/.')
from cosmosis.datablock import option_section, names
cosmo = names.cosmological_parameters
import numpy as np
import os

dirname = os.path.split(__file__)[0]

def setup(options):
    variant = options.get_string(option_section, 'variant', default='actplanckspt3g_extended')
    lens_only = options.get_bool(option_section, 'lens_only', default=True)
    like_corrections = options.get_bool(option_section, 'like_corrections', default=False)
    like_only = options.get_bool(option_section, 'like_only', default=False)

    data_directory = os.path.join(dirname, 'data/v1.2/')
    data_directory = options.get_string(option_section, 'data_directory', default=data_directory)

    if not os.path.exists(data_directory):
        raise FileNotFoundError('Required data file not found at {}.\nPlease obtain it and place it correctly.\nThe script get-act-data.sh will download and place it.'.format(data_directory))

    # lmax = options.get_int(option_section, 'lmax', default=4000)
    mock = options.get_bool(option_section, 'mock', default=False)
    nsims_act = options.get_int(option_section, 'nsims_act', default=792) # Number of sims used for covmat; used in Hartlap correction
    nsims_planck = options.get_int(option_section, 'nsims_planck', default=400) # Number of sims used for covmat; used in Hartlap correction
    # no_like_corrections = options.get_bool(option_section, 'no_like_corrections', default=False)
    # Any ells above this will be discarded; likelihood must at least request ells up to this
    trim_lmax = options.get_int(option_section, 'trim_lmax', default=2998)
    apply_hartlap = options.get_bool(option_section, 'apply_hartlap', default=True)
    # Limber integral parameters
    # limber = options.get_bool(option_section, 'limber', default=False)
    # nz = options.get_int(option_section, 'nz', default=100)
    # kmax = options.get_int(option_section, 'kmax', default=10)
    scale_cov = options.get_string(option_section, 'scale_cov', default='')
    if not scale_cov:
        scale_cov = None
    else:
        scale_cov = float(scale_cov)
    varying_cmb_alens = options.get_bool(option_section, 'varying_cmb_alens', default=False) # Whether to divide the theory spectrum by Alens

    if varying_cmb_alens and not block.has_value(cosmo, 'A_lens'):
        raise RuntimeError('You have specified varying_cmb_alens: True to vary A_lens in the CMB lensing spectra, but given no A_lens value in the parameter file.')

    # This dict will now have entries like `data_binned_clkk` (binned data vector), `cov`
    # (covariance matrix) and `binmat_act` (binning matrix to be applied to a theory
    # curve starting at ell=0).
    # variant,lens_only=lens_only,like_corrections=like_corrections)
    data_dict = act_dr6_spt_lenslike.load_data(ddir=data_directory,
                                           variant=variant,lens_only=lens_only,
                                           like_corrections=like_corrections,apply_hartlap=apply_hartlap,
                                           mock=mock,nsims_act=nsims_act,nsims_planck=nsims_planck,
                                           trim_lmax=trim_lmax,scale_cov=scale_cov)

    data_dict['cosmosis_like_only'] = like_only
    data_dict['trim_lmax'] = trim_lmax
    data_dict['varying_cmb_alens'] = varying_cmb_alens
    # data_dict['limber'] = limber
    return data_dict

# def get_limber_clkk():
#     raise NotImplementedError("Direct Limber calcution of Clkk not implemented in cosmosis version of likelihood")

def execute(block, config):
    data_dict = config

    # These are the CMB lensing convergence spectra (not potential or deflection)
    # as well as the TT, EE, TE, BB CMB spectra (needed for likelihood corrections)
    # in uK^2 units. All of these are C_ell (not D_ell), no ell or 2pi factors.
    ell = block[names.cmb_cl, 'ell']

    if ell.max() < data_dict['trim_lmax']:
        raise ValueError(f"An lmax of at least {data_dict['trim_lmax']} is required.")

    f1 = ell * (ell + 1) / (2 * np.pi)
    cl_tt = block[names.cmb_cl, 'tt'] / f1
    cl_ee = block[names.cmb_cl, 'ee'] / f1
    cl_te = block[names.cmb_cl, 'te'] / f1
    cl_bb = block[names.cmb_cl, 'bb'] / f1

    # Slightly different normalization here
    cl_pp = block[names.cmb_cl, 'pp'] / f1

    if data_dict['varying_cmb_alens']:
        cl_pp /= block[cosmo, "A_lens"]

    # if data_dict['limber']:
    #     cl_kk = get_limber_clkk()
    # else:
    #     cl_kk = act_dr6_lenslike.pp_to_kk(cl_pp, ell)
    cl_kk = act_dr6_spt_lenslike.pp_to_kk(cl_pp, ell)

    # from matplotlib import pyplot as plt
    # plt.ion()
    # import pdb; pdb.set_trace()

    # Then call the act code
    lnlike, bclkk = act_dr6_spt_lenslike.generic_lnlike(data_dict,ell, cl_kk, ell, cl_tt, cl_ee, cl_te, cl_bb, data_dict['trim_lmax'],
                                                        # do_norm_corr=data_dict['act_cmb_rescale'],
                                                        # act_calib=data_dict['act_calib'],
                                                        # no_actlike_cmb_corrections=data_dict['no_actlike_cmb_corrections'],
                                                        return_theory=True)
    block[names.likelihoods, 'act_dr6_spt_lens_like'] = lnlike

    # from matplotlib import pyplot as plt
    # plt.ion()
    # import pdb; pdb.set_trace()  

    if not data_dict['cosmosis_like_only']:
        block[names.data_vector, 'act_dr6_lens_theory'] = bclkk
        block[names.data_vector, 'act_dr6_lens_data'] = data_dict['data_binned_clkk']
        block[names.data_vector, 'act_dr6_lens_covariance'] = data_dict['cov']
        block[names.data_vector, 'act_dr6_lens_inverse_covariance'] = data_dict['cinv']


    return 0