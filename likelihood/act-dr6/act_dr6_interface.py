try:
    import act_dr6_mflike
    import mflike
except ImportError:
    raise RuntimeError('The act_dr6_mflike python module is required for the act_dr6_like likelihood. Try running: pip install act_dr6_mflike.')
from cosmosis.datablock import names
cosmo = names.cosmological_parameters
import numpy as np
import os


foreground_params = [
    "T_effd",
    "beta_d",
    "beta_s",
    "alpha_s",
    "a_tSZ",
    "a_kSZ",
    "a_p",
    "beta_p",
    "a_c",
    "beta_c",
    "a_s",
    "T_d",
    "a_gtt",
    "xi",
    "alpha_dT",
    "alpha_p",
    "alpha_tSZ",
    "a_gte",
    "a_pste",
    "alpha_dE",
    "a_gee",
    "a_psee",
]

cal_params = [
    "calG_all",
    "calE_dr6_pa5_f090",
    "calE_dr6_pa5_f150",
    "calE_dr6_pa6_f090",
    "calE_dr6_pa6_f150",
    "cal_dr6_pa4_f220",
    "cal_dr6_pa5_f090",
    "cal_dr6_pa5_f150",
    "cal_dr6_pa6_f090",
    "cal_dr6_pa6_f150",
]

dirname = os.path.split(__file__)[0]


def setup(options):
    act = act_dr6_mflike.ACTDR6MFLike(packages_path=dirname)
    fg = mflike.BandpowerForeground(act.get_fg_requirements())
    return act, fg


def execute(block, config):
    act, fg = config

    cl_dict = {
        "tt": np.append(np.array([0,0]), block[names.cmb_cl, 'tt']),
        "te": np.append(np.array([0,0]), block[names.cmb_cl, 'te']),
        "ee": np.append(np.array([0,0]), block[names.cmb_cl, 'ee']),
    }

    nuisance = {}

    for p in foreground_params:
        nuisance[p] = block["cmb_foreground_params", p]
    for p in cal_params:
        nuisance[p] = block["act_params", p]
    

    foreground_model = fg.get_foreground_model_totals(**nuisance)
    loglike = act.loglike(cl_dict, foreground_model, **nuisance)

    # Then call the act code
    block[names.likelihoods, 'act_dr6_like'] = loglike

    return 0
