try:
    import act_dr6_cmbonly
except ImportError:
    raise RuntimeError('The act_dr6_cmbonly python module is required for the act_dr6_lite likelihood. It can be obtained from https://github.com/ACTCollaboration/DR6-ACT-lite')
from cosmosis.datablock import names
cosmo = names.cosmological_parameters
import numpy as np
import os


cal_params = [
    "A_act",
    "P_act"
]

dirname = os.path.split(__file__)[0]


def setup(options):
    act = act_dr6_cmbonly.ACTDR6CMBonly(packages_path=dirname)
    return act


def execute(block, config):
    act = config

    cl_dict = {
        "tt": np.append(np.array([0,0]), block[names.cmb_cl, 'tt']),
        "te": np.append(np.array([0,0]), block[names.cmb_cl, 'te']),
        "ee": np.append(np.array([0,0]), block[names.cmb_cl, 'ee']),
    }

    nuisance = {}

    for p in cal_params:
        nuisance[p] = block["act_params", p]
    
    loglike = act.loglike(cl_dict, **nuisance)

    # Then call the act code
    block[names.likelihoods, 'act_dr6_lite_like'] = loglike

    return 0