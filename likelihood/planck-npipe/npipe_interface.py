from cosmosis.datablock import option_section
try:
    from planckpr4lensing import PlanckPR4LensingMarged, PlanckPR4Lensing
except ImportError:
    raise ImportError("Please install the planckpr4lensing with: pip install git+https://github.com/carronj/planck_PR4_lensing")
import numpy as np



def setup(options):
    marged = options.get_bool(option_section, "use_marginalized", default=True)
    if marged:
        calculator = PlanckPR4LensingMarged()
        print("Using primary CMB-marginalized PR4 likelihood. TT, EE, TE, BB, will not be used")
    else:
        calculator = PlanckPR4Lensing()
        print("NOT using primary CMB-marginalized PR4 likelihood. TT, EE, TE, BB, will be used")
    return calculator, marged



def execute(block, config):
    calculator, marged = config
    ell = block['cmb_cl', 'ell']
    A = block['planck', 'a_planck']

    # Convert from D_ell to the PP pre-factor, ell**2 (ell+1)**2 / 2pi
    pp = block['cmb_cl', 'pp'] * ell * (ell + 1.)
    cl = {"pp":pp}

    # If we are not using the marginalized version, we need to provide the full set of Cls
    # If we are using the marginalized version these are pre-marginalized over
    if not marged:
        cl["tt"] = block['cmb_cl', 'tt']
        cl["te"] = block['cmb_cl', 'te']
        cl["ee"] = block['cmb_cl', 'ee']
        cl["bb"] = block['cmb_cl', 'bb']


    # npipe wants to start at zero
    if ell[0] == 2:
        ell = np.concatenate([[0, 1], ell])
        for key in cl:
            cl[key] = np.concatenate([[0.0, 0.0], cl[key]])

    block["likelihoods", "npipe_like"] = calculator.log_likelihood(cl, A_planck=A)


    return 0
