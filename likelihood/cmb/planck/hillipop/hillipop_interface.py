from cosmosis.datablock import option_section
import numpy as np
try:
    from planck_2020_hillipop import TT, TE, EE, TTTEEE
except ImportError:
    raise ImportError("Please install the planck_2020_hillipop with: pip install planck-2020-hillipop")


nuisance_parameters = [
    "Aradio",
    "beta_radio",
    "Adusty",
    "AdustT",
    "beta_dusty",
    "beta_dustT",
    "AsyncT",
    "Acib",
    "beta_cib",
    "Atsz",
    "Aksz",
    "xi",
    "AdustP",
    "beta_dustP",
    "AsyncP",
    "A_planck",
    "cal100A",
    "cal100B",
    "cal143A",
    "cal143B",
    "cal217A",
    "cal217B",
    "pe100A",
    "pe100B",
    "pe143A",
    "pe143B",
    "pe217A",
    "pe217B",
]

def setup(options):
    raise RuntimeError("We are not quite sure if the Hillipop and Lollipop interfaces are correctly set up yet. You can remove this line to help test.")
    mode = options.get_string(option_section, "mode", default="TTTEEE").upper()
    if mode == "TT":
        calculator = TT()
        print("Using Hillipop TT likelihood")
    elif mode == "TE":
        calculator = TE()
        print("Using Hillipop TE likelihood")
    elif mode == "EE":
        calculator = EE()
        print("Using Hillipop EE likelihood")
    elif mode == "TTTEEE":
        calculator = TTTEEE()
        print("Using Hillipop TTTEEE likelihood")
    else:
        raise ValueError(f"Unknown Hillipop mode {mode}. Choose from TT, TE, EE, TTTEEE")

    return calculator

def make_cl_start_at_zero(cl, ell):
    if ell[0] == 0:
        return
    ellmin = ell[0]
    new_zero_bit = np.zeros(int(ellmin))
    for key in list(cl.keys()):
        cl[key] = np.concatenate([new_zero_bit, cl[key]])

def execute(block, config):
    calculator = config

    cl = {}
    cl["tt"] = block['cmb_cl', 'tt']
    cl["te"] = block['cmb_cl', 'te']
    cl["ee"] = block['cmb_cl', 'ee']
    cl["bb"] = block['cmb_cl', 'bb']

    ell = block['cmb_cl', 'ell']
    make_cl_start_at_zero(cl, ell)


    nuisance = {}
    for param in nuisance_parameters:
        nuisance[param] = block['planck', param]

    if 'beta_dusty' not in nuisance:
        nuisance['beta_dusty'] = nuisance['beta_cib']



    block["likelihoods", "hillipop_like"] = calculator.loglike(cl, **nuisance)


    return 0
