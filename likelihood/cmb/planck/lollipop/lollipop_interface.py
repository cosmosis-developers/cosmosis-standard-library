from cosmosis.datablock import option_section
import numpy as np
try:
    from planck_2020_lollipop import lowlB, lowlE, lowlEB
except ImportError:
    raise ImportError("Please install the planck_2020_lollipop with: pip install planck-2020-lollipop")


nuisance_parameters = [
    "A_planck",
]

def setup(options):
    raise RuntimeError("We are not quite sure if the Hillipop and Lollipop interfaces are correctly set up yet. You can remove this line to help test.")
    mode = options.get_string(option_section, "mode", default="TTTEEE").lower()
    if mode == "lowle":
        calculator = lowlE()
        print("Using Lollipop lowlE likelihood")
    elif mode == "lowlb":
        calculator = lowlB()
        print("Using Lollipop lowlB likelihood")
    elif mode == "lowleb":
        calculator = lowlEB()
        print("Using Lollipop lowlEB likelihood")
    else:
        raise ValueError(f"Unknown Lollipop mode {mode}. Choose from lowlE, lowlB, lowlEB")

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
    cl["ee"] = block['cmb_cl', 'ee']
    cl["bb"] = block['cmb_cl', 'bb']

    ell = block['cmb_cl', 'ell']
    make_cl_start_at_zero(cl, ell)


    nuisance = {}
    for param in nuisance_parameters:
        nuisance[param] = block['planck', param]

    block["likelihoods", "lollipop_like"] = calculator.loglike(cl, **nuisance)


    return 0
