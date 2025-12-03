import numpy as np
from cosmosis.datablock import option_section, names


def setup(options):
    return None


def execute(block):
    # inputs
    pk_lin = block["matter_power_lin", "p_k"]
    pk_nonlin = block["matter_power_nl", "p_k"]
    A_mod = block["amod_parameter", "a_mod"]

    # for now assume pk_lin and pk_nonlin are defined on the same grids
    z_same = np.allclose(block["matter_power_lin", "z"], block["matter_power_nl", "z"])
    k_same = np.allclose(block["matter_power_lin", "k_h"], block["matter_power_nl", "k_h"])
    if not (z_same and k_same):
        raise ValueError("The z and k sampling of the linear and non-linear matter power spectra must be the same for the amod module")

    # calculate new non-linear power
    pk_m = pk_lin + A_mod * (pk_nonlin - pk_lin)
    block["matter_power_nl", "p_k"] = pk_m

    return 0


def cleanup(config):
    pass
