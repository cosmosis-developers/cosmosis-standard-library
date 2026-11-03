from cosmosis.datablock import option_section, names
from scipy.interpolate import interp2d
import numpy as np

def setup(options):
    config=0
    return config

def execute(block, config):
    test = config

    P0 = block['matter_power_nl','p_k']
    k0 = block['matter_power_nl','k_h']
    z0 = block['matter_power_nl','z']

    P1 = block['matter_power_lin','p_k']
    k1 = block['matter_power_lin','k_h']
    z1 = block['matter_power_lin','z']

    I = interp2d(np.log10(k0),z0,np.log10(P0))

    P_nl_interp = 10**I(np.log10(k1),z1)


    block.replace_double_array_1d('matter_power_nl','k_h',k1)
    block.replace_double_array_1d('matter_power_nl','z',z1)
    block.replace_double_array_nd('matter_power_nl','p_k',P_nl_interp)

    #import pdb ; pdb.set_trace()

    return 0
