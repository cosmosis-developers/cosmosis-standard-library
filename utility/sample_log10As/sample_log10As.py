#########################################
# Module by David Sanchez Cid
# (david.sanchez@ciemat.es)
# THIS IS A MODULE TO BE USED IN HSC
# RE-ANALYSIS PROJECT IN DESC
# SAMPLE log_10{10^10 As} instead of As
#########################################

from cosmosis.datablock import option_section, names

import numpy as np

def setup(options):
    # Name of the original variable
    # As_name = options.get_string(option_section, "As_name",default="A_s")
    log10As_name = options.get_string(option_section, 'log10As_name',default="log10As")
    return log10As_name

def execute(block,config):
    # As_name = config
    log10As_name = config
    
    # Extract As value
    # As = block[names.cosmological_parameters, As_name]
    
    # Extract log10As value
    log10As = block[names.cosmological_parameters, log10As_name]
    
    # Transform to HSC sampled quantity
    # log10As = np.log10(10**10 * As)
    As = 10 ** log10As / 10 ** 10
    
    # Transform from HSC sampled quantity to As
    print('Input log10As', log10As)
    print('Output As', As)
    
    block[names.cosmological_parameters,'A_s'] = As
    
    return(0)

def cleanup(config):
    pass