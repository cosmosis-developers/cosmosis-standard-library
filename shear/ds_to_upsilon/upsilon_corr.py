"""
Author: Tianqing Zhang
Last edit: 2023.11.08
"""

import numpy as np
from cosmosis.datablock import names, option_section
from scipy.interpolate import InterpolatedUnivariateSpline as ius


def upsilon_correction(r, r0, ds0_fit, ds0):
    
    return r0**2/r**2 * (ds0_fit - ds0)


def setup(options):
    
    r0 = options.get_double(option_section, "r0", 4.0)
    
    
    return {"r0":r0}


def execute(block, config):
    
    r0 = config['r0']
    
    nbin_lens = block['galaxy_shear_xi', 'nbin_a']
    nbin_source = block['galaxy_shear_xi', 'nbin_b']
    
    rp = block['galaxy_shear_xi', 'rp']
    
    for lens_id in range(1, nbin_lens+1):
        
        delta_sigma_r0 = block['upsilon_parameters', f'ds0_{lens_id}']
        
        for source_id in range(1, nbin_source+1):
            
            f_ds = block.get_double('f_ds', 'bin_{0}_{1}'.format(lens_id,source_id), 1.0)
            
            ds_ij =  block['galaxy_shear_xi', 'bin_{0}_{1}'.format(lens_id,source_id)]
            
            ds0 = ius(rp, ds_ij)(r0)
                        
            block['galaxy_shear_xi', 'bin_{0}_{1}'.format(lens_id,source_id)] += f_ds * upsilon_correction(rp, r0, delta_sigma_r0, ds0)
            
            
            
    return 0
            
      
    
