"""
Author: Tianqing Zhang
Last edited: Oct 28th 2023
Module to calculate various distances from redshifts
"""

import sys
import os
from cosmosis.datablock import names, option_section
import numpy as np
import astropy.cosmology

def setup(options):
    
    config = {}
    
    config['zmin_background'] = options.get_double(option_section, 'zmin_background', default=0.0)
    config['zmax_background'] = options.get_double(option_section, 'zmax_background', default=4.0)
    config['nz_background'] = options.get_int(option_section, 'nz_background', default=300)
    
    return config

def execute(block, config):
    
        
    # Get cosmological parameters
    pars = names.cosmological_parameters
    # print(block.keys(section=pars))
    ombh2      = block[pars, 'ombh2']
    h          = block[pars, "h0"]
    #log1e10As  = block[pars, "log1e10As"]
    omch2      = block[pars, "omch2"]
    ns         = block[pars, "n_s"]
    w0         = block[pars, "w"]
    mnu   = block[pars, "mnu"]
    omnuh2     = 0.00064 * (mnu/0.06)
    Omm        = (ombh2 + omch2 + ombh2)/h**2
    Omk   = block[pars, "omega_k"]
    Omde       = 1.0-Omm-Omk
    wa    = block[pars, "wa"]
    
    z_background = np.linspace(config["zmin_background"], config["zmax_background"],config["nz_background"])
    
    block[names.distances, "nz"] = len(z_background)
    block[names.distances, "z"] = z_background
    block[names.distances, "a"] = 1/(z_background+1)

    c = astropy.cosmology.w0waCDM(h*100, Omm, Omde, Ob0=ombh2/h**2, w0=w0, wa=wa)
    D_C = c.comoving_distance(z_background).value # Mpc
    H = c.H(z_background).value * 1e3 / 299792458.0
    D_H = 1 / H
    
    if Omk == 0:
        D_M = D_C
    elif Omk < 0:
        s = np.sqrt(-Omk)
        D_M = (D_H / s)  * np.sin(s * D_C / D_H)
    else:
        s = np.sqrt(Omk)
        D_M = (D_H / s) * np.sinh(s * D_C / D_H)

    D_L = D_M * (1 + z_background)
    D_A = D_M / (1 + z_background)
    D_V = ((1 + z_background)**2 * z_background * D_A**2 / H)**(1./3.)
    
    block[names.distances, "D_C"] = D_C # Note that this is in unit of Mpc, not in Mpc/h which is usually used in weak lensing analysis
    block[names.distances, "D_M"] = D_M
    block[names.distances, "D_L"] = D_L
    block[names.distances, "D_A"] = D_A
    block[names.distances, "D_V"] = D_V
    block[names.distances, "H"] = H
    
    
    return 0