from cosmosis import option_section
from cosmosis.datablock import names
import numpy as np
import camb

def get_rd_from_camb(h, omega_m, ombh2, N_eff):
    """
    Use CAMB to compute the sound horizon at recombination.
    """
    omega_b = ombh2 / h**2
    omega_c = omega_m - omega_b
    omch2 = omega_c * h**2
    cp = camb.set_params(ombh2=ombh2, omch2=omch2, H0=h*100, nnu=N_eff)
    res = camb.get_background(cp)
    return res.get_derived_params()['rdrag']


def setup(options):
    return {
    }

def execute(block, config):
    h_fid = block[names.cosmological_parameters, "h0"]
    rdh_sample = block[names.cosmological_parameters, "rdh"]
    rd_prime = block[names.distances, "rs_zdrag"]

    rd_sample = rdh_sample / h_fid
    f = rd_prime / rd_sample


    block[names.cosmological_parameters, "h0"] /= f
    block[names.distances, "D_L"] *=  f
    block[names.distances, "D_A"] *= f
    block[names.distances, "D_V"] *= f
    block[names.distances, "D_M"] *= f
    block[names.distances, "D_C"] *= f
    block[names.distances, "H"] /= f
    block[names.distances, "mu"] += 5.0 * np.log10(f)

    block[names.distances, "rs_zdrag"] = rd_sample

    # This is the "final" value of H0*rd using the h that is consistent
    # with the r_d and Omega_m values.
    block["distances", "h0rd"] = block["cosmological_parameters", "h0"] * block[names.distances, "rs_zdrag"]

    return 0