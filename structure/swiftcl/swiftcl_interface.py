import swiftcl_vendor
from cosmosis import option_section
import numpy as np
import os
dirname = os.path.dirname(__file__) 
pid = os.getpid()


def setup(options):
    ell_min = options.get_double(option_section, "ell_min", 10)
    ell_max = options.get_double(option_section, "ell_max", 3000)
    n_ell = options.get_int(option_section, "n_ell", 100)
    jit = options.get_bool(option_section, "jit", False)


    ell = np.geomspace(ell_min, ell_max, n_ell)
    cache_dir = os.path.join(dirname, "data" + str(pid))
    return ell, cache_dir, jit

def execute(block, config):
    ell, cache_dir, jit = config

    z = block["nz_source", "z"]
    bin1 = block["nz_source", "bin1"]
    bin2 = block["nz_source", "bin2"]

    n1 = np.vstack([z, bin1]).T
    n2 = np.vstack([z, bin2]).T

    z0 = [z.min(), z.max()]
    z1 = [z.min(), z.max()]

    computer = swiftcl_vendor.ClComp(l=ell, path=cache_dir, jit=jit, z0=z0, z1=z1)

    z_distance = block["distances", "z"]
    d_m = block["distances", "d_m"]
    chis = np.interp(z, z_distance, d_m)

    

    computer.C_l(
    n1=n1,
    n2=n2,
    chis1=chis,
    chis2=chis,
    D1=D_k,
    D2=D_k,
    P=P,
    H1=H,
    H2=H,
    H0=cosmo_ccl["h"] * 100,
    O_m=cosmo_ccl["Omega_c"] + cosmo_ccl["Omega_b"],
)