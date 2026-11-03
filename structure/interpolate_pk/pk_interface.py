"""
Interpolate the non-linear matter power spectrum onto the k and z grid of
the linear matter power spectrum.

This is useful when an emulator (e.g. the cosmic emulator) gives the non-linear
power on a different, usually more limited, grid than the one used for the
linear power. The interpolation is linear in log10(k) and log10(P). Points
outside the original grid take the value of the nearest edge.
"""
from cosmosis.datablock import names
import numpy as np
from scipy.interpolate import RegularGridInterpolator

nl = names.matter_power_nl
lin = names.matter_power_lin


def setup(options):
    return None


def execute(block, config):
    z0, k0, P0 = block.get_grid(nl, "z", "k_h", "P_k")
    z1, k1 = block[lin, "z"], block[lin, "k_h"]

    # RegularGridInterpolator needs ascending coordinates
    z_order = np.argsort(z0)
    k_order = np.argsort(k0)
    z0, k0, P0 = z0[z_order], k0[k_order], P0[np.ix_(z_order, k_order)]

    interpolator = RegularGridInterpolator((z0, np.log10(k0)), np.log10(P0), method="linear")

    # Clip the new points to the old range so that we use the nearest edge value
    # for points outside it, rather than failing.
    zz, kk = np.meshgrid(np.clip(z1, z0[0], z0[-1]),
                         np.clip(np.log10(k1), np.log10(k0[0]), np.log10(k0[-1])),
                         indexing="ij")
    P_interp = 10 ** interpolator((zz, kk))

    block.replace_grid(nl, "z", z1, "k_h", k1, "P_k", P_interp)

    return 0


def cleanup(config):
    pass
