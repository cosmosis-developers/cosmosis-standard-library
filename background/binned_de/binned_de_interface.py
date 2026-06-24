"""
CosmoSIS background module: non-parametric (piecewise-binned) dark energy.

Computes the cosmological background expansion and distances for a dark-energy
component specified in redshift bins, following Kessler et al. (arXiv:2606.05853).
Two parametrizations are supported via the ``mode`` option:

  mode = w_bins   : piecewise-constant equation of state w_DE(z); the density
                    f_DE(z) = rho_DE(z)/rho_DE(0) is obtained in closed form.
  mode = fde_bins : piecewise-constant density f_DE(z) directly (Phase 2).

The module writes the full ``distances`` block (z, a, D_C, D_M, D_A, D_L, D_V,
H, MU, rs_zdrag, and optionally THETASTAR), so any downstream likelihood that
reads distances from the block (desi_dr2, des_sn, thetastar_like,
lensing_ratio_geom, compressed_cmb_prior) runs unchanged.  It replaces CAMB.

The physics lives in binned_de_core.py (pure numpy/astropy, unit-tested
standalone); this file is only the cosmosis block I/O glue.

Cosmosis ini options
--------------------
mode            : 'w_bins' (default) or 'fde_bins'
bin_edges       : space-separated boundaries, e.g. 0.0 0.1 0.4 0.6 0.8 1.1 1.6 4.2
                  (w_bins: n_bins+1 edges; fde_bins: include z_min as the second edge)
n_bins          : number of free bins (default 7)
zmax_background : linear grid max (default 4.0)
nz_background   : linear grid points (default 400)
n_logz          : log-spaced points appended to zmax_logz (default 100)
zmax_logz       : max redshift of log grid (default 1100.0)
tcmb            : CMB temperature K (default 2.7255)
neff            : effective number of relativistic species (default 3.044)
write_thetastar : also write distances/THETASTAR (default True)
rd_mode         : 'fitting' (Brieden 2022, default)
"""

import os
import sys
import numpy as np
from cosmosis.datablock import option_section, names

sys.path.insert(0, os.path.dirname(__file__))
from binned_de_core import (                       # noqa: E402
    _make_fde_from_w, _make_w_func, _make_fde_step,
    r_drag_brieden, z_star_hu_sugiyama,
    build_cosmology, compute_distances, build_grid,
)


def setup(options):
    config = {}
    config["mode"] = options.get_string(option_section, "mode", "w_bins").lower()

    raw_edges = options[option_section, "bin_edges"]
    if isinstance(raw_edges, str):
        raw_edges = raw_edges.replace(",", " ").split()
    config["edges"] = np.atleast_1d(np.asarray(raw_edges, dtype=float))
    config["n_bins"] = options.get_int(option_section, "n_bins", 7)

    config["zmax_background"] = options.get_double(option_section, "zmax_background", 4.0)
    config["nz_background"] = options.get_int(option_section, "nz_background", 400)
    config["n_logz"] = options.get_int(option_section, "n_logz", 400)
    config["zmax_logz"] = options.get_double(option_section, "zmax_logz", 1100.0)

    config["tcmb"] = options.get_double(option_section, "tcmb", 2.7255)
    config["neff"] = options.get_double(option_section, "neff", 3.044)
    config["write_thetastar"] = options.get_bool(option_section, "write_thetastar", True)
    config["rd_mode"] = options.get_string(option_section, "rd_mode", "fitting").lower()

    if config["mode"] not in ("w_bins", "fde_bins"):
        raise ValueError(f"[binned_de] unknown mode '{config['mode']}'")
    if config["rd_mode"] != "fitting":
        raise ValueError(f"[binned_de] unsupported rd_mode '{config['rd_mode']}' (only 'fitting')")

    config["z"] = build_grid(config["edges"], config["zmax_background"],
                             config["nz_background"], config["n_logz"],
                             config["zmax_logz"])

    print(f"[binned_de] mode={config['mode']}  n_bins={config['n_bins']}  edges={config['edges']}")
    print(f"[binned_de] grid: {z.size} points, z_max={z[-1]:.1f}, "
          f"rd_mode={config['rd_mode']}, write_thetastar={config['write_thetastar']}")
    return config


def execute(block, config):
    cp = names.cosmological_parameters

    h0 = block[cp, "h0"]
    om0 = block[cp, "omega_m"]
    ok0 = block[cp, "omega_k"] if block.has_value(cp, "omega_k") else 0.0
    omega_b_phys = block[cp, "omega_b"] * h0 ** 2
    mnu = block[cp, "mnu"] if block.has_value(cp, "mnu") else 0.0
    n_massive = (int(round(block[cp, "num_massive_neutrinos"]))
                 if block.has_value(cp, "num_massive_neutrinos") else 3)

    edges = config["edges"]
    n_bins = config["n_bins"]
    bin_vals = np.array([block[cp, f"de_bin_{i+1}"] for i in range(n_bins)], dtype=float)

    if config["mode"] == "w_bins":
        de_func = _make_fde_from_w(edges, bin_vals, w_tail=-1.0)
        w_func = _make_w_func(edges, bin_vals, w_tail=-1.0)
    else:
        de_func, w_func = _make_fde_step(edges, bin_vals)

    cosmo, Onu0 = build_cosmology(100.0 * h0, om0, ok0, mnu, n_massive,
                                  config["tcmb"], config["neff"], de_func, w_func)

    z = config["z"]
    dist = compute_distances(cosmo, z)

    # Cold matter (CDM + baryons) physical density. For mnu>0 remove the massive
    # neutrino density; for mnu=0 all of omega_m is cold (Onu0 is radiation-like).
    omega_bc = (om0 - Onu0 if mnu > 0.0 else om0) * h0 ** 2
    r_drag = r_drag_brieden(omega_b_phys, omega_bc, config["neff"])

    d = names.distances
    block[d, "nz"] = z.size
    block[d, "z"] = z
    block[d, "a"] = 1.0 / (1.0 + z)
    for key in ("D_C", "D_M", "D_L", "D_A", "D_V", "MU", "H"):
        block[d, key] = dist[key]
    block[d, "rs_zdrag"] = r_drag
    for nm in ("D_C", "D_M", "D_L", "D_A", "D_V"):
        block.put_metadata(d, nm, "unit", "Mpc")
    block.put_metadata(d, "H", "unit", "1.0/Mpc")
    block.put_metadata(d, "rs_zdrag", "unit", "Mpc")

    if config["write_thetastar"]:
        z_star = z_star_hu_sugiyama(omega_bc, omega_b_phys)
        r_star = r_drag / 1.02                       # paper Sec. II relation
        D_M_star = float(np.interp(z_star, z, dist["D_M"]))
        block[d, "THETASTAR"] = 100.0 * r_star / D_M_star

    return 0


def cleanup(config):
    pass
