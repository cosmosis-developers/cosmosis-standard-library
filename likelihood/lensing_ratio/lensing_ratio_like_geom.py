"""
Cosmosis likelihood module: lensing ratio geometry-only prediction.

Theory ratios are computed purely from effective Sigma_crit^{-1} integrals
over the lens and source n(z)s in the block, so the only sensitivity is
to photo-z shifts and cosmological distances.

Reads a pkl file produced by compute_lensing_ratios.ipynb containing:
  measured_ratios, cmb_measured_ratios  (scalar ratio vectors)
  ggl_combinations, cmb_combinations    (index lists)
  ratio_cov                             (joint covariance, [GGL|CMB] ordering)
  n_ggl_combos, n_cmb_combos           (sizes)

Distance computation
--------------------
Distances are read directly from the ``distances`` section of the data block
(written by CAMB or any background module).  This makes the module fully
model-agnostic: any dark energy model handled by the upstream background
module is automatically supported with no changes here.

Required block entries (written by CAMB):
  distances/z   — redshift grid
  distances/d_m — comoving transverse distance D_M(0→z) [Mpc]

The grid must extend to at least z_cmb (default 1089.80) so that the CMB
source plane is covered.  Set ``zmax_background >= 1200`` in the [camb] ini
section.  The module raises a clear error if the requirement is not met.

CAMB must run before this module in every pipeline.

Neutrino mass handling
----------------------
Handled automatically: CAMB integrates H(z) with the correct neutrino
physics and writes the resulting distances to the block.  No special
treatment is needed here.

Cosmosis ini options
--------------------
ratio_file        : path to the pkl file
z_cmb             : CMB last-scattering redshift (default 1089.80)
lens_nz_section   : block section for lens n(z)  (default nz_lens)
source_nz_section : block section for source n(z) (default nz_source)
"""

import pickle
import numpy as np
from cosmosis.datablock import option_section, names


# ---------------------------------------------------------------------------
# Physical constant
# ---------------------------------------------------------------------------

# 4πG/c² in Mpc/Msun — prefactor for Σ_crit^{-1} = (4πG/c²) × D_l [Mpc] × β
_FOUR_PI_G_OVER_C2 = (4.0 * np.pi * 6.674e-11      # G [m³ kg⁻¹ s⁻²]
                      / 2.99792458e8**2              # / c² [m² s⁻²] → [m/kg]
                      * 1.98892e30 / 3.08568e22)     # × (kg/Msun)/(m/Mpc) → [Mpc/Msun]


# ---------------------------------------------------------------------------
# n(z) helpers
# ---------------------------------------------------------------------------

def _normalize_nz(z, nz):
    return nz / np.trapz(nz, z)


def _normalize_nz_collection(z, nzs):
    nzs = np.atleast_2d(np.asarray(nzs, dtype=float))
    return np.vstack([_normalize_nz(z, row) for row in nzs])


# ---------------------------------------------------------------------------
# Distance helper — block-based, curvature-correct
# ---------------------------------------------------------------------------

def _da_z1z2_from_block(z1, z2, z_grid, dm_grid, ok, h0):
    """
    Angular diameter distance D_A(z1→z2) in Mpc from tabulated D_M(0→z).

    Handles flat, open, and closed cosmologies.  D_M is the comoving
    transverse distance as written by CAMB to ``distances/d_m``.

    Algorithm
    ---------
    1. Interpolate D_M at z1, z2 from the block grid.
    2. Invert to radial comoving distance D_C (which is additive):
         flat:   D_C = D_M
         open:   D_C = (D_H/K) arcsinh(K D_M),  K = sqrt(Ωk) / D_H
         closed: D_C = (D_H/K) arcsin(K D_M),   K = sqrt(|Ωk|) / D_H
    3. Subtract:  D_C(z1→z2) = D_C(z2) − D_C(z1)
    4. Re-apply curvature to get D_M(z1→z2).
    5. D_A(z1→z2) = D_M(z1→z2) / (1+z2),  clipped to zero when z2 ≤ z1.
    """
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)

    dh = 2997.92458 / h0          # c/H0 [Mpc]

    dm1 = np.interp(z1, z_grid, dm_grid)
    dm2 = np.interp(z2, z_grid, dm_grid)

    if abs(ok) < 1e-6:            # flat
        dc1, dc2 = dm1, dm2
    elif ok > 0:                   # open
        K   = np.sqrt(ok) / dh
        dc1 = np.arcsinh(dm1 * K) / K
        dc2 = np.arcsinh(dm2 * K) / K
    else:                          # closed
        K   = np.sqrt(-ok) / dh
        dc1 = np.arcsin(np.clip(dm1 * K, -1.0, 1.0)) / K
        dc2 = np.arcsin(np.clip(dm2 * K, -1.0, 1.0)) / K

    dc12 = dc2 - dc1              # additive radial separation

    if abs(ok) < 1e-6:
        dm12 = dc12
    elif ok > 0:
        K    = np.sqrt(ok) / dh
        dm12 = np.sinh(K * dc12) / K
    else:
        K    = np.sqrt(-ok) / dh
        dm12 = np.sin(np.clip(K * dc12, -np.pi / 2.0, np.pi / 2.0)) / K

    return np.maximum(dm12, 0.0) / (1.0 + z2)


# ---------------------------------------------------------------------------
# Sigma_crit geometry
# ---------------------------------------------------------------------------

def _sigma_crit_inverse_grid(d_l, d_s, d_ls):
    """
    Sigma_crit^{-1} on a (n_l_z, n_s_z) grid [Mpc^2/Msun].

    d_l  : (n_l_z,)         D_A(0→z_l) [Mpc]
    d_s  : (n_s_z,)         D_A(0→z_s) [Mpc]
    d_ls : (n_l_z, n_s_z)  D_A(z_l→z_s) [Mpc]
    """
    d_s_g = d_s[np.newaxis, :]
    d_l_g = d_l[:, np.newaxis]
    beta = np.where(
        d_s_g > 0.0,
        np.maximum(d_ls, 0.0) / np.where(d_s_g > 0.0, d_s_g, 1.0),
        0.0,
    )
    return _FOUR_PI_G_OVER_C2 * d_l_g * beta


def _sigma_crit_eff_grid(z_l, lens_nzs, z_s, source_nzs, d_l, d_s, d_ls):
    """Effective Sigma_crit^{-1} for every (lens bin, source bin) pair [Mpc^2/Msun]."""
    lens_nzs   = _normalize_nz_collection(z_l, lens_nzs)
    source_nzs = _normalize_nz_collection(z_s, source_nzs)
    kernel = _sigma_crit_inverse_grid(d_l, d_s, d_ls)    # (n_l_z, n_s_z)

    out = np.zeros((lens_nzs.shape[0], source_nzs.shape[0]))
    for i, n_l in enumerate(lens_nzs):
        wk = n_l[:, np.newaxis] * kernel
        for j, n_s in enumerate(source_nzs):
            out[i, j] = np.trapz(np.trapz(wk * n_s[np.newaxis, :], z_s, axis=1), z_l)
    return out


def _sigma_crit_eff_cmb(z_l, lens_nzs, d_l, d_cmb, d_l_cmb):
    """
    Effective Sigma_crit^{-1} for each lens bin to the CMB source plane [Mpc^2/Msun].

    d_cmb   : scalar   D_A(0→z_cmb) [Mpc]
    d_l_cmb : (n_l_z,) D_A(z_l→z_cmb) [Mpc]
    """
    lens_nzs = _normalize_nz_collection(z_l, lens_nzs)
    beta     = np.where(d_cmb > 0.0, np.maximum(d_l_cmb, 0.0) / d_cmb, 0.0)
    kernel   = _FOUR_PI_G_OVER_C2 * d_l * beta
    return np.array([np.trapz(n_l * kernel, z_l) for n_l in lens_nzs])


# ---------------------------------------------------------------------------
# CosmoSIS interface
# ---------------------------------------------------------------------------

def setup(options):
    config = {}

    filename = options[option_section, "ratio_file"]
    with open(filename, "rb") as f:
        config["data"] = pickle.load(f)

    config["z_cmb"]          = options.get_double(option_section, "z_cmb", 1089.80)
    config["lens_section"]   = options.get_string(option_section, "lens_nz_section",   "nz_lens")
    config["source_section"] = options.get_string(option_section, "source_nz_section", "nz_source")

    d     = config["data"]
    n_ggl = d["n_ggl_combos"]
    n_cmb = d["n_cmb_combos"]

    data_parts = []
    if n_ggl > 0:
        data_parts.append(d["measured_ratios"])
    if n_cmb > 0:
        data_parts.append(d["cmb_measured_ratios"])
    config["data_vec"] = np.concatenate(data_parts)
    config["cov"]      = d["ratio_cov"]
    config["inv_cov"]  = np.linalg.inv(config["cov"])

    print(f"[lensing_ratio_like_geom] file: {filename}")
    print(f"[lensing_ratio_like_geom] GGL={n_ggl}  CMB={n_cmb}  "
          f"total={n_ggl + n_cmb}  cov={config['cov'].shape}")
    return config


def execute(block, config):
    d = config["data"]

    h0  = block[names.cosmological_parameters, "h0"]
    ok0 = (block[names.cosmological_parameters, "omega_k"]
           if block.has_value(names.cosmological_parameters, "omega_k") else 0.0)
    z_cmb = config["z_cmb"]

    # ---- Block distances (required) -----------------------------------------
    if not block.has_value("distances", "z"):
        raise RuntimeError(
            "[lensing_ratio_like_geom] 'distances/z' not found in block. "
            "CAMB must run before this module. Add 'camb' to modules= before "
            "lensing_ratio_geom_like in your ini file."
        )
    z_grid  = block["distances", "z"]
    dm_grid = block["distances", "d_m"]
    if float(z_grid[-1]) < z_cmb:
        raise RuntimeError(
            f"[lensing_ratio_like_geom] block distance grid only reaches "
            f"z={z_grid[-1]:.1f} but z_cmb={z_cmb:.2f} is required. "
            f"Set zmax_background >= 1200 in the [camb] ini section."
        )

    # ---- n(z) ---------------------------------------------------------------
    z_l = block[config["lens_section"],   "z"]
    z_s = block[config["source_section"], "z"]
    n_lens   = d["nbin_lens"]
    n_source = d["nbin_source"]
    lens_nzs   = np.vstack([block[config["lens_section"],   f"bin_{i+1}"] for i in range(n_lens)])
    source_nzs = np.vstack([block[config["source_section"], f"bin_{i+1}"] for i in range(n_source)])

    # ---- Distances from block -----------------------------------------------
    d_l = np.interp(z_l, z_grid, dm_grid) / (1.0 + z_l)
    d_s = np.interp(z_s, z_grid, dm_grid) / (1.0 + z_s)

    z_l_rep  = np.repeat(z_l, z_s.size)
    z_s_tile = np.tile(z_s, z_l.size)
    d_ls = _da_z1z2_from_block(
        z_l_rep, z_s_tile, z_grid, dm_grid, ok0, h0
    ).reshape(z_l.size, z_s.size)

    d_cmb   = float(np.interp(z_cmb, z_grid, dm_grid)) / (1.0 + z_cmb)
    d_l_cmb = _da_z1z2_from_block(
        z_l, np.full(z_l.size, z_cmb), z_grid, dm_grid, ok0, h0
    )

    # ---- Sigma_crit effective integrals -------------------------------------
    sig_inv_grid = _sigma_crit_eff_grid(z_l, lens_nzs, z_s, source_nzs, d_l, d_s, d_ls)
    sig_inv_cmb  = _sigma_crit_eff_cmb(z_l, lens_nzs, d_l, d_cmb, d_l_cmb)

    # ---- Theory ratios ------------------------------------------------------
    theory_parts = []

    if d["n_ggl_combos"] > 0:
        theory_parts.append(np.array([
            sig_inv_grid[l - 1, si - 1] / sig_inv_grid[l - 1, sref - 1]
            for l, si, sref in d["ggl_combinations"]
        ]))

    if d["n_cmb_combos"] > 0:
        theory_parts.append(np.array([
            sig_inv_cmb[l - 1] / sig_inv_grid[l - 1, sj - 1]
            for l, sj in d["cmb_combinations"]
        ]))

    theory = np.concatenate(theory_parts)
    delta  = config["data_vec"] - theory
    chi2   = float(delta @ config["inv_cov"] @ delta)

    block[names.likelihoods, "lensing_ratio_geom_like"] = -0.5 * chi2
    block["data_vector", "lensing_ratio_geom_chi2"]     = chi2

    return 0


def cleanup(config):
    pass
