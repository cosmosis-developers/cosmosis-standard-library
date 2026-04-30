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

The choice of basis (CMB-only vs GGL+CMB) is encoded in the file itself —
pass the appropriate pkl path as data_file.

Cosmosis ini options
--------------------
data_file         : path to the pkl file
z_cmb             : CMB last-scattering redshift (default 1089.80, Planck 2018 z*)
lens_nz_section   : block section for lens n(z)  (default nz_lens)
source_nz_section : block section for source n(z) (default nz_source)
"""

import pickle
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import wCDM
from cosmosis.datablock import option_section, names


# ---------------------------------------------------------------------------
# Geometry helpers (self-contained, no external project dependency)
# ---------------------------------------------------------------------------

def _normalize_nz(z, nz):
    area = np.trapz(nz, z)
    return nz / area


def _normalize_nz_collection(z, nzs):
    nzs = np.atleast_2d(np.asarray(nzs, dtype=float))
    return np.vstack([_normalize_nz(z, row) for row in nzs])


def _sigma_crit_inverse_grid(z_l, z_s, cosmo):
    """
    Return Sigma_crit^{-1}(z_l, z_s) on a (n_lens, n_source) grid [Mpc^2/Msun].
    Uses angular diameter distances — correct for curved cosmology.
    """
    d_l = cosmo.angular_diameter_distance(z_l).to_value(u.Mpc)    # D_A(z_l), (n_l,)
    d_s = cosmo.angular_diameter_distance(z_s).to_value(u.Mpc)    # D_A(z_s), (n_s,)

    # D_A(z_l, z_s) on a (n_l, n_s) grid — vectorised via flat arrays
    z_l_rep = np.repeat(z_l, z_s.size)
    z_s_tile = np.tile(z_s, z_l.size)
    d_ls = cosmo.angular_diameter_distance_z1z2(z_l_rep, z_s_tile).to_value(u.Mpc)
    d_ls = d_ls.reshape(z_l.size, z_s.size)                       # (n_l, n_s)

    d_s_g = d_s[np.newaxis, :]
    d_l_g = d_l[:, np.newaxis]

    # beta = D_A(z_l, z_s) / D_A(z_s); zero when z_s <= z_l
    beta = np.where(d_s_g > 0.0, np.maximum(d_ls, 0.0) / np.where(d_s_g > 0.0, d_s_g, 1.0), 0.0)

    prefactor = 4.0 * np.pi * const.G / const.c**2
    return (prefactor * (d_l_g * u.Mpc) * beta).to(u.Mpc**2 / u.Msun).value


def _sigma_crit_eff_grid(z_l, lens_nzs, z_s, source_nzs, cosmo):
    """
    Effective Sigma_crit^{-1} for every (lens bin, source bin) pair [Mpc^2/Msun].
    lens_nzs   : (n_lens, len(z_l))
    source_nzs : (n_source, len(z_s))
    """
    lens_nzs   = _normalize_nz_collection(z_l, lens_nzs)
    source_nzs = _normalize_nz_collection(z_s, source_nzs)
    kernel = _sigma_crit_inverse_grid(z_l, z_s, cosmo)   # (n_lens_z, n_source_z)

    out = np.zeros((lens_nzs.shape[0], source_nzs.shape[0]))
    for i, n_l in enumerate(lens_nzs):
        wk = n_l[:, np.newaxis] * kernel
        for j, n_s in enumerate(source_nzs):
            out[i, j] = np.trapz(np.trapz(wk * n_s[np.newaxis, :], z_s, axis=1), z_l)
    return out


def _sigma_crit_eff_cmb(z_l, lens_nzs, z_cmb, cosmo):
    """
    Effective Sigma_crit^{-1} for each lens bin to the CMB source plane [Mpc^2/Msun].
    """
    lens_nzs = _normalize_nz_collection(z_l, lens_nzs)
    kernel   = _sigma_crit_inverse_grid(z_l, np.array([z_cmb]), cosmo)[:, 0]  # (n_lens_z,)

    out = np.zeros(lens_nzs.shape[0])
    for i, n_l in enumerate(lens_nzs):
        out[i] = np.trapz(n_l * kernel, z_l)
    return out


# ---------------------------------------------------------------------------
# Cosmosis interface
# ---------------------------------------------------------------------------

def setup(options):
    config = {}

    filename = options[option_section, "data_file"]
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
    om0 = block[names.cosmological_parameters, "omega_m"]
    w   = block[names.cosmological_parameters, "w"]       if block.has_value(names.cosmological_parameters, "w")       else -1.0
    ok0 = block[names.cosmological_parameters, "omega_k"] if block.has_value(names.cosmological_parameters, "omega_k") else 0.0
    ode0 = 1.0 - om0 - ok0
    cosmo = wCDM(H0=h0 * 100.0, Om0=om0, Ode0=ode0, w0=w, Tcmb0=2.725)

    n_lens   = d["nbin_lens"]
    n_source = d["nbin_source"]

    z_l = block[config["lens_section"],   "z"]
    z_s = block[config["source_section"], "z"]
    lens_nzs   = np.vstack([block[config["lens_section"],   f"bin_{i+1}"] for i in range(n_lens)])
    source_nzs = np.vstack([block[config["source_section"], f"bin_{i+1}"] for i in range(n_source)])

    sig_inv_grid = _sigma_crit_eff_grid(z_l, lens_nzs, z_s, source_nzs, cosmo)  # (n_lens, n_source)
    sig_inv_cmb  = _sigma_crit_eff_cmb(z_l, lens_nzs, config["z_cmb"], cosmo)   # (n_lens,)

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
