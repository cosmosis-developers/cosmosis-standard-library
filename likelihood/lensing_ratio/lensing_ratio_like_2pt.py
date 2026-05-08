"""
Cosmosis likelihood module: lensing ratio full 2pt modeling (Y3-style).

Theory ratios are computed from actual gamma_t / w_g-kappa predictions
in the block (interpolated to the data theta grid), giving sensitivity to
IA, magnification, baryonic effects, etc. – everything that enters the 2pt
functions.

This is the CMB-extended equivalent of the Y3 shear_ratio_likelihood.py.

Reads a pkl file produced by compute_lensing_ratios.ipynb.

The choice of basis (CMB-only vs GGL+CMB) is encoded in the file itself —
pass the appropriate pkl path as data_file.

Cosmosis ini options
--------------------
data_file          : path to the pkl file
ggl_section        : block section for gamma_t theory (default galaxy_shear_xi)
cmb_section        : block section for w_g-kappa_CMB theory
                     (default galaxy_cmbkappa_xi_so)
bin_avg            : T/F  whether theory is already bin-averaged (default F)
"""

import pickle
import numpy as np
from scipy.interpolate import interp1d
from cosmosis.datablock import option_section, names


def setup(options):
    config = {}

    filename = options[option_section, "data_file"]
    with open(filename, "rb") as f:
        config["data"] = pickle.load(f)

    config["ggl_section"] = options.get_string(option_section, "ggl_section", "galaxy_shear_xi")
    config["cmb_section"] = options.get_string(option_section, "cmb_section", "galaxy_cmbkappa_xi_so")
    config["bin_avg"]     = options.get_bool(option_section, "bin_avg", False)

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

    print(f"[lensing_ratio_like_2pt] file: {filename}")
    print(f"[lensing_ratio_like_2pt] GGL={n_ggl}  CMB={n_cmb}  "
          f"total={n_ggl + n_cmb}  cov={config['cov'].shape}")
    return config


def _get_theory_xi(block, section, l, s, theta_data, bin_avg):
    """Return theory correlation interpolated to theta_data (arcmin)."""
    xi = block[section, f"bin_{l}_{s}"]
    if bin_avg:
        return xi  # already at the right points
    theta_theory_rad = block[section, "theta"]
    theta_theory_amin = np.degrees(theta_theory_rad) * 60.0
    return interp1d(theta_theory_amin, xi, bounds_error=False, fill_value=np.nan)(theta_data)


def _weighted_ratio(num, den, inv_cov_r, theta_mask):
    """
    Optimal inverse-variance-weighted scalar ratio:
        r = (num/den)[mask] · (P · 1) / (1^T · P · 1)
    """
    P  = inv_cov_r[np.ix_(theta_mask, theta_mask)]
    P1 = P @ np.ones(theta_mask.sum())
    x  = (num / den)[theta_mask]
    return float(x @ P1 / P1.sum())


def execute(block, config):
    d          = config["data"]
    theta_data = d["theta_data"]          # arcmin, (n_theta,)
    bin_avg    = config["bin_avg"]

    theory_parts = []

    # inv_cov_individual_ratios holds [GGL | CMB] angular inv-covs in order
    inv_cov_all = d["inv_cov_individual_ratios"]
    n_ggl       = d["n_ggl_combos"]

    # ---- GGL ratios --------------------------------------------------------
    if n_ggl > 0:
        ggl_sec   = config["ggl_section"]
        theta_min = d.get("theta_min_ggl")
        theta_max = d.get("theta_max_ggl")

        theory_ggl = []
        for idx, (l, si, sref) in enumerate(d["ggl_combinations"]):
            gt_si   = _get_theory_xi(block, ggl_sec, l, si,   theta_data, bin_avg)
            gt_sref = _get_theory_xi(block, ggl_sec, l, sref, theta_data, bin_avg)
            mask    = _theta_mask(theta_data, theta_min, theta_max, lens_bin=l)
            r = _weighted_ratio(gt_si, gt_sref, inv_cov_all[idx], mask)
            theory_ggl.append(r)

        theory_parts.append(np.array(theory_ggl))

    # ---- CMB ratios --------------------------------------------------------
    if d["n_cmb_combos"] > 0:
        cmb_sec   = config["cmb_section"]
        ggl_sec   = config["ggl_section"]
        theta_min = d.get("theta_min_cmb")
        theta_max = d.get("theta_max_cmb")

        theory_cmb = []
        for idx, (l, sj) in enumerate(d["cmb_combinations"]):
            w_cmb = _get_theory_xi(block, cmb_sec, l, 1,  theta_data, bin_avg)
            g_sj  = _get_theory_xi(block, ggl_sec, l, sj, theta_data, bin_avg)
            mask  = _theta_mask(theta_data, theta_min, theta_max, lens_bin=l)
            r = _weighted_ratio(w_cmb, g_sj, inv_cov_all[n_ggl + idx], mask)
            theory_cmb.append(r)

        theory_parts.append(np.array(theory_cmb))

    theory = np.concatenate(theory_parts)
    delta  = config["data_vec"] - theory
    chi2   = float(delta @ config["inv_cov"] @ delta)

    block[names.likelihoods, "lensing_ratio_2pt_like"] = -0.5 * chi2
    block["data_vector", "lensing_ratio_2pt_chi2"]     = chi2

    return 0


def _theta_mask(theta, theta_min, theta_max, lens_bin=None):
    mask = np.ones(len(theta), dtype=bool)
    if theta_min is not None:
        lo = np.asarray(theta_min).flat[lens_bin - 1] if (
            np.ndim(theta_min) > 0 and lens_bin is not None
        ) else float(theta_min)
        mask &= theta > lo
    if theta_max is not None:
        hi = np.asarray(theta_max).flat[lens_bin - 1] if (
            np.ndim(theta_max) > 0 and lens_bin is not None
        ) else float(theta_max)
        mask &= theta <= hi
    return mask


def cleanup(config):
    pass
