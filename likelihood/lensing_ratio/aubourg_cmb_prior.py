"""
CosmoSIS module: compressed CMB Gaussian likelihood (Aubourg+15 style).

Implements a Gaussian prior on the three compressed CMB observables

    v = (omega_b,  omega_cb,  D_M(z_CMB) / r_d)

where D_M is the comoving transverse distance to the CMB last-scattering
surface and r_d is the sound horizon at the baryon drag epoch computed via
the Eisenstein & Hu (1998) fitting formula.

The default mean vector and covariance use Planck 2018
(TT,TE,EE+lowE+lensing, arXiv:1807.06209 Table 2), adapted to the
Aubourg+15 (arXiv:1411.1074) §2.3 compressed-likelihood framework.

Unlike strong_cmb_prior.py this module adds a SOFT Gaussian likelihood term
and does NOT solve for or overwrite omega_m / h0.  Those must be sampled free
parameters in the chain (4-D: omega_m, h0, w, omega_k).

Required block parameters (all must be sampled)
------------------------------------------------
  cosmological_parameters/omega_m
  cosmological_parameters/h0
  cosmological_parameters/w        (optional, default -1)
  cosmological_parameters/omega_k  (optional, default  0)

Pipeline ordering
-----------------
Does not depend on any other module output; can run first:
    modules = fits_nz aubourg_cmb_prior lensing_ratio_like_geom

CosmoSIS ini options
--------------------
omega_b_h2  : fixed omega_b = Omega_b h^2; enters r_d and the data vector
              (default 0.02237, Planck 2018)
omega_nu    : neutrino physical density omega_nu = Sum(m_nu)/(94.07 eV)
              (default 0.000638, corresponding to Sum(m_nu)=0.06 eV)
z_cmb       : CMB last-scattering redshift (default 1089.80, Planck 2018 z*)
dm_over_rd  : override for the fiducial D_M(z_CMB)/r_d mean
              (default 94.28, Planck 2018 estimate)
"""

import numpy as np
from astropy import units as u
from astropy.cosmology import wCDM
from cosmosis.datablock import option_section, names


# ---------------------------------------------------------------------------
# Planck 2018 compressed CMB defaults
# Reference: arXiv:1807.06209 Table 2 (TT,TE,EE+lowE+lensing)
# with Aubourg+15 (arXiv:1411.1074) correlation structure.
# ---------------------------------------------------------------------------

# Mean vector:  mu = (omega_b, omega_cb, D_M(z_*)/r_d)
#   omega_b  = 0.02237 ± 0.00015            (Planck 2018 Table 2)
#   omega_cb = Omega_m h^2 = 0.14301 ± 0.00096  (Planck 2018 Table 2)
#   D_M/r_d  ≈ 94.39   (computed from Planck 2018 flat LCDM best-fit via
#                        astropy wCDM + EH r_d; overrideable via dm_over_rd option)
_PLANCK18_MU_DEFAULT = np.array([0.02237, 0.14301, 94.39])

# 1-sigma uncertainties (Planck 2018 level)
_SIGMA = np.array([0.00015, 0.00096, 0.20])

# Correlation matrix following Aubourg+15 §2.3 sign structure,
# scaled to Planck 2018 precision:
#   rho(omega_b, omega_cb)   =  0.55  (positive: baryons boost both)
#   rho(omega_b, DM/rd)      = -0.35  (more baryons -> smaller r_d -> larger ratio)
#   rho(omega_cb, DM/rd)     = -0.21  (more matter -> faster expansion -> smaller D_M)
_CORR = np.array([
    [ 1.00,  0.55, -0.35],
    [ 0.55,  1.00, -0.21],
    [-0.35, -0.21,  1.00],
])
_PLANCK18_COV = np.outer(_SIGMA, _SIGMA) * _CORR


# ---------------------------------------------------------------------------
# EH sound horizon
# ---------------------------------------------------------------------------

def _eisenstein_hu_rd(omega_cb, omega_b, omega_nu=0.000638):
    """
    Comoving sound horizon at baryon drag epoch (Mpc).

    Fitting formula from Eisenstein & Hu (1998) / Aubourg+15 eq. A1.
    Default omega_nu corresponds to Sum(m_nu) = 0.06 eV.
    """
    return (
        55.154
        * np.exp(-72.3 * (omega_nu + 0.0006) ** 2)
        / (omega_cb ** 0.25351 * omega_b ** 0.12807)
    )


# ---------------------------------------------------------------------------
# CosmoSIS interface
# ---------------------------------------------------------------------------

def setup(options):
    config = {}
    config["omega_b"]  = options.get_double(option_section, "omega_b_h2", 0.02237)
    config["omega_nu"] = options.get_double(option_section, "omega_nu",   0.000638)
    config["z_cmb"]    = options.get_double(option_section, "z_cmb",      1089.80)

    mu = _PLANCK18_MU_DEFAULT.copy()
    dm_over_rd_override = options.get_double(option_section, "dm_over_rd", -1.0)
    if dm_over_rd_override > 0.0:
        mu[2] = dm_over_rd_override

    config["mean"]    = mu
    config["inv_cov"] = np.linalg.inv(_PLANCK18_COV)

    r_d_fid = _eisenstein_hu_rd(mu[1], mu[0], config["omega_nu"])
    print(f"[aubourg_cmb_prior] Planck 2018 mean: "
          f"omega_b={mu[0]:.5f}  omega_cb={mu[1]:.5f}  DM/rd={mu[2]:.3f}")
    print(f"[aubourg_cmb_prior] EH r_d at fiducial mean = {r_d_fid:.2f} Mpc")
    print(f"[aubourg_cmb_prior] 1-sigma: {_SIGMA}")
    return config


def execute(block, config):
    om0 = block[names.cosmological_parameters, "omega_m"]
    h0  = block[names.cosmological_parameters, "h0"]
    w   = (block[names.cosmological_parameters, "w"]
           if block.has_value(names.cosmological_parameters, "w") else -1.0)
    ok0 = (block[names.cosmological_parameters, "omega_k"]
           if block.has_value(names.cosmological_parameters, "omega_k") else 0.0)

    omega_b  = config["omega_b"]
    omega_cb = om0 * h0 ** 2

    r_d = _eisenstein_hu_rd(omega_cb, omega_b, config["omega_nu"])

    ode0  = 1.0 - om0 - ok0
    cosmo = wCDM(H0=h0 * 100.0, Om0=om0, Ode0=ode0, w0=w, Tcmb0=2.725)
    D_M   = cosmo.comoving_transverse_distance(config["z_cmb"]).to_value(u.Mpc)

    vec   = np.array([omega_b, omega_cb, D_M / r_d])
    delta = vec - config["mean"]
    chi2  = float(delta @ config["inv_cov"] @ delta)

    block[names.likelihoods, "aubourg_cmb_prior"]  = -0.5 * chi2
    block["data_vector", "aubourg_cmb_chi2"]        = chi2
    return 0


def cleanup(config):
    pass
