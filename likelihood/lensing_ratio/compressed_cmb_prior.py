"""
CosmoSIS module: compressed CMB Gaussian likelihood (Lemos & Lewis 2023 style).

Implements a Gaussian prior on the three compressed early-Universe CMB observables

    v = (theta_*, omega_b, omega_bc)

where theta_* = r_* / D_M(z_*) is the angular acoustic scale at recombination,
r_* is the comoving sound horizon at recombination, and D_M(z_*) is the
comoving transverse distance to the last-scattering surface.

The mean vector and covariance are from the CamSpec CMB likelihood compression
by Lemos & Lewis (2023) [arXiv:2302.12911, PRD 107, 103505], as quoted in
DESI DR2 BAO cosmology (arXiv:2503.14738, Appendix, Eqs. 18-19).

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
    modules = fits_nz compressed_cmb_prior lensing_ratio_like_geom

CosmoSIS ini options
--------------------
omega_b_h2    : fixed physical baryon density omega_b = Omega_b h^2
                (default 0.02223, from the Lemos & Lewis / DESI DR2 compression mean)
mu_theta_star : override prior mean for theta_*   (default: 0.01041, real Planck/DESI DR2)
mu_omega_b    : override prior mean for omega_b   (default: 0.02223)
mu_omega_bc   : override prior mean for omega_bc  (default: 0.14208)

For simulated/forecast chains set these to the values at the fiducial cosmology to
avoid tension between the prior and simulated BAO/SN data vectors. The covariance
(Planck measurement precision) is kept unchanged in both cases.
"""

import numpy as np
from scipy.integrate import quad
from astropy import units as u
from astropy.cosmology import w0waCDM
from cosmosis.datablock import option_section, names


# ---------------------------------------------------------------------------
# Compressed CMB defaults
# Source: Lemos & Lewis (2023), arXiv:2302.12911
#         as quoted in DESI DR2 (arXiv:2503.14738, Appendix, Eqs. 18-19)
# ---------------------------------------------------------------------------

# Mean vector:  mu = (theta_*, omega_b, omega_bc)
#   theta_* = 0.01041  (i.e. 100 theta_* = 1.0410)
#   omega_b = 0.02223
#   omega_bc = 0.14208
_MU = np.array([0.01041, 0.02223, 0.14208])

# Covariance (x 10^-9):
_COV = 1e-9 * np.array([
    [ 0.006621,   0.12444,  -1.1929],
    [ 0.12444,   21.344,   -94.001],
    [-1.1929,   -94.001,  1488.4 ],
])


# ---------------------------------------------------------------------------
# Sound horizon at recombination
# ---------------------------------------------------------------------------

# Photon physical density for T_CMB = 2.7255 K
_OMEGA_GAMMA = 2.47282e-5

# Speed of light (km/s)
_C_KM_S = 2.998e5


def _z_star(omega_cb, omega_b):
    """
    Redshift of recombination via Hu & Sugiyama (1996) fitting formula.
    Inputs are physical densities (omega = Omega * h^2).
    """
    g1 = 0.0783 * omega_b**(-0.238) / (1.0 + 39.5 * omega_b**0.763)
    g2 = 0.560 / (1.0 + 21.1 * omega_b**1.81)
    return 1048.0 * (1.0 + 0.00124 * omega_b**(-0.738)) * (1.0 + g1 * omega_cb**g2)


def _compute_r_star(omega_cb, omega_b, N_eff=3.044):
    """
    Comoving sound horizon at recombination r_* (Mpc).

    Numerical integration of c_s(z) / H(z) from z_* to z_max, exploiting
    that dark energy is completely negligible at z > 100:

        H(z) = 100 km/s/Mpc * sqrt(omega_cb*(1+z)^3 + omega_r*(1+z)^4)
        c_s(z) = c / sqrt(3 * (1 + R(z))),  R(z) = 3*omega_b / (4*omega_gamma*(1+z))

    Returns (r_star_Mpc, z_star).
    """
    omega_r = _OMEGA_GAMMA * (1.0 + 0.2271 * N_eff)
    z_rec = _z_star(omega_cb, omega_b)

    def integrand(z):
        R   = 3.0 * omega_b / (4.0 * _OMEGA_GAMMA * (1.0 + z))
        c_s = _C_KM_S / np.sqrt(3.0 * (1.0 + R))
        H_z = 100.0 * np.sqrt(omega_cb * (1.0 + z)**3 + omega_r * (1.0 + z)**4)
        return c_s / H_z

    r_s, _ = quad(integrand, z_rec, 1e6, limit=200, epsrel=1e-6)
    return r_s, z_rec


# ---------------------------------------------------------------------------
# CosmoSIS interface
# ---------------------------------------------------------------------------

def setup(options):
    config = {}
    config["omega_b"] = options.get_double(option_section, "omega_b_h2", 0.02223)

    # Allow per-chain override of the prior mean for simulated/forecast chains.
    # Defaults are the real Lemos & Lewis (2023) / DESI DR2 values.
    mu = _MU.copy()
    mu[0] = options.get_double(option_section, "mu_theta_star", _MU[0])
    mu[1] = options.get_double(option_section, "mu_omega_b",    _MU[1])
    mu[2] = options.get_double(option_section, "mu_omega_bc",   _MU[2])
    config["mean"]    = mu
    config["inv_cov"] = np.linalg.inv(_COV)

    source = "fiducial override" if not np.allclose(mu, _MU) else "Lemos & Lewis (2023) / DESI DR2"
    r_s_fid, z_rec_fid = _compute_r_star(mu[2], mu[1])
    print(f"[compressed_cmb_prior] mean ({source}): "
          f"theta_*={mu[0]:.7f}  omega_b={mu[1]:.7f}  omega_bc={mu[2]:.7f}")
    print(f"[compressed_cmb_prior] z_* = {z_rec_fid:.1f},  r_* = {r_s_fid:.2f} Mpc")
    return config


def execute(block, config):
    om0 = block[names.cosmological_parameters, "omega_m"]
    h0  = block[names.cosmological_parameters, "h0"]
    w   = (block[names.cosmological_parameters, "w"]
           if block.has_value(names.cosmological_parameters, "w") else -1.0)
    wa  = (block[names.cosmological_parameters, "wa"]
           if block.has_value(names.cosmological_parameters, "wa") else 0.0)
    ok0 = (block[names.cosmological_parameters, "omega_k"]
           if block.has_value(names.cosmological_parameters, "omega_k") else 0.0)

    if block.has_value(names.cosmological_parameters, "omega_b"):
        # omega_b is being sampled; block stores Omega_b (capital), convert to physical density
        omega_b = block[names.cosmological_parameters, "omega_b"] * h0**2
    else:
        omega_b = config["omega_b"]
    omega_bc = om0 * h0**2

    r_s, z_rec = _compute_r_star(omega_bc, omega_b)

    ode0  = 1.0 - om0 - ok0
    cosmo = w0waCDM(H0=h0 * 100.0, Om0=om0, Ode0=ode0, w0=w, wa=wa, Tcmb0=2.725)
    D_M   = cosmo.comoving_transverse_distance(z_rec).to_value(u.Mpc)

    theta_star = r_s / D_M

    vec   = np.array([theta_star, omega_b, omega_bc])
    delta = vec - config["mean"]
    chi2  = float(delta @ config["inv_cov"] @ delta)

    block[names.likelihoods, "compressed_cmb_prior_like"] = -0.5 * chi2
    block["data_vector", "compressed_cmb_chi2"]           = chi2
    return 0


def cleanup(config):
    pass
