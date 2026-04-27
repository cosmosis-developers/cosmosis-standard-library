"""
Cosmosis module: strong CMB prior (Das & Spergel 2011 style).

Given fixed Omega_m h^2, Omega_b h^2, and the CMB acoustic angle theta_A
(all from Planck), plus sampled w and omega_k, numerically solves for Om0
such that theta_A(Om0, w, omega_k) matches the Planck target.  Then derives
h0 = sqrt(Omh2 / Om0) and writes omega_m, h0 to the block.

This is a deterministic constraint: Om0 and h0 are not sampled but derived
at every likelihood call.  The effective free parameters are only w and omega_k.

Pipeline ordering
-----------------
Must run BEFORE lensing_ratio_like_geom (which reads omega_m and h0):
    modules = fits_nz strong_cmb_prior lensing_ratio_like_geom

Options
-------
omega_m_h2  : Omega_m h^2 from Planck (default 0.14283, Planck 2015)
omega_b_h2  : Omega_b h^2 from Planck (default 0.02226, Planck 2015)
theta_A     : acoustic angle r_s/((1+z_cmb)*d_A(z_cmb)) (default 0.010409)
m_nu        : total neutrino mass in eV, enters r_s only (default 0.06)
z_cmb       : CMB source redshift (default 1089.0)
"""

import numpy as np
from scipy import optimize as op
from astropy.cosmology import wCDM
from astropy import units as u
from cosmosis.datablock import option_section, names

_T_CMB = 2.725  # K


def _get_rs(omh2, obh2, m_nu):
    """Sound horizon in comoving Mpc (Eisenstein & Hu fitting formula)."""
    omeganu  = 0.0107 * m_nu
    omegacb  = omh2 - omeganu
    return (55.154 * np.exp(-72.3 * (omeganu + 0.0006)**2)
            / (omegacb**0.25351 * obh2**0.12807))


def _residual(om0_arr, w0, ok0, z_cmb, omh2, rs, theta_A_target, m_nu):
    om0 = om0_arr[0]
    if om0 <= 0.0:
        return 1e10
    h    = np.sqrt(omh2 / om0)
    ode0 = 1.0 - ok0 - om0
    try:
        cosmo = wCDM(H0=h * 100.0, Om0=om0, Ode0=ode0, w0=w0,
                     m_nu=m_nu / 3.0 * u.eV, Tcmb0=_T_CMB * u.K)
        dA = cosmo.angular_diameter_distance(z_cmb).value
    except Exception:
        return 1e10
    if dA <= 0.0 or np.isnan(dA):
        return 1e10
    theta_test = rs / ((1.0 + z_cmb) * dA)
    return 1e6 * (theta_test - theta_A_target)**2 / theta_A_target**2


def setup(options):
    omh2    = options.get_double(option_section, "omega_m_h2", 0.14283)
    obh2    = options.get_double(option_section, "omega_b_h2", 0.02226)
    theta_A = options.get_double(option_section, "theta_A",    0.010409)
    m_nu    = options.get_double(option_section, "m_nu",       0.06)
    z_cmb   = options.get_double(option_section, "z_cmb",      1089.0)

    rs = _get_rs(omh2, obh2, m_nu)
    print(f"[strong_cmb_prior] Omh2={omh2:.5f}  Obh2={obh2:.5f}  "
          f"theta_A={theta_A:.6f}  r_s={rs:.2f} Mpc")

    return {"omh2": omh2, "theta_A": theta_A, "m_nu": m_nu, "z_cmb": z_cmb, "rs": rs}


def execute(block, config):
    w0  = block[names.cosmological_parameters, "w"]
    ok0 = block[names.cosmological_parameters, "omega_k"]

    result = op.minimize(
        _residual,
        np.array([0.3]),
        args=(w0, ok0, config["z_cmb"], config["omh2"],
              config["rs"], config["theta_A"], config["m_nu"]),
        method="Nelder-Mead",
    )
    om0 = result.x[0]

    if not result.success or om0 <= 0.0:
        return 1

    h0   = np.sqrt(config["omh2"] / om0)
    ode0 = 1.0 - ok0 - om0

    if h0 <= 0.1 or h0 >= 2.0 or ode0 < -1.0:
        return 1

    block[names.cosmological_parameters, "omega_m"] = float(om0)
    block[names.cosmological_parameters, "h0"]      = float(h0)
    return 0


def cleanup(config):
    pass
