"""
Core physics for the binned dark-energy background module — pure numpy/astropy,
no cosmosis dependency (so it is unit-testable standalone).

Provides:
  - PiecewiseDECosmology: astropy FLRW subclass with user f_DE(z), w(z)
  - builders for piecewise-constant w(z) [closed-form f_DE] and f_DE(z) [step]
  - Brieden (2022) r_drag, Hu & Sugiyama (1996) z_*
  - build_cosmology / compute_distances helpers used by the cosmosis interface
"""

import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import FLRW, FlatLambdaCDM


# ---------------------------------------------------------------------------
# Custom astropy cosmology
# ---------------------------------------------------------------------------

class PiecewiseDECosmology(FLRW):
    """FLRW with a user-supplied dark-energy density f_DE(z) and (informational) w(z)."""

    def __init__(self, *, H0, Om0, Ode0, de_func, w_func,
                 Tcmb0=2.7255, Neff=3.044, m_nu=0.0, Ob0=None, name=None):
        self._de_func = de_func
        self._w_func = w_func
        super().__init__(H0=H0, Om0=Om0, Ode0=Ode0, Tcmb0=Tcmb0,
                         Neff=Neff, m_nu=m_nu, Ob0=Ob0, name=name)

    def w(self, z):
        return self._w_func(np.asarray(z, dtype=float))

    def de_density_scale(self, z):
        return self._de_func(np.asarray(z, dtype=float))


# ---------------------------------------------------------------------------
# Binned dark-energy density builders
# ---------------------------------------------------------------------------

def _make_fde_from_w(edges, w_vals, w_tail=-1.0):
    """
    Closed-form f_DE(z) for piecewise-constant w (no integration).

    edges  : (n+1,) boundaries [z_0=0, ..., z_n]
    w_vals : (n,)   constant w per bin
    w_tail : w for z >= z_n (default -1, ΛCDM)
    Returns a vectorized callable with f_DE(0) = 1.
    """
    edges = np.asarray(edges, dtype=float)
    w_vals = np.asarray(w_vals, dtype=float)
    n = w_vals.size

    edge_ratios = ((1.0 + edges[1:]) / (1.0 + edges[:-1])) ** (3.0 * (1.0 + w_vals))
    C = np.concatenate([[1.0], np.cumprod(edge_ratios)])  # C[i] = f_DE(z_i)

    def fde(z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, n - 1)
        out = C[idx] * ((1.0 + z) / (1.0 + edges[idx])) ** (3.0 * (1.0 + w_vals[idx]))
        tail = z >= edges[-1]
        if np.any(tail):
            out[tail] = C[n] * ((1.0 + z[tail]) / (1.0 + edges[-1])) ** (3.0 * (1.0 + w_tail))
        return out

    return fde


def _make_w_func(edges, w_vals, w_tail=-1.0):
    """Informational piecewise-constant w(z)."""
    edges = np.asarray(edges, dtype=float)
    w_vals = np.asarray(w_vals, dtype=float)
    n = w_vals.size

    def wfunc(z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, n - 1)
        out = w_vals[idx].astype(float)
        out[z >= edges[-1]] = w_tail
        return out

    return wfunc


def _make_fde_step(edges, amps, low_clamp=1.0, high_clamp=1.0):
    """
    Piecewise-constant density (fde_bins mode).

    edges : (m+1,) boundaries [0, z_min, 0.1, ..., 4.2]
    amps  : (m-1,) f_DE amplitudes for intervals [z_min,..) ... [.., z_n);
            interval 0 ([0, z_min)) clamped to low_clamp (=1); z>=z_n -> high_clamp (=1).
    Returns (fde_callable, w_callable).
    """
    edges = np.asarray(edges, dtype=float)
    amps = np.asarray(amps, dtype=float)
    vals = np.concatenate([[low_clamp], amps])  # per-interval density, length m
    m = vals.size

    def fde(z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, m - 1)
        out = vals[idx]
        out[z >= edges[-1]] = high_clamp
        return out

    def wfunc(z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return np.full(z.shape, -1.0)

    return fde, wfunc


# ---------------------------------------------------------------------------
# Sound horizon and recombination redshift
# ---------------------------------------------------------------------------

def r_drag_brieden(omega_b, omega_bc, neff):
    """
    Sound horizon at baryon drag, Brieden et al. (2022) / paper Eq. 4 [Mpc].
    omega_b = Omega_b h^2, omega_bc = (Omega_m - Omega_nu) h^2.
    (The Omega_b h^2 exponent -0.13 follows the canonical Brieden formula.)
    """
    return (147.05
            * (omega_b / 0.02236) ** (-0.13)
            * (omega_bc / 0.1432) ** (-0.23)
            * (neff / 3.04) ** (-0.1))


def z_star_hu_sugiyama(omega_cb, omega_b):
    """Recombination redshift z_* via Hu & Sugiyama (1996). omega = Omega h^2."""
    g1 = 0.0783 * omega_b ** (-0.238) / (1.0 + 39.5 * omega_b ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * omega_b ** 1.81)
    return 1048.0 * (1.0 + 0.00124 * omega_b ** (-0.738)) * (1.0 + g1 * omega_cb ** g2)


# ---------------------------------------------------------------------------
# Cosmology construction and distances
# ---------------------------------------------------------------------------

def build_cosmology(H0, Om0, Ok0, mnu, n_massive, tcmb, neff, de_func, w_func):
    """
    Build a PiecewiseDECosmology for total matter density Om0 (CosmoSIS omega_m
    convention: CDM + baryons + massive neutrinos).

    astropy's FLRW.Om0 is COLD matter only (CDM + baryons); massive neutrinos are
    tracked separately via m_nu and contribute through Onu(z).  So for mnu>0 we pass
    Om0_cold = Om0 - Onu0 (matching lensing_ratio_like_geom / compressed_cmb_prior).
    For mnu=0, Onu0 is the relativistic (radiation-like) neutrino density and must NOT
    be removed from the matter budget.

    Ode0 = 1 - Om0_cold - Ok0 - Ogamma0 - Onu0  enforces f_DE(0)=1.
    Returns (cosmo, Onu0).  Onu0 is independent of the Om0 passed to the reference.
    """
    m_nu = (np.full(n_massive, mnu / n_massive) * u.eV) if mnu > 0.0 else 0.0 * u.eV
    ref = FlatLambdaCDM(H0=H0, Om0=Om0, Tcmb0=tcmb, Neff=neff, m_nu=m_nu)
    Onu0 = ref.Onu0
    Om0_cold = (Om0 - Onu0) if mnu > 0.0 else Om0
    Ode0 = 1.0 - Om0_cold - Ok0 - ref.Ogamma0 - Onu0
    cosmo = PiecewiseDECosmology(H0=H0, Om0=Om0_cold, Ode0=Ode0, de_func=de_func,
                                 w_func=w_func, Tcmb0=tcmb, Neff=neff, m_nu=m_nu)
    return cosmo, Onu0


def compute_distances(cosmo, z):
    """
    Return dict of distance arrays on grid z (Mpc, plus H in 1/Mpc).
    Handles flat/open/closed via D_C -> D_M.
    """
    D_C = cosmo.comoving_distance(z).to_value(u.Mpc)

    Ok0 = cosmo.Ok0
    if abs(Ok0) < 1e-8:
        D_M = D_C
    else:
        sqrtOk0 = np.sqrt(abs(Ok0))
        dh = cosmo.hubble_distance.to_value(u.Mpc)
        if Ok0 > 0:
            D_M = dh / sqrtOk0 * np.sinh(sqrtOk0 * D_C / dh)
        else:
            D_M = dh / sqrtOk0 * np.sin(sqrtOk0 * D_C / dh)

    D_L = D_M * (1.0 + z)
    D_A = D_M / (1.0 + z)

    mu = np.zeros(z.size)
    mu[0] = -np.inf
    mu[1:] = 5.0 * np.log10(D_L[1:]) + 25.0

    H_z = (cosmo.H(z) / const.c).to_value(1.0 / u.Mpc)
    with np.errstate(invalid="ignore", divide="ignore"):
        D_V = ((1.0 + z) ** 2 * z * D_A ** 2 / H_z) ** (1.0 / 3.0)
    D_V[0] = 0.0

    return {"D_C": D_C, "D_M": D_M, "D_L": D_L, "D_A": D_A, "MU": mu,
            "H": H_z, "D_V": D_V}
