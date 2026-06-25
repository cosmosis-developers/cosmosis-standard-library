"""
Offline unit tests for binned_de_interface: the PiecewiseDECosmology subclass and
the closed-form f_DE builder must reproduce standard astropy cosmologies.

Run:  /opt/anaconda3/bin/python3 background/binned_de/test_binned_de.py
"""
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, wCDM

from binned_de_core import (
    PiecewiseDECosmology, _make_fde_from_w, _make_w_func, _make_fde_step,
    build_grid, build_cosmology, compute_distances,
)

EDGES = np.array([0.0, 0.1, 0.4, 0.6, 0.8, 1.1, 1.6, 4.2])
N = 7
H0, OM0, TCMB, NEFF = 67.27, 0.3156, 2.7255, 3.044
Z_TEST = np.array([0.1, 0.5, 1.0, 2.0, 1089.8])


def _build(w_vals, w_tail, ok0=0.0, m_nu=0.0 * u.eV, mode="w"):
    """Build PiecewiseDECosmology the way the module's execute() does."""
    ref = FlatLambdaCDM(H0=H0, Om0=OM0, Tcmb0=TCMB, Neff=NEFF, m_nu=m_nu)
    Ode0 = 1.0 - OM0 - ok0 - ref.Ogamma0 - ref.Onu0
    if mode == "w":
        de = _make_fde_from_w(EDGES, w_vals, w_tail=w_tail)
        wf = _make_w_func(EDGES, w_vals, w_tail=w_tail)
    else:
        de, wf = _make_fde_step(w_vals[0], w_vals[1])  # (edges, amps)
    return PiecewiseDECosmology(H0=H0, Om0=OM0, Ode0=Ode0, de_func=de, w_func=wf,
                                Tcmb0=TCMB, Neff=NEFF, m_nu=m_nu)


def _reldiff(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.abs(b))


def _check(name, got, ref, tol):
    r = _reldiff(got, ref)
    status = "PASS" if r < tol else "FAIL"
    print(f"  [{status}] {name:42s} max rel diff = {r:.2e}  (tol {tol:.0e})")
    return r < tol


def main():
    ok = True

    print("Test 1: w_i = -1 (tail -1)  ->  FlatLambdaCDM")
    cosmo = _build(np.full(N, -1.0), w_tail=-1.0)
    ref = FlatLambdaCDM(H0=H0, Om0=OM0, Tcmb0=TCMB, Neff=NEFF)
    ok &= _check("comoving_distance", cosmo.comoving_distance(Z_TEST).value,
                 ref.comoving_distance(Z_TEST).value, 1e-9)
    ok &= _check("de_density_scale (==1)", cosmo.de_density_scale(Z_TEST),
                 np.ones_like(Z_TEST), 1e-12)

    print("Test 2: w_i = -0.9 (tail -0.9), flat  ->  FlatwCDM")
    cosmo = _build(np.full(N, -0.9), w_tail=-0.9)
    ref = FlatwCDM(H0=H0, Om0=OM0, w0=-0.9, Tcmb0=TCMB, Neff=NEFF)
    ok &= _check("comoving_distance", cosmo.comoving_distance(Z_TEST).value,
                 ref.comoving_distance(Z_TEST).value, 1e-9)
    ok &= _check("de_density_scale", cosmo.de_density_scale(Z_TEST),
                 ref.de_density_scale(Z_TEST), 1e-10)

    print("Test 3: w_i = -0.9 (tail -0.9), open Ok0=0.05  ->  wCDM")
    ok0 = 0.05
    cosmo = _build(np.full(N, -0.9), w_tail=-0.9, ok0=ok0)
    ref_ll = FlatLambdaCDM(H0=H0, Om0=OM0, Tcmb0=TCMB, Neff=NEFF)
    Ode0 = 1.0 - OM0 - ok0 - ref_ll.Ogamma0 - ref_ll.Onu0
    ref = wCDM(H0=H0, Om0=OM0, Ode0=Ode0, w0=-0.9, Tcmb0=TCMB, Neff=NEFF)
    ok &= _check("comoving_distance (D_C)", cosmo.comoving_distance(Z_TEST).value,
                 ref.comoving_distance(Z_TEST).value, 1e-9)
    ok &= _check("comoving_transverse (D_M)", cosmo.comoving_transverse_distance(Z_TEST).value,
                 ref.comoving_transverse_distance(Z_TEST).value, 1e-9)

    print("Test 4: w_i = -0.95, flat, m_nu=0.06 eV  ->  FlatwCDM(m_nu)")
    mnu = np.full(3, 0.02) * u.eV
    cosmo = _build(np.full(N, -0.95), w_tail=-0.95, m_nu=mnu)
    ref = FlatwCDM(H0=H0, Om0=OM0, w0=-0.95, Tcmb0=TCMB, Neff=NEFF, m_nu=mnu)
    ok &= _check("comoving_distance", cosmo.comoving_distance(Z_TEST).value,
                 ref.comoving_distance(Z_TEST).value, 1e-9)

    print("Test 5: fde_bins all amps = 1  ->  FlatLambdaCDM")
    edges_fde = np.array([0.0, 0.03, 0.1, 0.4, 0.6, 0.8, 1.1, 1.6, 4.2])
    de, wf = _make_fde_step(edges_fde, np.ones(7))
    ref_ll = FlatLambdaCDM(H0=H0, Om0=OM0, Tcmb0=TCMB, Neff=NEFF)
    Ode0 = 1.0 - OM0 - ref_ll.Ogamma0 - ref_ll.Onu0
    cosmo = PiecewiseDECosmology(H0=H0, Om0=OM0, Ode0=Ode0, de_func=de, w_func=wf,
                                 Tcmb0=TCMB, Neff=NEFF)
    ok &= _check("comoving_distance", cosmo.comoving_distance(Z_TEST).value,
                 ref_ll.comoving_distance(Z_TEST).value, 1e-9)
    ok &= _check("de_density_scale (==1)", cosmo.de_density_scale(Z_TEST),
                 np.ones_like(Z_TEST), 1e-12)

    print("Test 6: production tail (w_tail=-1) vs constant-w at z_star (DE negligible)")
    c_tail = _build(np.full(N, -0.9), w_tail=-1.0)     # production: LCDM tail
    c_full = _build(np.full(N, -0.9), w_tail=-0.9)     # exact wCDM
    dm_tail = c_tail.comoving_transverse_distance(1089.8).value
    dm_full = c_full.comoving_transverse_distance(1089.8).value
    ok &= _check("D_M(z*) tail-choice insensitivity", [dm_tail], [dm_full], 1e-4)

    print("Test 7: fde_bins step => H(z) jump at a bin edge; f_DE(0)=1")
    amps = np.array([1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0])  # bump in [0.1,0.4)
    de, wf = _make_fde_step(edges_fde, amps)
    f0 = de(0.0)[0]
    f_lo = de(0.2)[0]   # inside the bumped bin
    f_hi = de(0.5)[0]   # outside
    print(f"        f_DE(0)={f0:.6f} (want 1)  f_DE(0.2)={f_lo:.3f} (want 1.5)  "
          f"f_DE(0.5)={f_hi:.3f} (want 1.0)")
    ok &= abs(f0 - 1.0) < 1e-12 and abs(f_lo - 1.5) < 1e-12 and abs(f_hi - 1.0) < 1e-12
    print(f"  [{'PASS' if ok else 'FAIL'}] step values")

    print("Test 8: fde_bins step integration — production grid vs fine reference")
    amps8 = np.array([1.0, 1.3, 1.1, 1.0, 1.0, 1.0, 1.0])   # positive (E^2>0)
    de8, wf8 = _make_fde_step(edges_fde, amps8)
    cosmo8, _ = build_cosmology(H0, OM0, 0.0, 0.0, 3, TCMB, NEFF, de8, wf8)
    grid_prod = build_grid(edges_fde, 4.0, 400, 400, 1100.0, split_edges=True)
    grid_ref  = build_grid(edges_fde, 4.0, 40000, 4000, 1100.0, split_edges=True)
    dm_prod = np.interp(Z_TEST, grid_prod, compute_distances(cosmo8, grid_prod)["D_M"])
    dm_ref  = np.interp(Z_TEST, grid_ref,  compute_distances(cosmo8, grid_ref)["D_M"])
    ok &= _check("D_M (prod grid vs fine ref)", dm_prod, dm_ref, 1e-4)

    print("Test 9: fde_bins negative amp => E(z)^2<0 => non-finite distances (rejected)")
    amps9 = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # f_DE=-1 in [0.1,0.4)
    de9, wf9 = _make_fde_step(edges_fde, amps9)
    cosmo9, _ = build_cosmology(H0, OM0, 0.0, 0.0, 3, TCMB, NEFF, de9, wf9)
    with np.errstate(invalid="ignore"):
        dist9 = compute_distances(cosmo9, build_grid(edges_fde, split_edges=True))
    nonfinite = not np.isfinite(dist9["D_M"]).all()
    print(f"        D_M has non-finite values: {nonfinite} (want True)")
    ok &= nonfinite

    print()
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
