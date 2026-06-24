"""
Offline validation of binned_de_core distances against CAMB (the Test A oracle),
at a fixed open-wCDM point with massive neutrinos.  No cosmosis needed.

Run:  /opt/anaconda3/bin/python3 background/binned_de/test_vs_camb.py
"""
import numpy as np
import camb
import binned_de_core as c

# Fixed point (matches values_compare_test.ini): open wCDM, constant w, mnu=0.06.
H0, OM0, OB, OK0, MNU = 67.27, 0.3156, 0.0492, 0.05, 0.06
W, NEFF, TCMB, NMASS = -0.9, 3.044, 2.7255, 3
h = H0 / 100.0

z_test = np.array([0.15, 0.30, 0.51, 0.71, 0.93, 1.32, 1.48, 2.33])  # DESI-like
z_star_planck = 1089.80

# ---- CAMB --------------------------------------------------------------------
ombh2 = OB * h**2
omnuh2 = MNU / 93.14
omch2 = OM0 * h**2 - ombh2 - omnuh2
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=OK0,
                   mnu=MNU, nnu=NEFF, num_massive_neutrinos=NMASS, TCMB=TCMB)
pars.set_dark_energy(w=W, wa=0.0, dark_energy_model="ppf")
r = camb.get_background(pars)

DA_camb = r.angular_diameter_distance(z_test)            # Mpc
DM_camb = DA_camb * (1 + z_test)
H_camb = r.hubble_parameter(z_test) / 299792.458         # 1/Mpc
DV_camb = ((1 + z_test)**2 * z_test * DA_camb**2 / H_camb)**(1.0/3.0)
rdrag_camb = r.get_derived_params()["rdrag"]
DM_star_camb = r.angular_diameter_distance(z_star_planck) * (1 + z_star_planck)

# ---- binned_de_core (constant w => w_tail = W to match CAMB at all z) --------
edges = np.array([0.0, 0.1, 0.4, 0.6, 0.8, 1.1, 1.6, 4.2])
de = c._make_fde_from_w(edges, np.full(7, W), w_tail=W)
wf = c._make_w_func(edges, np.full(7, W), w_tail=W)
cosmo, Onu0 = c.build_cosmology(H0, OM0, OK0, MNU, NMASS, TCMB, NEFF, de, wf)

DM_bin = cosmo.comoving_transverse_distance(z_test).value
DA_bin = cosmo.angular_diameter_distance(z_test).value
H_bin = (cosmo.H(z_test).value) / 299792.458
DV_bin = ((1 + z_test)**2 * z_test * DA_bin**2 / H_bin)**(1.0/3.0)
omega_bc = (OM0 - Onu0) * h**2
rdrag_bin = c.r_drag_brieden(OB * h**2, omega_bc, NEFF)
DM_star_bin = cosmo.comoving_transverse_distance(z_star_planck).value


def report(name, a, b):
    rel = np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.abs(b))
    print(f"  {name:14s} max|rel diff| = {rel:.3e}")
    return rel


print("binned_de_core vs CAMB (open wCDM, w=-0.9, Ok=0.05, mnu=0.06):")
report("D_M(z)", DM_bin, DM_camb)
report("D_A(z)", DA_bin, DA_camb)
report("H(z)", H_bin, H_camb)
report("D_V(z)", DV_bin, DV_camb)
print(f"\n  D_M(z*) : binned {DM_star_bin:.2f}  CAMB {DM_star_camb:.2f}  "
      f"rel {abs(DM_star_bin/DM_star_camb-1):.3e}")
print(f"  r_drag  : binned {rdrag_bin:.3f}  CAMB {rdrag_camb:.3f}  "
      f"rel {abs(rdrag_bin/rdrag_camb-1):.3e}")
print(f"  theta*  : binned {100*(rdrag_bin/1.02)/DM_star_bin:.5f}  "
      f"CAMB(RECFAST r*) {r.get_derived_params()['thetastar']:.5f}")
print("\nExpect D_M/D_A/H/D_V agreement <~1e-3 (integration); r_drag within ~0.3% (Brieden vs RECFAST).")
