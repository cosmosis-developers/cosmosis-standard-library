# -*- coding: utf-8 -*-
import sys
import os

# from ia_lib import tatt, tidal_alignment, tidal_torque, del4
from des_ia_lib import del4
from des_ia_lib.common import resample_power
import numpy as np

# We now return you to your module.


from cosmosis.datablock import names, option_section
import scipy.interpolate as interp


class Pk_interp(object):
    def __init__(self, ks, Pks):
        if np.all(Pks > 0):
            self.interp_func = interp.interp1d(
                np.log(ks), np.log(Pks), bounds_error=False, fill_value=-np.inf
            )
            self.interp_type = "loglog"
        elif np.all(Pks < 0):
            self.interp_func = interp.interp1d(
                np.log(ks), np.log(-Pks), bounds_error=False, fill_value=-np.inf
            )
            self.interp_type = "minus_loglog"
        else:
            self.interp_func = interp.interp1d(
                np.log(ks), Pks, bounds_error=False, fill_value=0.0
            )
            self.interp_type = "log_ell"

    def __call__(self, ks):
        if self.interp_type == "loglog":
            cl = np.exp(self.interp_func(np.log(ks)))
        elif self.interp_type == "minus_loglog":
            cl = -np.exp(self.interp_func(np.log(ks)))
        else:
            assert self.interp_type == "log_ell"
            cl = self.interp_func(np.log(ks))
        return cl


def compute_c1_baseline():
    C1_M_sun = 5e-14  # h^-2 M_S^-1 Mpc^3
    M_sun = 1.9891e30  # kg
    Mpc_in_m = 3.0857e22  # meters
    C1_SI = C1_M_sun / M_sun * (Mpc_in_m) ** 3  # h^-2 kg^-1 m^3
    # rho_crit_0 = 3 H^2 / 8 pi G
    G = 6.67384e-11  # m^3 kg^-1 s^-2
    H = 100  #  h km s^-1 Mpc^-1
    H_SI = H * 1000.0 / Mpc_in_m  # h s^-1
    rho_crit_0 = 3 * H_SI ** 2 / (8 * np.pi * G)  #  h^2 kg m^-3
    f = C1_SI * rho_crit_0
    return f


def grow(P, Dz, power):
    Pz = np.zeros((len(Dz), len(P)))
    for i, Dzi in enumerate(Dz):
        Pz[i] = P * (Dzi ** power)
    return Pz


def amp_3d(C, num_z, num_k):
    C = np.atleast_1d(C)

    if C.shape == (1,):
        return C * np.ones((num_z, num_k))
    elif C.shape == (num_z,):
        return np.outer(C, np.ones(num_k))
    else:
        assert C.shape == (num_z, num_k)
        return C


def resample_k(k_in, P_in, k_out):
    p_interp = Pk_interp(k_lin, P_in)
    return p_interp(k_out)


def get_IA_terms(
    block,
    k_out,
    k_nl,
    z_out,
    P_lin,
    P_nl,
    Dz,
    A1,
    A2,
    Adel,
    bias_ta,
    bias_tt,
    alpha1,
    alpha2,
    alphadel,
    z_piv,
    Omega_m,
    sub_lowk=False,
):

    # This function reads in PT IA terms (computed at z=0), interpolates them onto k_out,
    # evolves them in z using the growth function Dz, and combines them into physically motivated terms
    # C1, C2 are amplitudes for TT and TA contributions and can be either scalars, 1d array with length Dz, or 2d array with shape (len(Dz), len(k_nl))
    # same for bias_red, bias_blue

    # get from the block
    k_use = k_nl  # this sets which k grid will be used. Changing this may break some things.
    P_IA_dict = {}
    for key in [
        "P_tt_EE",
        "P_tt_BB",
        "P_ta_dE1",
        "P_ta_dE2",
        "P_ta_EE",
        "P_ta_BB",
        "P_mix_A",
        "P_mix_B",
        "P_mix_D_EE",
        "P_mix_D_BB",
        "Plin",
    ]:
        z, k_IA, p = block.get_grid("fastpt", "z", "k_h", key)
        if sub_lowk and key in [
            "P_tt_EE",
            "P_tt_BB",
            "P_ta_EE",
            "P_ta_BB",
            "P_mix_D_EE",
            "P_mix_D_BB",
        ]:
            to_sub = p[:, 0][:, np.newaxis]
            p -= to_sub
            p[:, 0] = p[:, 1]

        assert np.allclose(z, z_out)

        try:
            assert np.allclose(k_use, k_IA)
            P_IA_dict[key] = p
        except (AssertionError, ValueError):
            # interpolate and re-apply correct growth factor if necessary
            p0_orig = p[0]
            p0_out = Pk_interp(k_IA, p0_orig)(k_use)
            if key == "Plin":
                P_IA_dict[key] = grow(p0_out, Dz, 2)
            else:
                P_IA_dict[key] = grow(p0_out, Dz, 4)

    # include power law evolution of amplitude. Right now, this step is always done, alpha = 0 means no evolution.
    # Note, this currently compatible with scalar A1 and A2 or with vector A1 and A2 with length = z_out

    C1_RHOCRIT = compute_c1_baseline()

    C1 = (
        -1.0
        * A1
        * C1_RHOCRIT
        * Omega_m
        / Dz
        * ((1.0 + z_out) / (1.0 + z_piv)) ** alpha1
    )
    Cdel = (
        -1.0
        * Adel
        * C1_RHOCRIT
        * Omega_m
        / Dz
        * ((1.0 + z_out) / (1.0 + z_piv)) ** alphadel
    )
    C2 = (
        5
        * A2
        * C1_RHOCRIT
        * Omega_m
        / Dz ** 2
        * ((1.0 + z_out) / (1.0 + z_piv)) ** alpha2
    )

    # make all the amplitude terms (num_z, num_k).
    # This should be compatible with the z-dependent values passed from above.
    # C1, C2, bias_ta, bias_tt = amp_3d(C1, len(Dz), len(k_out)), amp_3d(C2, len(Dz), len(k_out)), amp_3d(bias_ta, len(Dz), len(k_out)), amp_3d(bias_tt, len(Dz), len(k_out))
    C1 = amp_3d(C1, len(Dz), len(k_use))
    Cdel = amp_3d(Cdel, len(Dz), len(k_use))
    C2 = amp_3d(C2, len(Dz), len(k_use))

    # Get nonlinear Pk. lin Pk is already read-in at the fast-pt k grid from the fastpt block).
    P_IA_dict["P_nl"] = P_nl

    # LA/NLA terms not included here...
    P_IA_out = {}
    p_ta_ii_EE = Cdel ** 2 * P_IA_dict["P_ta_EE"] + C1 * Cdel * (
        2 * P_IA_dict["P_ta_dE1"] + 2 * P_IA_dict["P_ta_dE2"]
    )
    p_ta_ii_BB = Cdel ** 2 * P_IA_dict["P_ta_BB"]
    p_ta_gi = Cdel * (P_IA_dict["P_ta_dE1"] + P_IA_dict["P_ta_dE2"])

    P_IA_out["lin_II_EE"] = C1 * C1 * P_IA_dict["Plin"]
    P_IA_out["nla_II_EE"] = C1 * C1 * P_IA_dict["P_nl"]
    P_IA_out["lin_GI"] = C1 * P_IA_dict["Plin"]
    P_IA_out["nla_GI"] = C1 * P_IA_dict["P_nl"]
    P_IA_out["ta_II_EE"] = p_ta_ii_EE
    P_IA_out["ta_II_BB"] = p_ta_ii_BB
    P_IA_out["ta_GI"] = p_ta_gi

    # TT
    P_IA_out["tt_GI"] = C2 * (P_IA_dict["P_mix_A"] + P_IA_dict["P_mix_B"])
    P_IA_out["tt_II_EE"] = C2 * C2 * P_IA_dict["P_tt_EE"]
    P_IA_out["tt_II_BB"] = C2 * C2 * P_IA_dict["P_tt_BB"]

    # mixed
    p_mix_ii_EE = (
        2.0
        * C2
        * (
            C1 * P_IA_dict["P_mix_A"]
            + C1 * P_IA_dict["P_mix_B"]
            + Cdel * P_IA_dict["P_mix_D_EE"]
        )
    )
    p_mix_ii_BB = 2.0 * Cdel * C2 * P_IA_dict["P_mix_D_BB"]
    P_IA_out["mix_II_EE"] = p_mix_ii_EE
    P_IA_out["mix_II_BB"] = p_mix_ii_BB
    # factors of 2 come from cross-term combinatorics.
    # This factor should be replaced by bin-specific (c1*c2+c2*c1), which is not currently supported by this interface.
    # downstream bin-specific values for C1 and C2 can't currently be applied to tatt,
    # since the output mixes up the different scalings.

    return P_IA_out, k_use


def setup(options):
    sub_lowk = options.get_bool(option_section, "sub_lowk", False)
    ia_model = options.get_string(option_section, "ia_model", "nla")
    name = options.get_string(option_section, "name", default="").lower()
    param_name = options.get_string(option_section, "param_name", default="").lower()
    do_galaxy_intrinsic = options.get_bool(option_section, "do_galaxy_intrinsic", False)
    no_IA_E = options.get_bool(option_section, "no_IA_E", False)
    no_IA_B = options.get_bool(option_section, "no_IA_B", False)

    if name:
        suffix = "_" + name
    else:
        suffix = ""

    if param_name:
        param_name = '_' + param_name
    else:
        param_name = ""
    return (
        sub_lowk,
        ia_model,
        suffix,
        param_name,
        do_galaxy_intrinsic,
        no_IA_E,
        no_IA_B,
    )


def execute(block, config):
    (
        sub_lowk,
        ia_model,
        suffix,
        param_name,
        do_galaxy_intrinsic,
        no_IA_E,
        no_IA_B,
    ) = config

    # Load linear and non-linear matter power spectra
    lin = names.matter_power_lin
    nl = names.matter_power_nl
    cosmo = names.cosmological_parameters
    omega_m = block[cosmo, "omega_m"]

    # Load the matter power spectra
    z_lin, k_lin, p_lin = block.get_grid(lin, "z", "k_h", "p_k")
    z_nl, k_nl, p_nl = block.get_grid(nl, "z", "k_h", "p_k")

    # use ind to handle mild scale-dependence in growth
    ind = np.where(k_lin > 0.03)[0][0]
    Dz = np.sqrt(p_lin[:, ind] / p_lin[0, ind])

    # Re-sample nonlinear power onto same grid as linear
    assert (
        z_nl == z_lin
    ).all(), "Expected identical z values for matter power NL & Linear in IA code"

    # pre-factors to turn off E and B modes
    E_factor = 0 if no_IA_E else 1
    B_factor = 0 if no_IA_B else 1

    ia_section = "intrinsic_alignment_parameters" + param_name

    # check for deprecated parameters
    if (ia_section, "C1") in block:
        raise ValueError("Deprecated TATT parameter specified: " + "C1")

    if (ia_section, "C2") in block:
        raise ValueError("Deprecated TATT parameter specified: " + "C2")

    # Get main parameters - note that all are optional.
    A1 = block.get_double(ia_section, "A1", 1.0)
    A2 = block.get_double(ia_section, "A2", 1.0)
    alpha1 = block.get_double(ia_section, "alpha1", 0.0)
    alpha2 = block.get_double(ia_section, "alpha2", 0.0)
    alphadel = block.get_double(ia_section, "alphadel", alpha1)
    z_piv = block.get_double(ia_section, "z_piv", 0.0)


    if (ia_section, "Adel") in block:
        if (ia_section, "bias_ta") in block:
            raise ValueError("bias_ta is not used when Adel is specified.")
        else:
            Adel = block.get_double(ia_section, "Adel", 1.0)
            bias_ta = bias_tt = 1.0
    else:
        bias_ta = block.get_double(ia_section, "bias_ta", 1.0)
        bias_tt = block.get_double(ia_section, "bias_tt", 1.0)
        Adel = bias_ta * A1



    IA_terms, k_use = get_IA_terms(
        block,
        k_lin,
        k_nl,
        z_lin,
        p_lin,
        p_nl,
        Dz,
        A1,
        A2,
        Adel,
        bias_ta,
        bias_tt,
        alpha1,
        alpha2,
        alphadel,
        z_piv,
        omega_m,
        sub_lowk=sub_lowk,
    )

    ##### complete the proper IA model defintions.
    # Linear alignment modell
    if ia_model == "lin":
        ii_ee_total = E_factor * IA_terms["lin_II_EE"]
        ii_bb_total = 0.0 * IA_terms["ta_II_BB"]
        gi_e_total = E_factor * IA_terms["lin_GI"]

    # Non-linear linear alignment model
    elif ia_model == "nla":
        ii_ee_total = E_factor * IA_terms["nla_II_EE"]
        ii_bb_total = 0.0 * IA_terms["ta_II_BB"]
        gi_e_total = E_factor * IA_terms["nla_GI"]

    # Tidal alignment model
    elif ia_model == "ta":
        ii_ee_total = E_factor * (IA_terms["nla_II_EE"] + IA_terms["ta_II_EE"])
        ii_bb_total = B_factor * IA_terms["ta_II_BB"]
        gi_e_total = E_factor * (IA_terms["nla_GI"] + IA_terms["ta_GI"])

    # Tidal torquing model
    elif ia_model == "tt":
        # In a realistic scenario, this should probably include a tidal-alignment-like
        # term (due to renormalization)
        ii_ee_total = E_factor * IA_terms["tt_II_EE"]
        ii_bb_total = B_factor * IA_terms["tt_II_BB"]
        gi_e_total = E_factor * IA_terms["tt_GI"]

    # Full Tidal Alignment + Tidal Torquing
    elif ia_model == "tatt":
        ii_ee_total = E_factor * (
            IA_terms["nla_II_EE"]
            + IA_terms["ta_II_EE"]
            + IA_terms["tt_II_EE"]
            + IA_terms["mix_II_EE"]
        )
        ii_bb_total = B_factor * (
            IA_terms["ta_II_BB"] + IA_terms["tt_II_BB"] + IA_terms["mix_II_BB"]
        )
        gi_e_total = E_factor * (
            IA_terms["nla_GI"] + IA_terms["ta_GI"] + IA_terms["tt_GI"]
        )  # no mix contribution to GI

    elif ia_model == "mixed_only":
        # not a physically sensible option, but useful for showing the extra
        # contribution we get by considering the cross-terms....
        ii_ee_total = E_factor * IA_terms["mix_II_EE"]
        ii_bb_total = B_factor * IA_terms["mix_II_BB"]
        gi_e_total = np.zeros_like(ii_ee_total)
    else:
        raise ValueError(ia_model + " is not a supported IA model")

    #  Saving results to block. Total EE and BB contributions
    block.put_grid(
        "intrinsic_power_ee" + suffix, "z", z_lin, "k_h", k_use, "p_k", ii_ee_total
    )
    block.put_grid(
        "intrinsic_power_bb" + suffix, "z", z_lin, "k_h", k_use, "p_k", ii_bb_total
    )

    # Total GI contribution
    block.put_grid(
        names.matter_intrinsic_power + suffix,
        "z",
        z_lin,
        "k_h",
        k_use,
        "p_k",
        gi_e_total,
    )

    # We also save the EE total to intrinsic power for consistency with other modules
    block.put_grid(
        names.intrinsic_power + suffix, "z", z_lin, "k_h", k_use, "p_k", ii_ee_total
    )

    # If we've been told to include galaxy-intrinsic power then we
    # need to check if the galaxy bias has already been applied to
    # it or not. We'd prefer people not do that, apparently, so
    # we print out some stuff if it happens.
    if do_galaxy_intrinsic:
        gm = "matter_galaxy_power" + suffix
        z, k, p_gm = block.get_grid(gm, "z", "k_h", "p_k")

        # Check that the bias has not already been applied
        if p_gm.shape == p_nl.shape and np.allclose(p_gm, p_nl):
            b_temp = 1
        else:
            print("WARNING: bias has already been applied to P_gm.")
            print("b_temp=P_gm/P_NL is being applied to P_gal_I by tatt_interface.py")
            b_temp = p_gm / p_nl
            # could use a single b1 or b1(z) to avoid potential scale-dependent issues
            print("b_temp = ", b_temp)


        gal_i_total = b_temp * gi_e_total
        block.put_grid(
            names.galaxy_intrinsic_power + suffix,
            "z",
            z_lin,
            "k_h",
            k_use,
            "p_k",
            gal_i_total,
        )

    return 0
