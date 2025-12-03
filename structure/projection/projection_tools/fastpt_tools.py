import sys,os
import numpy as np

# hack because fastpt not yet updated for numpy 1.24
if not hasattr(np, 'int'):
    np.int = int
import scipy.integrate
scipy.integrate.trapz = scipy.integrate.trapezoid

import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as intspline
from scipy.interpolate import RectBivariateSpline

from fastpt import FASTPT

try:
    from fastpt.P_extend import k_extend
except ModuleNotFoundError:
    from fastpt.utils.P_extend import k_extend

from cosmosis.datablock import names

def get_Pk_basis_funcs(block, pt_type,
    matter_power_lin_name=names.matter_power_lin,
    matter_power_nl_name=names.matter_power_nl,
    output_nl_grid=True, use_pnl_for_k2=True,
    k_growth=1.e-3, fpt_upsample=4):
    """
    Get the z=0,k-dependent basis functions required
    to construct the galaxy-galaxy
    and galaxy-matter power spectra for a given pt_type


    Parameters
    ----------
    block: DataBlock instance
        block from which to read data
    pt_type: str
        string indicating the type of perturbation theory
        model to use - one of "linear", "oneloop_lag_bk",
        "oneloop_eul_bk".
    matter_power_lin_name : str (default = names.matter_power_lin)
        Name of the section for the linear matter P(k)
    matter_power_nl_name : str (default = names.matter_power_nl)
        Name of the section for the non-linear matter P(k)
    output_on_nl_grid: bool (default=True)
        Output the terms at the same k values as the non-linear
        P(k). If False, interpolate P_nl and P_lin onto the
        fast-pt k grid. It shouldn't matter which we
        do here.
    use_pnl_for_k2 : bool (default=True)
        Use the non-linear matter power for the k^2 term
    k_growth : float (default=1.e-3)
        k value at which to calcaulate growth factor
    fpt_upsample : upsample the linear P(k) by this factor
        when passing it to fast-pt.

    Returns
    -------
    k_out: float array
        array of k values at which P(k) terms are sampled
    PXXNL_out: dict
        Dictionary of pt basis functions
    """

    #Read in linear and non-linear P(k)s from block
    zlin, klin, Plin = block.get_grid(matter_power_lin_name,
        "z", "k_h", "p_k")
    znl, knl, Pnl = block.get_grid(matter_power_nl_name,
        "z", "k_h", "p_k")

    assert np.allclose(zlin, znl) #shit is already convoluted enough
    #without different z values for the linear and non-linear P(k)s

    log_klin = np.log(klin)
    Plin_z0 = Plin[0]

    #Calculate the growth factor from linear P(k) at k closest to k_growth
    k_growth_ind = np.argmin(np.abs(klin-k_growth))
    growth = np.sqrt(Plin[:, k_growth_ind] / Plin[0, k_growth_ind])

    # NM: This whole next section is quite confusing.

    nk = fpt_upsample * len(klin) # higher res increases runtime, decreases potential ringing at low-k
    # eps = 1e-6
    eps = 0.
    kmin = np.log10((1. + eps) * klin[0])
    kmax = np.log10((1. - eps) * klin[-1])
    klin_fpt = np.logspace(kmin, kmax, nk)
    log_klin_fpt = np.log(klin_fpt)
    plin_interp = interp1d(log_klin, np.log(Plin_z0))

    # This interpolation should be at the input bounds.
    # Extrapolate used to avoid failure due to numerical noise.
    # No actual extrapolation is done.
    # We could handle this using a margin or something else
    Plin_fpt = np.exp(plin_interp(log_klin_fpt))

    if (knl[0] < klin_fpt[0]) or (knl[-1] > klin_fpt[-1]):
        EK1 = k_extend(klin_fpt, np.log10(knl[0]), np.log10(knl[-1]))
        klin_fpt = EK1.extrap_k()
        Plin_fpt = EK1.extrap_P_low(Plin_fpt)
        Plin_fpt = EK1.extrap_P_high(Plin_fpt)

    log_klin_fpt = np.log(klin_fpt)
    log_knl = np.log(knl)

    n_pad = len(klin_fpt)
    fastpt = FASTPT(klin_fpt, to_do=['one_loop_dd'],
        low_extrap=-5, high_extrap=3, n_pad=n_pad)

    if pt_type in ['oneloop_lag_bk']:
        PXXNL_lpt = fastpt.one_loop_dd_bias_lpt_NL(Plin_klin_fpt, C_window=.75)
        PXXNL_lpt_z0 = {"Pb1L" : PXXNL_lpt[1], "Pb1L2" : PXXNL_lpt[2],
            "Pb1Lb2L" : PXXNL_lpt[3], "Pb2L" : PXXNL_lpt[4],
            "Pb2L2" : PXXNL_lpt[5], "sig4" : PXXNL_lpt[6]*np.ones_like(klin_fpt)}

        if output_nl_grid:
            # interpolate to nl k grid.
            for key, pk in PXXNL_lpt_z0:
                if key == "sig4":
                    PXXNL_lpt_z0[key] = PXXNL_lpt[6]*np.ones_like(knl)
                PXXNL_lpt_z0[key] = intspline(log_klin_fpt, pk)(log_knl)

        #Apply growth factor to make k,z arrays
        PXXNL_out = {}
        for key, pk in PXXNL_lpt_z0:
            PXXNL_out[key] = np.outer(growth**4, pk)

    elif pt_type in ['oneloop_eul_bk']:
        PXXNL_b1b2bsb3nl = fastpt.one_loop_dd_bias_b3nl(Plin_fpt, C_window=.75)
        PXXNL_b1b2bsb3nl_z0 = {"Pd1d2" : PXXNL_b1b2bsb3nl[2],
            "Pd2d2" : PXXNL_b1b2bsb3nl[3], "Pd1s2" : PXXNL_b1b2bsb3nl[4],
            "Pd2s2" : PXXNL_b1b2bsb3nl[5], "Ps2s2" : PXXNL_b1b2bsb3nl[6],
            "sig3nl" : PXXNL_b1b2bsb3nl[8], "sig4" : PXXNL_b1b2bsb3nl[7]}

        if output_nl_grid:
            # interpolate to nl k grid.
            for key, pk in PXXNL_b1b2bsb3nl_z0.items():
                if key == "sig4":
                    PXXNL_b1b2bsb3nl_z0[key] = pk*np.ones_like(knl)
                else:
                    PXXNL_b1b2bsb3nl_z0[key] = intspline(log_klin_fpt, pk)(log_knl)

        #Apply growth factor to make k,z arrays
        PXXNL_out = {}
        for key, pk in PXXNL_b1b2bsb3nl_z0.items():
            PXXNL_out[key] = np.outer(growth**4, pk)

    else:
        raise ValueError("pt_type %s not valid"%pt_type)

    #We also need to add P_nl, P_lin and k^2P terms to this dictionary
    #if output_nl_grid=True, resample P_lin onto k_nl. Otherwise,
    #resample P_nl onto the fast-pt grid.
    if output_nl_grid:

        PXXNL_out["Pnl"] = Pnl

        #start of by assuming we don't need to resample,
        #then check whether we do....
        resample_P_lin = False
        #we certainly need to resample P_lin if klin and knl
        #have different lengths
        if len(klin) != len(knl):
            resample_P_lin = True
        #if they have the same length, still need to resample
        #if the k values are different
        elif not np.allclose(klin, knl):
            resample_P_lin = True
        plin_z0_interp = intspline(log_klin_fpt, np.log(Plin_fpt))
        Plin_z0_on_nl_grid = np.exp(plin_z0_interp(log_knl))
        PXXNL_out["Plin_from_growth"] = np.outer(growth**2, Plin_z0_on_nl_grid)

        #Now add k^2P(k) term
        knl2_matrix = np.tile(knl ** 2, (Pnl.shape[0], 1))
        if use_pnl_for_k2:
            PXXNL_out["k2P"] = np.multiply(knl2_matrix, Pnl)
        else:
            PXXNL_out["k2P"] = np.multiply(knl2_matrix, Plin_on_nl_grid)

        #Set k_out to knl
        k_out = knl

    else:

        #In this case work out if we need to resample P_nl
        resample_P_nl = False
        if len(knl) != len(klin_fpt):
            resample_P_nl = True
        elif not np.allclose(knl, klin_fpt):
            resample_P_lin = True
        if resample_P_nl:
            pnl_interp = RectBivariateSpline(znl, log_knl, np.log(Pnl))
            pnl_on_fpt_grid = np.exp(pnl_interp(znl, log_knl, grid=True))
        else:
            pnl_on_fpt_grid = Pnl

        PXXNL_out["Pnl"] = pnl_on_fpt_grid
        Plin_out = np.outer(growth**2, Plin_klin_fpt)
        PXXNL_out["Plin_from_growth"] = Plin_out

        #Now add k^2P(k) term
        knl2_matrix = np.tile(klin_fpt ** 2, (Pnl.shape[0], 1))
        if use_pnl_for_k2:
            PXXNL_out["k2P"] = np.multiply(knl2_matrix, pnl_on_fpt_grid)
        else:
            PXXNL_out["k2P"] = np.multiply(klin_fpt, Plin_out)

        #Set k_out to klin_fpt
        k_out = klin_fpt

    return k_out, PXXNL_out

def get_bias_params_bin(block, bin_num, pt_type, bias_section):
    """
    Load bias values for a given bin from the block.

    Parameters
    ----------
    block: DataBlock instance
        block from which to read data
    bin_num: int
        bin index (starting from 1)
    pt_type: str
        PT type (one of 'oneloop_lag_bk', 'oneloop_eul_bk')
    bias_section: str
        section in datablock from which to read bias parameters

    Returns
    -------
    bias_values: dict
        Dictionary of bias values

    """
    #b1E is always required
    b1E = block[bias_section, "b1E_bin%d"%bin_num]

    if pt_type in ['oneloop_lag_bk']:
        param_names = ['b1E', 'b1L', 'b2L', 'bkE']
        param_defaults = [None, b1E - 1, 0.0, 0.0]

    elif pt_type in ['oneloop_eul_bk']:
        param_names = ['b1E', 'b2E', 'bsE', 'b3nlE', 'bkE']
        param_defaults = [None, 0.0, (-4. / 7.) * (b1E - 1), (b1E - 1), 0.0]

    else:
        raise ValueError('No predefined pt_type given')

    bias_values = {}
    #Loop through bias param names and read in to dictionary
    for param_name, default in zip(param_names, param_defaults):
        bias_values[param_name] = block.get_double(bias_section,
            "%s_bin%d"%(param_name, bin_num), default)

    return bias_values


def get_PXm(bias_values, Pk_basis_funcs, pt_type):
    """
    Get P_XX(k) - the tracer-matter power spectrum, generated by summing
    the P(k) terms in Pk_basis_funcs with the bias coefficients in
    bias_values. Also return the individual
    terms as a dictionary.

    Parameters
    ----------
    bias_values: dict
        Dictionary of bias values for the tracer
        (as returned by get_bias_params_bin)
    Pk_basis_funcs: dict
        Dictionary of basis functions to be combined (as
        returned by get_pk_basis_funcs)
    pt_type: str
        PT type - one of 'oneloop_lag_bk', 'oneloop_eul_bk'

    Returns
    -------
    PXX_NL: 2d numpy array
        P_XX(z,k) array
    PXX_NL_terms: dict
        Dictionary of 2d arrays containing the terms
        contributing to PXX_NL.
    """

    #use shorter variable names here as we'll be using
    #these dictionaries a lot!
    PXmNL_terms = {}

    if pt_type in ['oneloop_lag_bk']:
        b1E, b1L, b2L, bk = (bias_values["b1E"], bias_values["b1L"],
            bias_values["b2L"], bias_values["bk"])

        PXmNL_terms["Pd1d1"] = b1E * Pk_basis_funcs["Pnl"]
        PXmNL_terms["Pb1L"] = 0.5 * b1L * Pk_basis_funcs["Pb1L"]
        PXmNL_terms["Pb2L"] = 0.5 * b2L * Pk_basis_funcs["Pb2L"]
        PXmNL_terms["k2P"] = bk * Pk_basis_funcs["k2P"]

        PXmNL = (PXmNL_terms["Pd1d1"] + PXmNL_terms["Pb1L"] + PXmNL_terms["Pb2L"]
            + PXmNL_terms["k2P"])

    elif pt_type in ['oneloop_eul_bk']:
        b1E, b2E, bsE, b3nl, bk = (bias_values["b1E"], bias_values["b2E"],
            bias_values["bsE"], bias_values["b3nlE"], bias_values["bkE"])

        PXmNL_terms["Pd1d1"] = b1E * Pk_basis_funcs["Pnl"]
        PXmNL_terms["Pd1d2"] = 0.5 * b2E * Pk_basis_funcs["Pd1d2"]
        PXmNL_terms["Pd1s2"] = 0.5 * bsE * Pk_basis_funcs["Pd1s2"]
        PXmNL_terms["sig3nl"] = 0.5 * b3nl * Pk_basis_funcs["sig3nl"]
        PXmNL_terms["k2P"] = bk * Pk_basis_funcs["k2P"]

        PXmNL = (PXmNL_terms["Pd1d1"] + PXmNL_terms["Pd1d2"] + PXmNL_terms["Pd1s2"]
            + PXmNL_terms["sig3nl"] + PXmNL_terms["k2P"])
    else:
        raise ValueError("pt_type %s is not valid"%(pt_type))
    # print 'ending PXm array'
    return PXmNL, PXmNL_terms


########################################################################################################################

def get_PXX(bias_values_bin1, bias_values_bin2, Pk_basis_funcs, pt_type):
    """
    Get P_XX(k) - the tracer-tracer power spectrum, generated by summing
    the P(k) terms in Pk_basis_funcs with the bias coefficients in
    bias_values_bin1 and bias_values_bin2. Also return the individual
    terms as a dictionary.

    Parameters
    ----------
    bias_values_bin1: dict
        Dictionary of bias values for the first tracer
        (as returned by get_bias_params_bin)
    bias_values_bin2: dict
        Dictionary of bias values for the second tracer
    Pk_basis_funcs: dict
        Dictionary of basis functions to be combined (as
        returned by get_pk_basis_funcs)
    pt_type: str
        PT type - one of 'oneloop_lag_bk', 'oneloop_eul_bk'

    Returns
    -------
    PXX_NL: 2d numpy array
        P_XX(z,k) array
    PXX_NL_terms: dict
        Dictionary of 2d arrays containing the terms
        contributing to PXX_NL.
    """

    #use shorter variable names here as we'll be using
    #these dictionaries a lot!
    bv1, bv2 = bias_values_bin1, bias_values_bin2
    Pk_terms = {}

    if pt_type == "oneloop_lag_bk":
        Pk_terms["Pd1d1"] = (bv1["b1E"] * bv2["b1E"]
            * Pk_basis_funcs["Pnl"])
        Pk_terms["Pb1L"] = 0.5 * (bv1["b1L"]+bv2["b1L"]) * Pk_basis_funcs["Pb1L"]
        Pk_terms["Pb1L2"] = 0.5 * (bv1["b1L"]*bv2["b1L"]) * Pk_basis_funcs["Pb1L2"]
        Pk_terms["Pb1Lb2L"] = (0.5 * (bv2["b1L"]*bv1["b2L"] +
            bv1["b1L"]*bv2["b2L"]) * Pk_basis_funcs["Pb1Lb2L"])
        Pk_terms["Pb2L"] = 0.5 * (bv1["b2L"] + bv2["b2L"]) * Pk_basis_funcs["Pb2L"]
        Pk_terms["Pb2L2"] = bv1["b2L"] * bv2["b2L"] * Pk_basis_funcs["Pb2L2"]
        Pk_terms["k2Pk"] = (bv1["b1E"] * bv2["bk"] + bv2["b1E"] * bv1["bk"]) * Pk_basis_funcs["k2P"]

        #Now sum terms to get total P(k)
        PXXNL = (Pk_terms["Pd1d1"] + Pk_terms["Pb1L"] + Pk_terms["Pb1L2"]
            + Pk_terms["Pb1Lb2L"] + Pk_terms["Pb2L"] + Pk_terms["Pb2L2"]
            + Pk_terms["k2Pk"])

    elif pt_type == "oneloop_eul_bk":
        Pk_terms["Pd1d1"] = (bv1["b1E"] * bv2["b1E"]
            * Pk_basis_funcs["Pnl"])
        Pk_terms["Pd1d2"] = (0.5 * (bv1["b1E"]*bv2["b2E"]
            + bv2["b1E"]*bv1["b2E"]) * Pk_basis_funcs["Pd1d2"])

        Pk_terms["Pd2d2"] = 0.25 * bv1["b2E"] * bv2["b2E"] * Pk_basis_funcs["Pd2d2"]

        Pk_terms["Pd1s2"] = (0.5 * (bv1["b1E"]*bv2["bsE"] +
            bv2["b1E"]*bv1["bsE"]) * Pk_basis_funcs["Pd1s2"])

        Pk_terms["Pd2s2"] = (0.25 * (bv2["b2E"]*bv1["bsE"] +
            bv1["b2E"]*bv2["bsE"]) * Pk_basis_funcs["Pd2s2"])

        Pk_terms["Ps2s2"] = 0.25 * bv1["bsE"]*bv2["bsE"] * Pk_basis_funcs["Ps2s2"]

        Pk_terms["sig3nl"] = (0.5*(bv1["b1E"]*bv2["b3nlE"] +
            bv2["b1E"]*bv1["b3nlE"]) * Pk_basis_funcs["sig3nl"])

        Pk_terms["k2P"] = ((bv1["b1E"]*bv2["bkE"]
            + bv2["b1E"]*bv1["bkE"]) * Pk_basis_funcs["k2P"])

        PXXNL = (Pk_terms["Pd1d1"] + Pk_terms["Pd1d2"] + Pk_terms["Pd2d2"]
            + Pk_terms["Pd1s2"] + Pk_terms["Pd2s2"] + Pk_terms["Ps2s2"]
            + Pk_terms["sig3nl"] + Pk_terms["k2P"])

    else:
        raise ValueError("pt_type %s is not valid"%(pt_type))

    return PXXNL, Pk_terms
