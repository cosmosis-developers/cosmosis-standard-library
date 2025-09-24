"""
An interface to the determine the SP(k) model of baryon feedback 
suppression to the non-linear matter power spectrum.

"""

from cosmosis.datablock import option_section, names as section_names
from cosmosis.utils import datablock_to_astropy  #new for cosmosis v3.13

import numpy as np
import astropy.cosmology
import warnings
from scipy.interpolate import LinearNDInterpolator

# pyspk tells the whole of python to always print every warning over and over again.
# Undo it like this so the warning is only printed out once
warnings_filters = warnings.filters[:]
import pyspk.model as spk
warnings.filters = warnings_filters


def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)


def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)


round_up = np.vectorize(my_ceil)
round_down = np.vectorize(my_floor)


def setup(options):

    verbose = options.get_bool(option_section, "verbose", default=False)
    SO = options.get_int(option_section, "SO", default=500)
    fb_table = options.get_string(option_section, "fb_table", default="")

    # If a baryon fraction - halo mass table is provided, load it and create an interpolator
    if fb_table:
        tab = np.loadtxt(fb_table, skiprows=1, delimiter=",")
        inter = LinearNDInterpolator(tab[:, [0, 1]], tab[:, 2], rescale=True)
    else:
        inter = None
        fb_table = None

    extrapolate = options.get_bool(option_section, "extrapolate", default=False)

    # These are parameters for spk
    config = {}
    config["verbose"] = verbose
    config["SO"] = SO
    config["fb_table"] = fb_table
    config["extrapolate"] = extrapolate
    config["interpolator"] = inter

    # Check that the chosen value for the spherical overdensity is either 200 or 500
    if SO not in [200, 500]:
        raise ValueError(
            f"[SPK]: The spherical overdensity SO must be either 200 or 500. You provided: {SO}"
        )

    return config


def check_parameter_choice(fb_table, spk_params):
    """
    Look at the parameters provided for SPK and check that
    they are consistent with exactly one method.
    """
    # SP(k) allows for five different methods to fix the fb-Mhalo relation
    # 0. A non-parametric table of fb values as a function of redshift and halo mass
    # 1. A power law with 3 parameters, fb_a, fb_pow, fb_pivot and an optional 4th parameter, m_pivot (default 1 M_odot)
    # 2. A redshift dependent power law with 3 parameters: alpha, beta, gamma (here m_pivot is fixed to 10^14 M_odot)
    # 3. A redshift dependent double power law with 5 parameters: epsilon, alpha, beta, gamma, m_pivot
    # 4. A non-parametric table of fb values as a function of redshift and halo mass

    spk_parameter_error = f"""SPK inputs were not set correctly. Set exactly one of the following sets of values:

0) fb_table (in the parameters, not values) to use a non-parametric table of fb values as a function of redshift and halo mass
1) fb_a, fb_pow, fb_pivot, and m_pivot to use a power law with 3 parameters and a custom pivot mass
2) alpha, beta, gamma, and m_pivot to use a redshift dependent power law with 3 parameters and a custon pivot mass
3) epsilon, alpha, beta, and gamma to use a redshift dependent double power law
4) fb_a, fb_pow, and fb_pivot to use a power law with 3 parameters and a fixed pivot mass of 10^14 M_odot
        
You set: {spk_params}

    """

    # If fb_table is set, we use method 0 irrespective of the other parameters
    # so no other parameters should have been set in this case
    if fb_table is not None:
        for value in spk_params.values():
            if value:
                raise ValueError(spk_parameter_error)

    # two little mini-functions to check if all or any of the parameters
    # in a list are set.
    is_valid = lambda params: all([spk_params[p] is not None for p in params])
    any_set = lambda params: any([spk_params[p] is not None for p in params])

    # find out which of the possible modes are valid
    mode1_valid = is_valid(["fb_a", "fb_pow", "fb_pivot", "m_pivot"])
    mode2_valid = is_valid(["alpha", "beta", "gamma", "m_pivot"])
    mode3_valid = is_valid(["alpha", "beta", "gamma", "epsilon", "m_pivot"])
    mode4_valid = is_valid(["fb_a", "fb_pow", "fb_pivot"])

    # This is a little complicated, because the parameters for mode 4
    # are a subset of those for mode 1, and those for mode 2 are a subset
    # of those for mode 3.
    # So we can't just check that
    # exactly one mode is valid. Instead we have to specifically check
    # that other parameters aren't set.

    # check if any of the other parameters which should not have
    # been set are in fact set. If so, raise an error.
    if mode1_valid:
        if any_set(["alpha", "beta", "gamma", "epsilon"]):
            raise ValueError(spk_parameter_error)
    elif mode3_valid:
        if any_set(["fb_a", "fb_pow", "fb_pivot"]):
            raise ValueError(spk_parameter_error)
    elif mode2_valid:
        if any_set(["fb_a", "fb_pow", "fb_pivot", "epsilon"]):
            raise ValueError(spk_parameter_error)
    elif mode4_valid:
        if any_set(["alpha", "beta", "gamma", "m_pivot"]):
            raise ValueError(spk_parameter_error)
    else:
        # Otherwise no model has been found to be fully specified,
        # so we raise the same error.
        raise ValueError(spk_parameter_error)


def execute(block, config):
    verbose = config["verbose"]
    SO = config["SO"]
    fb_table = config["fb_table"]
    extrapolate = config["extrapolate"]
    inter = config["interpolator"]

    # Pull the cosmological model from the block in astropy format
    cosmo = datablock_to_astropy(block)

    M_halo = None
    fb = None

    # Load the current non-linear matter power spectrum
    section = section_names.matter_power_nl
    k, z_array, P = block.get_grid(section, "k_h", "z", "P_k")

    params = [
        "fb_a",
        "fb_pow",
        "fb_pivot",
        "epsilon",
        "alpha",
        "beta",
        "gamma",
        "m_pivot",
    ]

    # Get any of the named parameters from above that are in the block
    spk_params = {}
    for param_i in params:
        if block.has_value("spk", param_i):
            spk_params[param_i] = block.get("spk", param_i)
        else:
            spk_params[param_i] = None

    # Check that the specified parameters match exactly one
    # sensible mode.
    check_parameter_choice(fb_table, spk_params)

    sup_array = np.empty_like(P)

    for i in range(len(z_array)):
        z = z_array[i]

        if inter is not None:
            min_mass = round_down(np.log10(spk.optimal_mass(SO, z, np.max(k))), 1)
            max_mass = round_up(np.log10(spk.optimal_mass(SO, z, np.min(k))), 1)
            M_halo = np.logspace(min_mass, max_mass, 100)
            fb = inter(z, M_halo)
            if np.isnan(np.sum(fb)):
                raise ValueError(
                    f"[SPK]: Requested values (z and/or M_halo) outside of the convex hull of the input points in fb_table. Hint: Check your fb_table and the requested redshifts."
                )

        # Get the suppression factor for this redshift
        k, sup = spk.sup_model(
            SO=SO,
            z=z,
            fb_a=spk_params["fb_a"],
            fb_pow=spk_params["fb_pow"],
            fb_pivot=spk_params["fb_pivot"],
            M_halo=M_halo,
            fb=fb,
            extrapolate=extrapolate,
            epsilon=spk_params["epsilon"],
            alpha=spk_params["alpha"],
            beta=spk_params["beta"],
            gamma=spk_params["gamma"],
            m_pivot=spk_params["m_pivot"],
            cosmo=cosmo,
            k_array=k,
            verbose=verbose,
        )
        sup_array[:, i] = sup

    # Scale the matter power spectrum by the suppression
    P_mod = P * sup_array

    # JZ I've also replaced this to avoid assuming the ordering
    # of the P grid in memory.
    section = section_names.matter_power_nl
    block.replace_grid(section, "k_h", k, "z", z_array, "P_k", P_mod)

    # Check for NaNs in the modified power spectrum which occur when 
    # baryon fraction values outside fitting limits are requested
    if np.isnan(P_mod).any(): return 1
    else: return 0


def test_check_parameter_choice():
    import pytest
    empty = {"fb_a": None, "fb_pow": None, "fb_pivot": None, "m_pivot": None, "alpha": None, "beta": None, "gamma": None, "epsilon": None}
    # These ones 
    v1 = empty | {"fb_a": 1, "fb_pow": 2, "fb_pivot": 3, "m_pivot": 4}
    check_parameter_choice(None, v1)
    v2 = empty | {"alpha": 1, "beta": 2, "gamma": 3, "m_pivot": 4}
    check_parameter_choice(None, v2)
    v3 = empty | {"epsilon": 0, "alpha": 1, "beta": 2, "gamma": 3}
    check_parameter_choice(None, v3)
    v4 = empty | {"fb_a": 1, "fb_pow": 2, "fb_pivot": 3}
    check_parameter_choice(None, v4)

    # These ones should raise an error
    with pytest.raises(ValueError):
        check_parameter_choice(None, empty)
    with pytest.raises(ValueError):
        check_parameter_choice(None, v1 | {"alpha": 1})
    with pytest.raises(ValueError):
        check_parameter_choice(None, v1 | {"epsilon": 1})
    with pytest.raises(ValueError):
        check_parameter_choice(None, v2 | {"fb_a": 1})
    with pytest.raises(ValueError):
        check_parameter_choice(None, v2 | {"epsilon": 1})
    with pytest.raises(ValueError):
        check_parameter_choice(None, v3 | {"fb_a": 1})
    with pytest.raises(ValueError):
        check_parameter_choice(None, v3 | {"fb_a": 1})
    with pytest.raises(ValueError):
        check_parameter_choice(None, v4 | {"alpha": 1})
    with pytest.raises(ValueError):
        check_parameter_choice(1, empty | {"alpha": 1})
    
# 0) fb_table (in the parameters, not values) to use a non-parametric table of fb values as a function of redshift and halo mass
# 1) fb_a, fb_pow, fb_pivot, and m_pivot to use a power law with 3 parameters and a custom pivot mass
# 2) alpha, beta, gamma, and m_pivot to use a redshift dependent power law with 3 parameters and a custon pivot mass
# 3) epsilon, alpha, beta, and gamma to use a redshift dependent double power law
# 4) fb_a, fb_pow, and fb_pivot to use a power law with 3 parameters and a fixed pivot mass of 10^14 M_odot


if __name__ == "__main__":
    test_check_parameter_choice()