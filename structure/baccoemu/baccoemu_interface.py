try:
    import baccoemu
except ImportError:
    raise RuntimeError("Cannot import baccoemu. Try installing with "
            "pip install 'baccoemu>=2.3.0'")

from cosmosis.datablock import option_section, names
from scipy.interpolate import RectBivariateSpline
from scipy import optimize
import numpy as np
import traceback


def setup(options):
    mode = options.get_string(option_section, "mode", default="nonlinear")
    nonlin_model = options.get_string(option_section, "nonlinear_model", default="Arico2023")
    baryon_model = options.get_string(option_section, "baryonic_model", default="Arico2021")
    bcm_extrap_cosmo = options.get_bool(option_section, "extrapolate_bcm_cosmo_params", default=False)

    nonlin_options = ["Angulo2021", "Arico2023"]
    baryon_options = ["Arico2021", "Burger2025"]
    if nonlin_model not in nonlin_options:
        raise ValueError(f"BaccoEmu: 'nonlinear_model' must be one of {nonlin_options}")
    if baryon_model not in baryon_options:
        raise ValueError(f"BaccoEmu: 'baryon_model' must be one of {baryon_options}")

    # do this only once in the setup phase
    emulator = baccoemu.Matter_powerspectrum(nonlinear_model_name=nonlin_model,
                                             baryonic_model_name=baryon_model)

    allowed_modes = ["nonlinear", "baryons", "nonlinear+baryons"]
    if mode not in allowed_modes:
        raise ValueError(f"BaccoEmu: 'mode' must be one of {allowed_modes}")

    return mode, emulator, bcm_extrap_cosmo


def check_params(params, bounds):
    """
    Check if needed parameters exist and are within the correct bounds.

    params: dict(name, value) - parameter values
    bounds: dict(name, [min, max]) - allowed ranges for parameters

    Returns None if check is successful. Raises a KeyError if a needed
    parameter does not exist or a ValueError if a parameter is out of
    bounds.
    """
    for p in bounds:
        if p not in params:
            raise KeyError(f"BaccoEmu: missing parameter {p}")
        if params[p] > bounds[p][1] or params[p] < bounds[p][0]:
            raise ValueError(f"BaccoEmu: parameter {p} = {params[p]} is out of bounds {bounds[p]}")


def get_bounds(emulator, mode="nonlinear"):
    """Get a dictionary of parameter bounds from a Bacco emulator instance."""
    keys = emulator.emulator[mode]["keys"]
    bounds = {key: emulator.emulator[mode]["bounds"][i] for i, key in enumerate(keys) if key != "expfactor"}
    return bounds


def setup_params(block, emulator, mode, bcm_extrap_cosmo=False):
    """
    Setup parameter values that are needed to run given emulator.


    """
    cosmo = names.cosmological_parameters
    baryons = names.baryon_parameters

    # In bacco, omega_cold refers to CDM + baryons
    omega_cold = block[cosmo, "omega_c"] + block[cosmo, "omega_b"]

    params = {
        "omega_cold": omega_cold,
        "omega_baryon": block[cosmo, "omega_b"],
        "neutrino_mass": block[cosmo, "mnu"],
        "hubble": block[cosmo, "h0"],
        "ns": block[cosmo, "n_s"],
        "w0": block.get_double(cosmo, "w", -1.0),
        "wa": block.get_double(cosmo, "wa", 0.0),
    }

    # bacco uses sigma8_cold, need to convert from A_s
    if block.has_value(cosmo, "sigma8_cold"):
        params["sigma8_cold"] = block[cosmo, "sigma8_cold"]
    else:
        A_s = block[cosmo, "a_s"]
        sigma8_cold = emulator.get_sigma8(cold=True, **(params | {"A_s": A_s, "expfactor": 1}))
        params["sigma8_cold"] = sigma8_cold
        # also write back to block, so that it can be re-used if necessary
        block[cosmo, "sigma8_cold"] = sigma8_cold

    if mode == "baryon":
        # load BCM parameters
        params |= {
            "M_c": block[baryons, "m_c"],
            "eta": block[baryons, "eta"],
            "beta": block[baryons, "beta"],
            "M1_z0_cen": block[baryons, "m1_z0_cen"],
            "theta_out": block.get_double(baryons, "theta_out", 0.0),  # not needed for Burger2025 model
            "theta_inn": block[baryons, "theta_inn"],
            "M_inn": block.get_double(baryons, "m_inn", 13.36),  # not needed for Burger2025 model
        }

    bounds = get_bounds(emulator, mode)

    if mode == "nonlinear":
        check_params(params, bounds)
        return params

    if mode == "baryon":
        if not bcm_extrap_cosmo:
            check_params(params, bounds)
            return params

        # we need to handle the situation where cosmological parameters might be within
        # the nonlinear emulator range, but outside the BCM emulator range
        # see Arico et al. (2023) section 3.4 for more details
        # this specific implementation is based on the code used by Garcia-Garcia et al. (2024)
        # (https://github.com/Cosmotheka/Cosmotheka_likelihoods/blob/main/ClLike/cl_like/bacco.py)
        cosmo_params = [p for p in bounds if p in emulator.emulator["nonlinear"]["keys"]]
        # clip Om and Ob, keeping baryon fraction the same
        #  this could probably be done analytically rather than optimizing numerically
        fb = params["omega_baryon"] / params["omega_cold"]
        omega_cold = np.clip(params["omega_cold"], *bounds["omega_cold"])
        omega_baryon = np.clip(params["omega_baryon"], *bounds["omega_baryon"])
        obj_func = lambda o: np.abs(o[1] / o[0] - fb)
        res = optimize.minimize(obj_func, (omega_cold, omega_baryon),
                                bounds=(bounds["omega_cold"], bounds["omega_baryon"]), tol=1e-3)
        params["omega_cold"] = res.x[0]
        params["omega_baryon"] = res.x[1]
        # clip all other parameters
        for p in cosmo_params:
            if p != "omega_cold" and p != "omega_baryon":
                params[p] = np.clip(params[p], *bounds[p])

        check_params(params, bounds)
        return params

    raise ValueError(f"unknown mode: {mode}")


def get_nonlinear_boost(block, emulator, k, a):
    params = setup_params(block, emulator, "nonlinear")
    _, F_nl = emulator.get_nonlinear_boost(k=k, expfactor=a, cold=False, **params)
    return F_nl


def get_baryonic_boost(block, emulator, k, a, bcm_extrap_cosmo=False):
    params = setup_params(block, emulator, "baryon", bcm_extrap_cosmo=bcm_extrap_cosmo)
    _, F_baryon = emulator.get_baryonic_boost(k=k, expfactor=a, **params)
    return F_baryon


def execute(block, config):
    mode, emulator, bcm_extrap_cosmo = config

    if mode == "baryons":
        # assume nonlinear Pk has already been calculated
        z, k, Pk = block.get_grid("matter_power_nl", "z", "k_h", "p_k")
    else:
        z, k, Pk = block.get_grid("matter_power_lin", "z", "k_h", "p_k")

    # This is required to avoid a bounds error
    zmask = z < 1.5
    a = 1 / (1 + z[zmask])
    kmask = (k < 4.69) & (k > 0.0001)

    try:
        if mode == "nonlinear":
            boost = get_nonlinear_boost(block, emulator, k[kmask], a)
        elif mode == "nonlinear+baryons":
            boost = get_nonlinear_boost(block, emulator, k[kmask], a)
            boost *= get_baryonic_boost(block, emulator, k[kmask], a, bcm_extrap_cosmo=bcm_extrap_cosmo)
        elif mode == "baryons":
            boost = get_baryonic_boost(block, emulator, k[kmask], a, bcm_extrap_cosmo=bcm_extrap_cosmo)
        else:
            raise RuntimeError("This should not happen")

        # interpolate and save
        I = RectBivariateSpline(np.log10(k[kmask]), z[zmask], np.log10(boost).T)
        boost_interp = 10 ** I(np.log10(k), z).T

        P_nl = boost_interp * Pk
        block.put_grid("matter_power_nl", "z", z, "k_h", k, "p_k", P_nl)

    except ValueError as error:
        # print traceback from exception
        print(traceback.format_exc())
        return 1

    return 0

