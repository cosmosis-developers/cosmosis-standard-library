"""Cosmosis interface for applying py-SP(k) baryonic suppression.

This module modifies the existing non-linear matter power spectrum in
``matter_power_nl/P_k`` by multiplying it by the py-SP(k) suppression factor.
"""

from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
from cosmosis.utils import datablock_to_astropy

import numpy as np
import pyspk as spk
from scipy.interpolate import LinearNDInterpolator

SPK_SECTION = "spk"
SPK_PARAMS = ("fb_a", "fb_pow", "fb_pivot", "epsilon", "alpha", "beta", "gamma", "m_pivot")


def setup(options):
    """Load static SP(k) options and initialize evaluator caches.

    Args:
        options: Cosmosis options block.

    Returns:
        dict: Module configuration shared by each ``execute`` call.
    """
    so = options.get_int(option_section, "SO", default=500)
    if so not in (200, 500):
        raise ValueError(f"[SPK] SO must be 200 or 500, received {so}.")

    fb_table = options.get_string(option_section, "fb_table", default="").strip()
    if fb_table:
        table = np.loadtxt(fb_table, skiprows=1, delimiter=",")
        interpolator = LinearNDInterpolator(table[:, [0, 1]], table[:, 2], rescale=True)
    else:
        fb_table = None
        interpolator = None

    return {
        "verbose": options.get_bool(option_section, "verbose", default=False),
        "SO": so,
        "fb_table": fb_table,
        "extrapolate": options.get_bool(option_section, "extrapolate", default=False),
        "interpolator": interpolator,
        "evaluator_cache": {},
    }


def check_parameter_choice(fb_table, spk_params):
    """Validate relation inputs and return the pyspk relation kind.

    Args:
        fb_table (str | None): Optional input table path for binned mode.
        spk_params (dict[str, float | None]): Runtime parameters from ``[spk]``.

    Returns:
        str: One of ``binned``, ``power_law``, ``cosmo_power_law``,
        ``double_power_law``.

    Raises:
        ValueError: If parameters do not define exactly one supported mode.
    """
    validation_error = f"""[SPK] Invalid parameter combination.
Provide exactly one relation definition:
1) fb_table (module option) only -> binned relation
2) fb_a, fb_pow, [fb_pivot] -> power-law relation
3) alpha, beta, gamma -> cosmology-based power-law relation
4) epsilon, alpha, beta, gamma, m_pivot -> double power-law relation

Received: {spk_params}
"""
    has_all = lambda keys: all(spk_params[key] is not None for key in keys)
    has_any = lambda keys: any(spk_params[key] is not None for key in keys)

    if fb_table is not None:
        if has_any(SPK_PARAMS):
            raise ValueError(validation_error)
        return "binned"

    if has_all(("epsilon", "alpha", "beta", "gamma", "m_pivot")):
        if has_any(("fb_a", "fb_pow", "fb_pivot")):
            raise ValueError(validation_error)
        return "double_power_law"

    if has_all(("alpha", "beta", "gamma")):
        if has_any(("fb_a", "fb_pow", "fb_pivot", "epsilon", "m_pivot")):
            raise ValueError(validation_error)
        return "cosmo_power_law"

    if has_all(("fb_a", "fb_pow")):
        if has_any(("epsilon", "alpha", "beta", "gamma", "m_pivot")):
            raise ValueError(validation_error)
        return "power_law"

    raise ValueError(validation_error)


def _read_spk_params(block):
    """Read SP(k) parameters from the datablock and return missing values as ``None``."""
    params = {}
    for param in SPK_PARAMS:
        params[param] = block.get(SPK_SECTION, param) if block.has_value(SPK_SECTION, param) else None
    return params


def _get_or_build_evaluator(config, relation_kind, k_array):
    """Get a cached pyspk evaluator or build one for the current k-grid."""
    cache = config["evaluator_cache"]
    cached = cache.get(relation_kind)
    if cached is not None and np.array_equal(cached["k"], k_array):
        return cached["evaluator"]

    evaluator = spk.build_sup_model_evaluator(
        SO=config["SO"], relation_kind=relation_kind, k_array=k_array
    )
    cache[relation_kind] = {"k": np.array(k_array, copy=True), "evaluator": evaluator}
    return evaluator


def _mhalo_and_fb_from_table(config, z, k_array):
    """Build binned ``M_halo`` and ``fb`` arrays from the user table at one redshift."""
    optimal_mass = np.asarray(spk.optimal_mass(config["SO"], z, k_array), dtype=float)
    logm = np.log10(optimal_mass)
    min_logm = np.floor(np.min(logm) * 10.0) / 10.0
    max_logm = np.ceil(np.max(logm) * 10.0) / 10.0
    m_halo = np.logspace(min_logm, max_logm, 100)
    fb = config["interpolator"](z, m_halo)

    if np.isnan(fb).any():
        raise ValueError(
            "[SPK] Requested (z, M_halo) values are outside the convex hull of fb_table. "
            "Check fb_table coverage or enable extrapolation only when physically justified."
        )

    return m_halo, fb


def execute(block, config):
    """Apply SP(k) suppression to the non-linear matter power spectrum.

    Args:
        block: Cosmosis datablock.
        config (dict): Setup configuration from :func:`setup`.

    Returns:
        int: ``0`` on success, ``1`` if the modified spectrum contains NaNs.
    """
    section = section_names.matter_power_nl
    k, z_array, p_nl = block.get_grid(section, "k_h", "z", "P_k")
    spk_params = _read_spk_params(block)
    relation_kind = check_parameter_choice(config["fb_table"], spk_params)
    evaluator = _get_or_build_evaluator(config, relation_kind, k)

    suppression = np.empty_like(p_nl)
    cosmo = None
    if relation_kind in ("cosmo_power_law", "double_power_law"):
        cosmo = datablock_to_astropy(block)

    for i, z in enumerate(z_array):
        if relation_kind == "binned":
            m_halo, fb = _mhalo_and_fb_from_table(config, z, k)
            _, sup = evaluator(
                z=z,
                M_halo=m_halo,
                fb=fb,
                extrapolate=config["extrapolate"],
                verbose=config["verbose"],
            )
        elif relation_kind == "power_law":
            kwargs = {
                "z": z,
                "fb_a": spk_params["fb_a"],
                "fb_pow": spk_params["fb_pow"],
                "verbose": config["verbose"],
            }
            if spk_params["fb_pivot"] is not None:
                kwargs["fb_pivot"] = spk_params["fb_pivot"]
            _, sup = evaluator(**kwargs)
        elif relation_kind == "cosmo_power_law":
            _, sup = evaluator(
                z=z,
                alpha=spk_params["alpha"],
                beta=spk_params["beta"],
                gamma=spk_params["gamma"],
                cosmo=cosmo,
                verbose=config["verbose"],
            )
        else:
            _, sup = evaluator(
                z=z,
                epsilon=spk_params["epsilon"],
                alpha=spk_params["alpha"],
                beta=spk_params["beta"],
                gamma=spk_params["gamma"],
                m_pivot=spk_params["m_pivot"],
                cosmo=cosmo,
                verbose=config["verbose"],
            )

        suppression[:, i] = sup

    p_nl_mod = p_nl * suppression
    block.replace_grid(section, "k_h", k, "z", z_array, "P_k", p_nl_mod)
    return 1 if np.isnan(p_nl_mod).any() else 0
