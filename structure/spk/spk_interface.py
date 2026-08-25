"""CosmoSIS interface for applying py-SP(k) baryonic suppression.

This module multiplies an input matter power spectrum grid by the py-SP(k)
suppression factor and writes the result to a configurable output section.
"""

from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
from cosmosis.utils import datablock_to_astropy

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from typing import Any

spk: Any | None = None
_PYSPK_IMPORT_ERROR: ImportError | None = None

SPK_SECTION = "spk"
SPK_PARAMS = ("fb_a", "fb_pow", "fb_pivot", "epsilon", "alpha", "beta", "gamma", "m_pivot")
Z_OUT_OF_RANGE_POLICIES = ("raise", "nan")


def _spk_or_raise():
    """Return the pyspk module, importing lazily when first needed."""
    global spk
    global _PYSPK_IMPORT_ERROR

    if spk is not None:
        return spk

    if _PYSPK_IMPORT_ERROR is not None:
        raise ImportError(
            "[SPK] Missing required dependency 'pyspk'. "
            "Install with: pip install 'pyspk>=2.0.1'"
        ) from _PYSPK_IMPORT_ERROR

    try:
        import pyspk as pyspk_module
    except ImportError as import_error:
        _PYSPK_IMPORT_ERROR = import_error
        raise ImportError(
            "[SPK] Missing required dependency 'pyspk'. "
            "Install with: pip install 'pyspk>=2.0.1'"
        ) from import_error

    # check that version is new enough
    version = pyspk_module.__version__.split('.')
    if version[0] == "1" or version == (2, 0, 0):
        raise ImportError(
            "[SPK] pyspk version must be >= 2.0.1, "
            f"found {pyspk_module.__version__}. "
            "Upgrade with: pip install --upgrade 'pyspk>=2.0.1'"
        )

    spk = pyspk_module
    return pyspk_module


def setup(options):
    """Load static SP(k) options and initialize evaluator caches.

    Args:
        options: Cosmosis options block.

    Returns:
        dict: Module configuration shared by each ``execute`` call.
    """
    _spk_or_raise()

    so = options.get_int(option_section, "SO", default=500)
    if so not in (200, 500):
        raise ValueError(f"[SPK] SO must be 200 or 500, received {so}.")

    input_section = options.get_string(
        option_section,
        "input_section",
        default=section_names.matter_power_nl,
    )
    output_section = options.get_string(
        option_section,
        "output_section",
        default=section_names.matter_power_nl,
    )
    suppression_section = options.get_string(
        option_section,
        "suppression_section",
        default="",
    ).strip()

    fb_table = options.get_string(option_section, "fb_table", default="").strip()
    if fb_table:
        table = np.loadtxt(fb_table, skiprows=1, delimiter=",")
        if table.ndim != 2 or table.shape[1] < 3:
            raise ValueError(
                "[SPK] fb_table must have at least three columns: "
                "z, M_halo, fb."
            )
        interpolator = LinearNDInterpolator(table[:, [0, 1]], table[:, 2], rescale=True)
    else:
        fb_table = None
        interpolator = None

    z_out_of_range = options.get_string(option_section, "z_out_of_range", default="raise").strip().lower()
    if z_out_of_range not in Z_OUT_OF_RANGE_POLICIES:
        valid = ", ".join(Z_OUT_OF_RANGE_POLICIES)
        raise ValueError(
            f"[SPK] z_out_of_range must be one of: {valid}. Received '{z_out_of_range}'."
        )

    return {
        "verbose": options.get_bool(option_section, "verbose", default=False),
        "SO": so,
        "input_section": input_section,
        "output_section": output_section,
        "suppression_section": suppression_section,
        "fb_table": fb_table,
        "extrapolate": options.get_bool(option_section, "extrapolate", default=False),
        "z_out_of_range": z_out_of_range,
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
    def has_all(keys):
        return all(spk_params[key] is not None for key in keys)

    def has_any(keys):
        return any(spk_params[key] is not None for key in keys)

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
    pyspk = _spk_or_raise()
    cache = config["evaluator_cache"]
    cached = cache.get(relation_kind)
    if cached is not None and np.array_equal(cached["k"], k_array):
        return cached["evaluator"]

    evaluator = pyspk.build_sup_model_evaluator(
        SO=config["SO"],
        relation_kind=relation_kind,
        k_array=k_array,
        z_out_of_range=config["z_out_of_range"],
    )
    cache[relation_kind] = {"k": np.array(k_array, copy=True), "evaluator": evaluator}
    return evaluator


def _write_grid(block, section, k_array, z_array, values, value_name):
    """Write a 2D grid to a section, replacing existing values when present."""
    if block.has_value(section, value_name):
        block.replace_grid(section, "k_h", k_array, "z", z_array, value_name, values)
    else:
        block.put_grid(section, "k_h", k_array, "z", z_array, value_name, values)


def _mhalo_and_fb_from_table(config, z, k_array):
    """Build binned ``M_halo`` and ``fb`` arrays from the user table at one redshift."""
    pyspk = _spk_or_raise()
    optimal_mass = np.asarray(pyspk.optimal_mass(config["SO"], z, k_array), dtype=float)
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
    input_section = config["input_section"]
    output_section = config["output_section"]
    suppression_section = config["suppression_section"]

    k, z_array, p_nl = block.get_grid(input_section, "k_h", "z", "P_k")
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

    _write_grid(block, output_section, k, z_array, p_nl_mod, "P_k")

    if suppression_section:
        _write_grid(block, suppression_section, k, z_array, suppression, "S_k")

    return 1 if np.isnan(p_nl_mod).any() else 0
