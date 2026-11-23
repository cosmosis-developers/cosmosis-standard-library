import cosmosis
import pytest
import numpy as np

from boltzmann.camb.camb_interface import make_camb_cache_key


def test_camb_cache_key_is_stable_and_sensitive():
    base = {
        "cosmology": {"H0": 67.5, "mnu": 0.06},
        "grid": {"z": np.array([0.0, 0.5, 1.0]), "nk": 100},
        "output": {"power_spectra": ["delta_tot"], "nonlinear": "both"},
    }

    assert make_camb_cache_key(base) == make_camb_cache_key(dict(reversed(base.items())))
    assert make_camb_cache_key(base) != make_camb_cache_key({
        **base, "cosmology": {**base["cosmology"], "mnu": 0.15}
    })
    assert make_camb_cache_key(base) != make_camb_cache_key({
        **base, "grid": {**base["grid"], "nk": 101}
    })
    assert make_camb_cache_key(base) != make_camb_cache_key({
        **base, "output": {**base["output"], "power_spectra": ["delta_nonu"]}
    })


def test_camb_cache_key_includes_runtime_configuration():
    signature = {
        "camb_params_repr": "params-v1",
        "more_config": {
            "mode": "power",
            "kmax": 10.0,
            "kmax_extrapolate": 10.0,
            "nk": 200,
            "power_spectra": ["delta_tot"],
        },
        "output_grid": {"z": np.array([0.0, 0.5, 1.0])},
    }

    changes = [
        {"camb_params_repr": "params-v2"},
        {"more_config": {**signature["more_config"], "nk": 201}},
        {"more_config": {**signature["more_config"], "power_spectra": ["delta_nonu"]}},
        {"output_grid": {"z": np.array([0.0, 0.5, 1.1])}},
    ]
    base_key = make_camb_cache_key(signature)
    for change in changes:
        modified = {**signature, **change}
        assert make_camb_cache_key(modified) != base_key

def build_camb_module(mode, **kwargs):
    camb_options = {
        "file": "boltzmann/camb/camb_interface.py",
        "mode": mode,
        "feedback": 0,
        "do_lensing": True,
        "do_reionization": True,
        "nonlinear": "both",
        "use_ppf_w": True,
    } | kwargs

    options = cosmosis.Inifile({"camb": camb_options})

    module = cosmosis.Module.from_options("camb", options)
    module.setup({"camb": camb_options})
    return module

def test_basic_camb():
    module = build_camb_module(mode="all")
    
    block = cosmosis.DataBlock()
    block["cosmological_parameters", "ombh2"] = 0.022
    block["cosmological_parameters", "omch2"] = 0.12
    block["cosmological_parameters", "h0"] = 0.675
    block["cosmological_parameters", "tau"] = 0.06
    block["cosmological_parameters", "n_s"] = 0.965
    block["cosmological_parameters", "A_s"] = 2.1e-9
    block["cosmological_parameters", "omega_k"] = 0.0

    module.execute(block)


def _camb_power_block(omch2=0.12):
    block = cosmosis.DataBlock()
    section = "cosmological_parameters"
    block[section, "ombh2"] = 0.022
    block[section, "omch2"] = omch2
    block[section, "h0"] = 0.675
    block[section, "tau"] = 0.06
    block[section, "n_s"] = 0.965
    block[section, "A_s"] = 2.1e-9
    block[section, "omega_k"] = 0.0
    return block


def test_reuse_computation_matches_and_invalidates():
    module = build_camb_module(mode="power", reuse_computation=True)
    first = _camb_power_block()
    second = _camb_power_block()
    changed = _camb_power_block(omch2=0.121)

    assert module.execute(first) == 0
    assert module.execute(second) == 0
    assert module.execute(changed) == 0

    reused = second["matter_power_lin", "p_k"]
    reference = first["matter_power_lin", "p_k"]
    changed_power = changed["matter_power_lin", "p_k"]
    epsilon = np.max(np.abs(reused - reference) /
                     np.maximum(np.abs(reference), 1e-30))
    changed_epsilon = np.max(np.abs(changed_power - reference) /
                             np.maximum(np.abs(reference), 1e-30))

    assert epsilon == pytest.approx(0.0, abs=1e-14)
    assert changed_epsilon > 1e-8
    assert np.all(np.isfinite(changed_power))
    assert np.all(changed_power > 0)


def test_zmin_non_zero():
    module = build_camb_module(mode="all", zmin=0.2)
    
    block = cosmosis.DataBlock()
    block["cosmological_parameters", "ombh2"] = 0.022
    block["cosmological_parameters", "omch2"] = 0.12
    block["cosmological_parameters", "h0"] = 0.675
    block["cosmological_parameters", "tau"] = 0.06
    block["cosmological_parameters", "n_s"] = 0.965
    block["cosmological_parameters", "A_s"] = 2.1e-9
    block["cosmological_parameters", "omega_k"] = 0.0

    # Unless we explicily tell it not to, CAMB will still integrate down to z=0 to get sigma_8(0),
    # even if we set zmin>0.
    block1 = block.clone()
    module.execute(block1)
    assert block1.has_value("cosmological_parameters", "sigma_8")

    # We should get a warning about this
    with pytest.warns(UserWarning, match="You have set no_integrate_to_z_0 to avoid integrating to redshift zero"):
        module2 = build_camb_module(mode="all", zmin=0.2, no_integrate_to_z_0=True)
    block2 = block.clone()

    # this should give us a warning
    module2.execute(block2)
    # If we set no_integrate_to_z_0 then we won't have sigma_8
    assert not block2.has_value("cosmological_parameters", "sigma_8")

    # if we set zmin=0 but no_integrate_to_z_0 then we should still get sigma_8(0)
    # because it is available
    module3 = build_camb_module(mode="all", zmin=0.0, no_integrate_to_z_0=True)
    block3 = block.clone()
    module3.execute(block3)
    assert block3.has_value("cosmological_parameters", "sigma_8")
