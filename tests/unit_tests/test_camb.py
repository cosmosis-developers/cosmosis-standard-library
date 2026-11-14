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


@pytest.mark.parametrize("parameter,changed", [("mnu", 0.15), ("A_s", 2.3e-9)])
def test_mass_and_amplitude_cache_invalidation(parameter, changed):
    options = dict(mode="power", nk=40, nz=12,
                   power_spectra="delta_tot delta_nonu")
    cached = build_camb_module(reuse_computation=True, reuse_cache_size=2, **options)
    fresh = build_camb_module(reuse_computation=False, **options)
    baseline = 0.06 if parameter == "mnu" else 2.1e-9
    original = None
    for index, value in enumerate((baseline, changed, baseline)):
        actual, expected = _camb_power_block(), _camb_power_block()
        for block in (actual, expected):
            block["cosmological_parameters", "mnu"] = 0.06
            block["cosmological_parameters", "num_massive_neutrinos"] = 1
            block["cosmological_parameters", parameter] = value
        assert cached.execute(actual) == 0
        assert fresh.execute(expected) == 0
        for section in ("matter_power_lin", "matter_power_nl",
                        "cdm_baryon_power_lin", "cdm_baryon_power_nl"):
            result, reference = actual[section, "p_k"], expected[section, "p_k"]
            assert np.all(np.isfinite(result)) and np.all(result > 0)
            epsilon = np.max(np.abs(result-reference) / np.maximum(np.abs(reference), 1e-30))
            assert epsilon <= 1e-12, (parameter, value, section, epsilon)
        power = actual["matter_power_lin", "p_k"]
        if index == 0:
            original = power.copy()
        elif index == 1:
            assert np.max(np.abs(power-original) / np.maximum(np.abs(original), 1e-30)) > 1e-8
        config = cached.data[1]
        assert (config["_reuse_hits"], config["_reuse_misses"]) == ((1, 2) if index == 2 else (0, index+1))


@pytest.mark.parametrize("settings,message", [
    ({"reuse_cache_size": 0}, "reuse_cache_size"),
    ({"reuse_cache_size": -1}, "reuse_cache_size"),
    ({"reuse_cache_max_size": 0}, "reuse_cache_max_size"),
    ({"reuse_cache_size": 4, "reuse_cache_max_size": 2}, "reuse_cache_max_size"),
    ({"reuse_cache_window": 0}, "reuse_cache_window"),
    ({"reuse_cache_threshold": -0.01}, "reuse_cache_threshold"),
    ({"reuse_cache_threshold": 1.01}, "reuse_cache_threshold"),
    ({"reuse_cache_threshold": np.nan}, "reuse_cache_threshold"),
    ({"reuse_cache_threshold": np.inf}, "reuse_cache_threshold"),
])
def test_invalid_cache_settings_fail_at_setup(settings, message):
    with pytest.raises(ValueError, match=message):
        build_camb_module(mode="power", **settings)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_cache_threshold_boundaries(threshold):
    module = build_camb_module(mode="power", reuse_cache_size=1,
                               reuse_cache_max_size=1, reuse_cache_window=1,
                               reuse_cache_threshold=threshold)
    assert module.data[1]["reuse_cache_threshold"] == threshold


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


def test_lru_capacity_recency_and_combined_inputs():
    options = dict(mode="power", nk=30, nz=8)
    cached = build_camb_module(reuse_computation=True, reuse_cache_size=2, **options)
    fresh = build_camb_module(reuse_computation=False, **options)
    inputs = {"A": (0.06, 2.1e-9), "B": (0.15, 2.1e-9),
              "C": (0.15, 2.3e-9)}
    keys = {}
    expected_counts = [(0, 1), (0, 2), (1, 2), (1, 3), (1, 4)]
    for index, label in enumerate(("A", "B", "A", "C", "B")):
        actual, expected = _camb_power_block(), _camb_power_block()
        mass, amplitude = inputs[label]
        for block in (actual, expected):
            block["cosmological_parameters", "mnu"] = mass
            block["cosmological_parameters", "num_massive_neutrinos"] = 1
            block["cosmological_parameters", "A_s"] = amplitude
        assert cached.execute(actual) == 0
        assert fresh.execute(expected) == 0
        config = cached.data[1]
        cache = config["_reuse_cache"]
        assert len(cache) <= 2
        assert (config["_reuse_hits"], config["_reuse_misses"]) == expected_counts[index]
        keys[label] = next(reversed(cache))
        expected_order = (("A",), ("A", "B"), ("B", "A"),
                          ("A", "C"), ("C", "B"))[index]
        assert list(cache) == [keys[item] for item in expected_order]
        for section in ("matter_power_lin", "matter_power_nl"):
            reference, result = expected[section, "p_k"], actual[section, "p_k"]
            assert np.all(np.isfinite(result)) and np.all(result > 0)
            epsilon = np.max(np.abs(result-reference) / np.maximum(np.abs(reference), 1e-30))
            assert epsilon <= 1e-12, (label, section, epsilon)


def test_adaptive_growth_preserves_capacity_and_lru_order():
    options = dict(mode="power", nk=30, nz=8)
    cached = build_camb_module(reuse_computation=True, reuse_cache_size=1,
                               reuse_cache_adaptive=True, reuse_cache_max_size=2,
                               reuse_cache_window=2, **options)
    fresh = build_camb_module(reuse_computation=False, **options)
    keys = {}
    labels = ("A", "B", "C", "B", "D", "A")
    orders = (("A",), ("B",), ("B", "C"), ("C", "B"),
              ("B", "D"), ("D", "A"))
    for index, label in enumerate(labels):
        density = 0.12 + 0.001 * (ord(label)-ord("A"))
        actual = _camb_power_block(omch2=density)
        expected = _camb_power_block(omch2=density)
        assert cached.execute(actual) == 0
        assert fresh.execute(expected) == 0
        config = cached.data[1]
        cache = config["_reuse_cache"]
        assert config["reuse_cache_size"] == (1 if index == 0 else 2)
        assert len(cache) <= config["reuse_cache_size"] <= 2
        keys[label] = next(reversed(cache))
        assert list(cache) == [keys[item] for item in orders[index]]
        for section in ("matter_power_lin", "matter_power_nl"):
            reference, result = expected[section, "p_k"], actual[section, "p_k"]
            assert np.all(np.isfinite(result)) and np.all(result > 0)
            epsilon = np.max(np.abs(result-reference) / np.maximum(np.abs(reference), 1e-30))
            assert epsilon <= 1e-12, (index, section, epsilon)
    assert config["_reuse_hits"] == 1
    assert config["_reuse_misses"] == 5


def test_sigma8_reuse_restores_normalized_amplitude():
    options = dict(mode="power", nk=40, nz=12)
    cached = build_camb_module(reuse_computation=True, reuse_cache_size=2, **options)
    fresh = build_camb_module(reuse_computation=False, **options)
    for target in (0.8, 0.85, 0.8):
        actual, expected = _camb_power_block(), _camb_power_block()
        for block in (actual, expected):
            block["cosmological_parameters", "sigma_8"] = target
        assert cached.execute(actual) == 0
        assert fresh.execute(expected) == 0
        for section in ("matter_power_lin", "matter_power_nl"):
            reference = expected[section, "p_k"]
            epsilon = np.max(np.abs(actual[section, "p_k"]-reference) /
                             np.maximum(np.abs(reference), 1e-30))
            assert epsilon <= 1e-12
        assert actual["cosmological_parameters", "A_s"] == pytest.approx(
            expected["cosmological_parameters", "A_s"], rel=1e-12, abs=0)
        assert actual["cosmological_parameters", "sigma_8"] == pytest.approx(target, rel=1e-6)
    assert cached.data[1]["_reuse_hits"] == 1
    assert cached.data[1]["_reuse_misses"] == 2


def test_reuse_computation_reuses_nonconsecutive_input():
    module = build_camb_module(mode="power", reuse_computation=True,
                               reuse_cache_size=2)
    first = _camb_power_block()
    other = _camb_power_block(omch2=0.121)
    again = _camb_power_block()

    assert module.execute(first) == 0
    assert module.execute(other) == 0
    assert module.execute(again) == 0

    epsilon = np.max(np.abs(again["matter_power_lin", "p_k"] -
                            first["matter_power_lin", "p_k"]) /
                     np.maximum(np.abs(first["matter_power_lin", "p_k"]), 1e-30))
    assert epsilon == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("sharing", [False, True])
def test_lru_matches_independent_fresh_spectra(sharing):
    options = dict(mode="power", nk=40, nz=12,
                   power_spectra="delta_tot delta_nonu")
    cached = build_camb_module(reuse_computation=True, reuse_cache_size=2,
                               share_nonlinear_ratio=sharing, **options)
    fresh = build_camb_module(reuse_computation=False, **options)
    for density in (0.12, 0.121, 0.12):
        actual = _camb_power_block(omch2=density)
        expected = _camb_power_block(omch2=density)
        assert cached.execute(actual) == 0
        assert fresh.execute(expected) == 0
        for section in ("matter_power_lin", "matter_power_nl",
                        "cdm_baryon_power_lin", "cdm_baryon_power_nl"):
            reference = expected[section, "p_k"]
            result = actual[section, "p_k"]
            assert np.all(np.isfinite(result))
            assert np.all(result > 0)
            epsilon = np.max(np.abs(result - reference) /
                             np.maximum(np.abs(reference), 1e-30))
            assert epsilon <= 1e-12, (density, section, epsilon)
            np.testing.assert_array_equal(actual[section, "k_h"], expected[section, "k_h"])
            np.testing.assert_array_equal(actual[section, "z"], expected[section, "z"])


def test_adaptive_reuse_is_opt_in_and_bounded():
    module = build_camb_module(mode="power", reuse_computation=True,
                               reuse_cache_size=2, reuse_cache_adaptive=True,
                               reuse_cache_max_size=4, reuse_cache_window=4,
                               reuse_cache_threshold=0.25, nk=20, nz=6)
    for omch2 in (0.12, 0.121, 0.122, 0.123, 0.12, 0.121, 0.122, 0.123):
        assert module.execute(_camb_power_block(omch2=omch2)) == 0

    config = module.data[1] if isinstance(module.data, (list, tuple)) else module.data
    assert config["reuse_cache_size"] <= 4
    assert config["_reuse_hits"] + config["_reuse_misses"] == 8


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_cosmological_input_fails_fast(invalid):
    module = build_camb_module(mode="power")
    block = _camb_power_block(omch2=invalid)
    with pytest.raises(ValueError, match="Non-finite cosmological input"):
        module.execute(block)


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
