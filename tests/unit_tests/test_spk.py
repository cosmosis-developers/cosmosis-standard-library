import numpy as np
import pytest

import cosmosis

from structure.spk import spk_interface as spk_module


class DummyOptions:
    def __init__(self, values):
        self.values = dict(values)

    def get_int(self, section, key, default=None):
        return int(self.values.get(key, default))

    def get_string(self, section, key, default=None):
        value = self.values.get(key, default)
        return "" if value is None else str(value)

    def get_bool(self, section, key, default=None):
        value = self.values.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "t", "yes", "y")
        return bool(value)


class FakePySpk:
    def __init__(self):
        self.build_calls = 0

    def build_sup_model_evaluator(self, SO, relation_kind, k_array):
        self.build_calls += 1
        k_local = np.array(k_array, copy=True)

        def evaluator(**kwargs):
            z = kwargs["z"]
            sup = np.ones_like(k_local) * (1.0 - 0.01 * z)
            return k_local, sup

        return evaluator

    @staticmethod
    def optimal_mass(SO, z, k_array):
        return np.ones_like(k_array) * 1.0e13


@pytest.fixture
def power_block():
    block = cosmosis.DataBlock()
    k = np.array([0.1, 0.2, 0.4])
    z = np.array([0.0, 1.0])
    p = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
            [50.0, 60.0],
        ]
    )
    block.put_grid("matter_power_nl", "k_h", k, "z", z, "P_k", p)
    block["spk", "fb_a"] = 0.40
    block["spk", "fb_pow"] = 0.30
    return block


def test_check_parameter_choice_power_law():
    params = {
        "fb_a": 0.40,
        "fb_pow": 0.30,
        "fb_pivot": None,
        "epsilon": None,
        "alpha": None,
        "beta": None,
        "gamma": None,
        "m_pivot": None,
    }
    relation = spk_module.check_parameter_choice(None, params)
    assert relation == "power_law"


def test_check_parameter_choice_invalid_combination():
    params = {
        "fb_a": 0.40,
        "fb_pow": 0.30,
        "fb_pivot": None,
        "epsilon": None,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "m_pivot": None,
    }
    with pytest.raises(ValueError, match="Invalid parameter combination"):
        spk_module.check_parameter_choice(None, params)


def test_setup_requires_pyspk(monkeypatch):
    monkeypatch.setattr(spk_module, "spk", None)
    monkeypatch.setattr(spk_module, "_PYSPK_IMPORT_ERROR", ImportError("missing"))
    options = DummyOptions({"SO": 500})

    with pytest.raises(ImportError, match="Missing required dependency 'pyspk'"):
        spk_module.setup(options)


def test_execute_writes_output_and_suppression(monkeypatch, power_block):
    fake = FakePySpk()
    monkeypatch.setattr(spk_module, "spk", fake)

    options = DummyOptions(
        {
            "SO": 500,
            "input_section": "matter_power_nl",
            "output_section": "matter_power_spk",
            "suppression_section": "spk_suppression",
        }
    )
    config = spk_module.setup(options)

    status = spk_module.execute(power_block, config)
    assert status == 0

    k_out, z_out, p_out = power_block.get_grid("matter_power_spk", "k_h", "z", "P_k")
    _, _, s_out = power_block.get_grid("spk_suppression", "k_h", "z", "S_k")

    assert np.allclose(k_out, np.array([0.1, 0.2, 0.4]))
    assert np.allclose(z_out, np.array([0.0, 1.0]))
    assert np.allclose(s_out[:, 0], np.ones(3))
    assert np.allclose(s_out[:, 1], np.ones(3) * 0.99)

    _, _, p_in = power_block.get_grid("matter_power_nl", "k_h", "z", "P_k")
    assert np.allclose(p_out[:, 0], p_in[:, 0])
    assert np.allclose(p_out[:, 1], p_in[:, 1] * 0.99)


def test_execute_reuses_cached_evaluator(monkeypatch, power_block):
    fake = FakePySpk()
    monkeypatch.setattr(spk_module, "spk", fake)

    options = DummyOptions({"SO": 500})
    config = spk_module.setup(options)

    first = spk_module.execute(power_block, config)
    second = spk_module.execute(power_block, config)

    assert first == 0
    assert second == 0
    assert fake.build_calls == 1
