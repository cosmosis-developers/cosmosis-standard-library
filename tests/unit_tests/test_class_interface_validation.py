import importlib.util
from pathlib import Path

import numpy as np
import pytest
from cosmosis.datablock import DataBlock


MODULE_PATH = Path(__file__).parents[2] / "boltzmann/class/class_interface.py"
spec = importlib.util.spec_from_file_location("class_interface_under_test", MODULE_PATH)
class_interface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(class_interface)


def cosmology_block(mnu=0.06, num_massive_neutrinos=1):
    block = DataBlock()
    values = {
        "A_s": 2.1e-9,
        "n_s": 0.965,
        "h0": 0.67,
        "ombh2": 0.0224,
        "omch2": 0.12,
        "tau": 0.054,
        "mnu": mnu,
        "num_massive_neutrinos": num_massive_neutrinos,
    }
    for name, value in values.items():
        block[class_interface.cosmo, name] = value
    return block


def config(**overrides):
    result = {
        "cmb": False,
        "lensing": False,
        "mpk": True,
        "kmax": 2.0,
        "zmax": 3.0,
        "save_matter_power_lin": True,
        "save_cdm_baryon_power_lin": False,
    }
    result.update(overrides)
    return result


def test_cdm_baryon_output_requires_massive_neutrinos():
    with pytest.raises(ValueError, match="save_cdm_baryon_power_lin"):
        class_interface.get_class_inputs(
            cosmology_block(mnu=0.0, num_massive_neutrinos=0),
            config(save_cdm_baryon_power_lin=True),
        )


def test_matter_power_requires_kmax_at_least_one():
    with pytest.raises(ValueError, match="kmax must be finite and >= 1.0"):
        class_interface.get_class_inputs(
            cosmology_block(), config(kmax=0.5)
        )


def test_valid_massive_neutrino_configuration_is_accepted():
    params = class_interface.get_class_inputs(
        cosmology_block(),
        config(save_cdm_baryon_power_lin=True),
    )
    assert params["N_ncdm"] == 1
    assert params["P_k_max_h/Mpc"] == 2.0


def test_negative_neutrino_mass_is_rejected():
    with pytest.raises(ValueError, match="mnu must be non-negative"):
        class_interface.get_class_inputs(
            cosmology_block(mnu=-0.1), config()
        )


def test_non_positive_bulk_limit_is_rejected_when_enabled():
    with pytest.raises(ValueError, match="bulk_pk_kmax must be finite and positive"):
        class_interface.get_class_inputs(
            cosmology_block(),
            config(use_bulk_pk=True, bulk_pk_kmax=0.0),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_bulk_limit_is_rejected_when_enabled(value):
    with pytest.raises(ValueError, match="bulk_pk_kmax must be finite and positive"):
        class_interface.get_class_inputs(
            cosmology_block(),
            config(use_bulk_pk=True, bulk_pk_kmax=value),
        )


def test_bulk_limit_boundary_is_accepted():
    params = class_interface.get_class_inputs(
        cosmology_block(), config(use_bulk_pk=True, bulk_pk_kmax=5.0)
    )
    assert params["P_k_max_h/Mpc"] == 2.0


def test_nonfinite_kmax_is_rejected_when_matter_power_is_enabled():
    with pytest.raises(ValueError, match="kmax must be finite and >= 1.0"):
        class_interface.get_class_inputs(
            cosmology_block(), config(kmax=float("nan"))
        )


def test_bulk_extraction_falls_back_to_scalar_api_when_unavailable():
    class ScalarOnly:
        def pk_lin(self, ki, zi):
            return ki + 10.0 * zi

    k = np.array([0.1, 1.0])
    z = np.array([0.0, 0.5])
    result = class_interface._extract_power_grid(
        ScalarOnly(), ScalarOnly().pk_lin, k, z, 0.5,
        config(use_bulk_pk=True, bulk_pk_kmax=5.0),
    )
    expected = np.array([[0.1, 5.1], [1.0, 6.0]])
    np.testing.assert_allclose(result, expected)


def test_bulk_extraction_uses_bulk_low_k_and_scalar_high_k():
    class BulkCapable:
        def __init__(self):
            self.bulk_calls = []

        def pk_lin(self, ki, zi):
            return -1.0

        def get_pk_all(self, k_values, z_values, nonlinear=False, cdmbar=False):
            self.bulk_calls.append((k_values.copy(), nonlinear, cdmbar))
            return np.full((len(z_values), len(k_values)), 7.0)

    c = BulkCapable()
    k = np.array([0.5, 2.0, 4.0])
    z = np.array([0.0, 1.0])
    result = class_interface._extract_power_grid(
        c, c.pk_lin, k, z, 1.0,
        config(use_bulk_pk=True, bulk_pk_kmax=2.0),
    )
    np.testing.assert_allclose(result, [[7.0, 7.0], [7.0, 7.0], [-1.0, -1.0]])
    assert len(c.bulk_calls) == 1
    np.testing.assert_allclose(c.bulk_calls[0][0], [0.5, 2.0])
