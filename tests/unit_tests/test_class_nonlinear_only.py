import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cosmosis.datablock import DataBlock, names


@pytest.mark.parametrize(
    "save_matter_power_lin,save_cdm_baryon_power_lin",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_nonlinear_output_for_all_linear_grid_settings(
    save_matter_power_lin, save_cdm_baryon_power_lin
):
    classy = pytest.importorskip("classy")
    interface_path = Path(__file__).parents[2] / "boltzmann" / "class" / "class_interface.py"
    spec = importlib.util.spec_from_file_location("class_interface_under_test", interface_path)
    interface = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interface)

    solver = classy.Class()
    solver.set({
        "h": 0.7,
        "omega_b": 0.0224,
        "omega_cdm": 0.12,
        "A_s": 2.1e-9,
        "n_s": 0.965,
        "output": "mPk",
        "non linear": "halofit",
        "P_k_max_h/Mpc": 2.0,
        "z_max_pk": 0.2,
        "N_ncdm": 1,
        "m_ncdm": 0.06,
        "N_ur": 2.0328,
    })
    solver.compute()
    block = DataBlock()
    block[names.cosmological_parameters, "h0"] = 0.7
    config = {
        "kmax": 1.0,
        "nk": 16,
        "zmax": 0.2,
        "background_only": False,
        "cmb": False,
        "lensing": False,
        "mpk": True,
        "save_matter_power_lin": save_matter_power_lin,
        "save_cdm_baryon_power_lin": save_cdm_baryon_power_lin,
    }
    try:
        interface.get_class_outputs(block, solver, config)
        z, kh, power = block.get_grid("matter_power_nl", "z", "k_h", "p_k")
        expected = np.array([
            [solver.pk(float(k * 0.7), float(zi)) * 0.7**3 for k in kh]
            for zi in z
        ])
        assert block.has_section("matter_power_nl")
        assert np.isfinite(power).all()
        assert block.has_section("matter_power_lin") is save_matter_power_lin
        assert block.has_section("cdm_baryon_power_lin") is save_cdm_baryon_power_lin
        np.testing.assert_allclose(power, expected, rtol=3e-15, atol=1e-10)
    finally:
        solver.struct_cleanup()
        solver.empty()
