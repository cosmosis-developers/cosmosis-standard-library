import numpy as np
import pytest
from boltzmann.camb.camb_interface import SharedNonlinearRatio


class FakeResult:
    def __init__(self):
        self.nonlinear_calls = 0
        self.k = np.array([0.1, 1.0])
        self.linear = np.ones((2, 2))

    def get_linear_matter_power_spectrum(self, **kwargs):
        if kwargs.get("nonlinear", False):
            self.nonlinear_calls += 1
            power = np.full((2, 2), 7.0)
        else:
            power = self.linear.copy()
        return self.k.copy(), np.array([0.0, 1.0]), power


@pytest.mark.parametrize("options", [
    dict(var1="Weyl", var2="Weyl"),
    dict(var1="delta_tot", var2="delta_cdm"),
])
def test_unsupported_requests_always_use_native_result(options):
    native = FakeResult()
    shared = SharedNonlinearRatio(native)
    for _ in range(2):
        _, _, result = shared.get_linear_matter_power_spectrum(nonlinear=True, **options)
        np.testing.assert_array_equal(result, np.full((2, 2), 7.0))
    assert native.nonlinear_calls == 2


@pytest.mark.parametrize("reason", ["grid", "zero", "nan", "units"])
def test_invalid_reuse_falls_back_to_native_result(reason):
    native = FakeResult()
    shared = SharedNonlinearRatio(native)
    options = dict(nonlinear=True, var1="delta_tot", var2="delta_tot")
    shared.get_linear_matter_power_spectrum(**options)
    if reason == "grid":
        native.k *= 2
    elif reason == "zero":
        native.linear[0, 0] = 0
    elif reason == "nan":
        native.linear[0, 0] = np.nan
    else:
        options["hubble_units"] = False
    _, _, result = shared.get_linear_matter_power_spectrum(**options)
    np.testing.assert_array_equal(result, np.full((2, 2), 7.0))
    assert native.nonlinear_calls == 2
