import numpy as np
import pytest
from boltzmann.camb.camb_interface import (
    SharedNonlinearRatio, estimate_nonlinear_ratio_saving,
)


@pytest.mark.parametrize("n, expected, eligible", [
    (1, -0.2, False),
    (2, 3.6, True),
    (4, 11.2, True),
])
def test_saving_estimate_matches_amortization_formula(n, expected, eligible):
    result = estimate_nonlinear_ratio_saving(n, 5.0, 1.0, 0.2)
    assert result["saving_seconds"] == pytest.approx(expected)
    assert result["eligible"] is eligible


@pytest.mark.parametrize("args", [
    (0, 5.0, 1.0, 0.2),
    (True, 5.0, 1.0, 0.2),
    (2, np.nan, 1.0, 0.2),
    (2, 5.0, -1.0, 0.2),
])
def test_saving_estimate_rejects_invalid_inputs(args):
    with pytest.raises(ValueError):
        estimate_nonlinear_ratio_saving(*args)


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
    reasons = {
        "positional_args": 0,
        "linear_request": 0,
        "unsupported_variable": 2 if options["var1"] == "Weyl" else 0,
        "cross_spectrum": 2 if options["var1"] != "Weyl" else 0,
    }
    assert shared.diagnostics() == {
        "native_nonlinear_calls": 0,
        "shared_calls": 0,
        "fallback_calls": 2,
        "fallback_reasons": reasons,
    }


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


def test_positional_camb_signature_is_normalized_and_shared():
    native = FakeResult()
    shared = SharedNonlinearRatio(native)
    positional = ("delta_tot", "delta_tot", True, True, True, None, True)
    _, _, first = shared.get_linear_matter_power_spectrum(*positional)
    _, _, second = shared.get_linear_matter_power_spectrum(*positional)
    np.testing.assert_array_equal(first, np.full((2, 2), 7.0))
    np.testing.assert_array_equal(second, np.full((2, 2), 7.0))
    assert native.nonlinear_calls == 1
    assert shared.diagnostics()["shared_calls"] == 1
    assert shared.diagnostics()["fallback_reasons"]["positional_args"] == 0
