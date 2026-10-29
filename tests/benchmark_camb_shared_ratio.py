"""Experimental native-grid ratio sharing; does not alter production execution."""
import time
import numpy as np
import camb
from tests.unit_tests.test_camb import build_camb_module, _camb_power_block
from boltzmann.camb.camb_interface import extract_camb_params


class SharedRatio:
    def __init__(self, result):
        self.result = result
        self.ratio = None
        self.grid = None

    def __getattr__(self, name):
        return getattr(self.result, name)

    def get_linear_matter_power_spectrum(self, *args, **kwargs):
        if not kwargs.get("nonlinear", False):
            return self.result.get_linear_matter_power_spectrum(*args, **kwargs)
        linear_args = dict(kwargs, nonlinear=False)
        k, z, linear = self.result.get_linear_matter_power_spectrum(*args, **linear_args)
        if self.ratio is None:
            kn, zn, nonlinear = self.result.get_linear_matter_power_spectrum(*args, **kwargs)
            np.testing.assert_array_equal(k, kn)
            np.testing.assert_array_equal(z, zn)
            if not np.all(np.isfinite(linear)) or np.any(linear <= 0):
                raise ValueError("Ratio sharing requires finite positive reference power")
            self.ratio = nonlinear / linear
            self.grid = (k.copy(), z.copy())
            return kn, zn, nonlinear
        np.testing.assert_array_equal(k, self.grid[0])
        np.testing.assert_array_equal(z, self.grid[1])
        return k, z, linear * self.ratio

    def get_matter_power_interpolator(self, **kwargs):
        return camb.results.CAMBdata.get_matter_power_interpolator(self, **kwargs)


if __name__ == "__main__":
    module = build_camb_module(mode="power", nz=40, nk=100)
    config, extra = module.data
    for mass in (0.06, 0.15, 0.30):
        block = _camb_power_block()
        block["cosmological_parameters", "mnu"] = mass
        result = camb.get_results(extract_camb_params(block, config, extra))
        shared = SharedRatio(result)
        timings = []
        powers = []
        for obj in (result, shared):
            start = time.perf_counter()
            outputs = []
            for variable in ("delta_tot", "delta_nonu", "delta_cdm"):
                p = obj.get_matter_power_interpolator(nonlinear=True, var1=variable, var2=variable)
                outputs.append(p.P(np.linspace(0, 3, 20), np.geomspace(1e-4, 1, 80)))
            timings.append(time.perf_counter()-start)
            powers.append(outputs)
        epsilon = max(float(np.max(abs(a-b)/np.maximum(abs(a), 1e-30)))
                      for a, b in zip(*powers))
        print("RESULT", mass, timings, "epsilon", epsilon, flush=True)
        assert epsilon <= 1e-12
