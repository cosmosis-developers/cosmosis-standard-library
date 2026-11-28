from pathlib import Path
from types import SimpleNamespace

import numpy as np

from structure.projection.projection_tools.kernel import TomoNzKernel


def test_w_kernel_integral_on_saved_ci_input():
    # Captured from PR246 run46147647172, artifact15266178469.
    # Original failure-1.npz SHA256 b2b419765bd938d0ac749df5b579cc6452be8f883a98c863d2f539b934fb52bd.
    filename = Path(__file__).parent / "data" / "w_kernel_ci_input.npz"
    with np.load(filename) as data:
        x = data["source_chi"]
        y = data["source_nchi"]
        chi = data["chi"]
        scale_factor = data["a"]
        reference = data["reference_w"]
    source = TomoNzKernel(np.linspace(0., 1., len(x)), [y])
    nchi_spline = SimpleNamespace(x=x, y=y, xmin=x[0], xmax=x[-1])
    actual = source.get_wofchi_vals(chi, scale_factor, nchi_spline)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[66], 313.2600081675379, rtol=1.e-12)
    valid = np.isfinite(reference)
    np.testing.assert_allclose(actual[valid], reference[valid], rtol=1.e-12, atol=1.e-12)

