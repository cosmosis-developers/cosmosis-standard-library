import os
from cosmosis import run_cosmosis
from cosmosis.runtime.handler import activate_segfault_handling
from cosmosis.postprocessing import run_cosmosis_postprocess
import pytest
activate_segfault_handling()

def run_demo(i, args=None, variables=None):
    if args is None:
        args = []
    if variables is None:
        variables = []

    print("Running demo", i)
    if i == 10:
        return run_demo_10()
    override = {("runtime", "verbosity"): "debug"}
    for arg in args:
        key, val = arg.split("=")
        sec, key = key.split(".")
        override[sec, key] = val

    variables_arg = {}
    for arg in variables:
        key, val = arg.split("=")
        sec, key = key.split(".")
        variables_arg[sec, key] = val

    run_cosmosis(f"demos/demo{i}.ini", override=override, variables=variables_arg)
    plots = run_cosmosis_postprocess([f"demos/demo{i}.ini"], outdir=f"output/plots/demo{i}")

    if plots is not None:
        plots.save()

def run_demo_10():
    override = {("grid", "nsample_dimension"): "10"}
    os.environ["HALOFIT"] = "takahashi"
    run_cosmosis(f"demos/demo10.ini", override=override)
    os.environ["HALOFIT"] = "mead2020"
    run_cosmosis(f"demos/demo10.ini", override=override)

    plots = run_cosmosis_postprocess(["output/demo10_mead2020.txt", "output/demo10_takahashi.txt"], outdir="output/plots/demo10")
    plots.save()


def test_demo1():
    run_demo(1)

def test_demo2():
    if not os.path.exists("likelihood/planck2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik"):
        pytest.skip("Planck data not found")
    run_demo(2)

def test_demo3():
    run_demo(3, ["grid.nsample_dimension=10"])

def test_demo4():
    if not os.path.exists("likelihood/planck2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik"):
        pytest.skip("Planck data not found")
    run_demo(4, variables=[
        "cosmological_parameters.n_s=0.962",
        "cosmological_parameters.h0=0.680",
        "cosmological_parameters.ombh2=0.0191",
    ], args=[
        "maxlike.tolerance=1.0"])

def test_demo5():
    run_demo(5, ["emcee.samples=30", "emcee.nsteps=10", "emcee.walkers=20"])

def test_demo6():
    run_demo(6)

def test_demo7():
    run_demo(7)

def test_demo8():
    run_demo(8)

def test_demo9():
    run_demo(9, ["multinest.live_points=50"])

def test_demo10():
    run_demo_10()

def test_demo11():
    run_demo(11, ["grid.nsample_dimension=10"])

def test_demo12():
    run_demo(12)

def test_demo13():
    run_demo(13, ["snake.threshold=3", "snake.nsample_dimension=10"])

def test_demo14():
    run_demo(14, ["zeus.samples=10", "zeus.nsteps=5"])

def test_demo15():
    run_demo(15)

def test_demo16():
    try:
        run_demo(16)
    except ValueError as error:
        err = str(error)
        if "minuit2 wrapper was not compiled" in err:
            pytest.skip("Minuit2 not available")
        else:
            raise

def test_demo17():
    run_demo(17, 
             variables=[
                "cosmological_parameters.n_s=0.96",
                "cosmological_parameters.omega_b=0.0468",
                "cosmological_parameters.h0=0.6881",
                "intrinsic_alignment_parameters.alpha=1.0",
                "intrinsic_alignment_parameters.a=1.0",
                ],
             args=["test.fatal_errors=T"])

def test_demo18():
    run_demo(18)

def test_demo19():
    run_demo(19, ["grid.nsample_dimension=20"])


if __name__ == "__main__":
    test_demo17()
