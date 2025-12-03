#!/usr/bin/env python
from cosmosis import run_cosmosis
from cosmosis.postprocessing import run_cosmosis_postprocess
from cosmosis.runtime.handler import activate_segfault_handling

import pytest
import os
import sys
activate_segfault_handling()

def check_likelihood(capsys, expected, *other_possible):
    captured = capsys.readouterr()
    expect = (expected, *other_possible)
    lines = [line for line in captured.out.split("\n") if "Likelihood =" in line]
    print(lines)
    lines = "\n".join(lines)
    msg = f"Likelihood was expected to be one of {expect} but this was not found. Found these lines: \n{lines}"
    assert any([f"Likelihood =  {val}" in lines for val in (expected, *other_possible)]), msg

def check_no_camb_warnings(capsys):
    captured = capsys.readouterr()
    lines = [line for line in captured.out.split("\n") if "UserWarning: Parameter" in line]
    assert len(lines)==0, f"Found some warnings from CAMB: {lines}"


def test_projection(capsys):
    run_cosmosis("examples/various-spectra.ini", override={("consistency","extra_relations"):"omega_x=omega_c+100"})
    with open("output/various-spectra/cosmological_parameters/values.txt") as f:
        assert "omega_x = 100.261" in f.read()
    check_no_camb_warnings(capsys)



def test_bao(capsys):
    run_cosmosis("examples/bao.ini")
    check_likelihood(capsys, "-157.0", "-157.1", "-156.9")
    check_no_camb_warnings(capsys)

def test_planck(capsys):
    if not os.path.exists("likelihood/planck2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik"):
        pytest.skip("Planck data not found")
    run_cosmosis("examples/planck.ini")
    check_likelihood(capsys, "-1441.14", "-1441.30", "-1441.46", "-502.5")
    check_no_camb_warnings(capsys)
    
def test_planck_class(capsys):
    if not os.path.exists("likelihood/planck2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik"):
        pytest.skip("Planck data not found")
    run_cosmosis("examples/planck_class.ini", override={("class","mpk"):"T"})
    check_no_camb_warnings(capsys)



planck_lite_expected_values = {
    ("T", "TTTEEE", "2018"): "-2864.26",
    ("F", "TTTEEE", "2018"): "-2863.30",
    ("T", "TT", "2018"): ["-1712.90", "-1712.89"],
    ("F", "TT", "2018"): "-1711.94",
    ("T", "TTTEEE", "2015"): "-2812.68",
    ("F", "TTTEEE", "2015"): "-2811.98",
    ("T", "TT", "2015"): "-1744.11",
    ("F", "TT", "2015"): "-1743.41",
}


@pytest.mark.parametrize("choices,expected", list(planck_lite_expected_values.items()))
def test_planck_lite(choices, expected, capsys):
    use_low_ell_bins, spectra, year = choices

    override = {
        ("camb","feedback"): "0",
        ("planck", "spectra"): spectra,
        ("planck", "year"): year,
        ("planck", "use_low_ell_bins"): use_low_ell_bins,
    }
    if isinstance(expected, str):
        expected = [expected]
    run_cosmosis("examples/planck_lite.ini", override=override)
    check_likelihood(capsys, *expected)




def test_wmap(capsys):
    if not os.path.exists("likelihood/wmap9/data/lowlP/mask_r3_p06_jarosik.fits"):
        pytest.skip("WMAP data not found")
    run_cosmosis("examples/wmap.ini")
    check_no_camb_warnings(capsys)

def test_pantheon_emcee(capsys):
    run_cosmosis("examples/pantheon.ini", override={("emcee","samples"):"20"})
    plots = run_cosmosis_postprocess(["examples/pantheon.ini"], outdir="output/pantheon")
    plots.save()
    assert os.path.exists("output/pantheon/cosmological_parameters--omega_m.png")
    check_no_camb_warnings(capsys)

def test_pantheon_plus_shoes(capsys):
    run_cosmosis("examples/pantheon_plus_shoes.ini", override={("runtime","sampler"):"test"})
    check_likelihood(capsys, "-738.23")
    check_no_camb_warnings(capsys)

def test_des_y1(capsys):
    run_cosmosis("examples/des-y1.ini")
    check_likelihood(capsys, "5237.3")
    check_no_camb_warnings(capsys)

def test_des_y1_cl_to_corr(capsys):
    run_cosmosis("examples/des-y1.ini", override={
        ("2pt_shear","file"): "./shear/cl_to_corr/cl_to_corr.py",
        ("2pt_shear","corr_type"): "xi"
        })
    check_likelihood(capsys, "5237.3")
    check_no_camb_warnings(capsys)

def test_des_y3(capsys):
    run_cosmosis("examples/des-y3.ini", override={
        ("pk_to_cl_gg","save_kernels"):"T",
        ("pk_to_cl","save_kernels"):"T"
        })
    check_likelihood(capsys, "6043.23", "6043.34", "6043.37", "6043.33", "6043.30")
    check_no_camb_warnings(capsys)

def test_des_y3_plus_planck(capsys):
    run_cosmosis("examples/des-y3-planck.ini")
    check_likelihood(capsys, "5679.6", "5679.7")
    check_no_camb_warnings(capsys)


def test_des_y3_class(capsys):
    run_cosmosis("examples/des-y3-class.ini")
    # class is not consistent across systems to the level needed here

def test_des_y3_shear(capsys):
    run_cosmosis("examples/des-y3-shear.ini")
    check_likelihood(capsys, "2957.03", "2957.12", "2957.11", "2957.13", "2957.07")
    check_no_camb_warnings(capsys)

def test_des_y3_mira_titan(capsys):
    run_cosmosis("examples/des-y3-mira-titan.ini")
    check_likelihood(capsys, "6048.0", "6048.1", "6048.2")
    check_no_camb_warnings(capsys)

def test_des_y3_mead(capsys):
    run_cosmosis("examples/des-y3.ini", 
                 override={("camb", "halofit_version"): "mead2020_feedback"},
                 variables={("halo_model_parameters", "logT_AGN"): "8.2"}
                 )
    check_likelihood(capsys, "6049.94", "6049.00", "6049.03", "6049.04", "6049.01")
    check_no_camb_warnings(capsys)

def test_act_dr6_lensing(capsys):
    try:
        import act_dr6_lenslike
    except ImportError:
        pytest.skip("ACT likelihood code not found")
    run_cosmosis("examples/act-dr6-lens.ini")
    check_likelihood(capsys, "-9.89", "-9.86", "-9.90")
    check_no_camb_warnings(capsys)

def test_des_y3_5x2pt(capsys):
    run_cosmosis("examples/des-y3-5x2pt.ini")
    check_no_camb_warnings(capsys)


def test_des_y3_6x2pt(capsys):
    if not os.path.exists("likelihood/planck2018/baseline/plc_3.0/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing"):
        pytest.skip("Planck data not found")
    run_cosmosis("examples/des-y3-6x2pt.ini")
    check_no_camb_warnings(capsys)

def test_euclid_emulator(capsys):
    run_cosmosis("examples/euclid-emulator.ini")
    assert os.path.exists("output/euclid-emulator/matter_power_nl/p_k.txt")
    check_no_camb_warnings(capsys)

def test_log_w_example(capsys):
    run_cosmosis("examples/w_model.ini")
    check_no_camb_warnings(capsys)

def test_theta_warning():
    with pytest.raises(RuntimeError):
        run_cosmosis("examples/w_model.ini", override={("consistency","cosmomc_theta"):"T"})

def test_des_kids(capsys):
    run_cosmosis("examples/des-y3_and_kids-1000.ini")
    check_likelihood(capsys, "-199.40", "-199.41")
    check_no_camb_warnings(capsys)


def test_kids(capsys):
    run_cosmosis("examples/kids-1000.ini")
    check_likelihood(capsys, "-47.6")
    check_no_camb_warnings(capsys)

def test_bacco():
    try:
        import tensorflow
    except ImportError:
        pytest.skip("Tensorflow not installed")
    # skip if running on CI with python 3.9 or 3.10 on macOS
    if sys.platform == "darwin" and sys.version_info[:2] in [(3, 9), (3, 10), (3, 11)] and os.environ.get("CI"):
        pytest.skip("Skipping Bacco on MacOS with Python 3.9-3.11 when running CI")

    # The baseline version just does non-linear matter power
    run_cosmosis("examples/bacco.ini")

    # This variant emulates NL power with baryonic effects
    run_cosmosis("examples/bacco.ini",
                 override={
                    ("bacco_emulator", "mode"): "nonlinear+baryons",
                })

    # This variant uses camb to get the NL power and only emulates the baryonic effects
    run_cosmosis("examples/bacco.ini",
                 override={
                    ("bacco_emulator", "mode"): "baryons",
                    ("camb", "nonlinear"): "pk",
                    ("camb", "halofit_version"): "takahashi",
                })

def test_hsc_harmonic(capsys):
    try:
        import sacc
    except ImportError:
        pytest.skip("Sacc not installed")
    run_cosmosis("examples/hsc-y3-shear.ini")
    check_likelihood(capsys, "-109.0")

def test_hsc_real(capsys):
    try:
        import sacc
    except ImportError:
        pytest.skip("Sacc not installed")
    run_cosmosis("examples/hsc-y3-shear-real.ini")
    check_likelihood(capsys, "-122.5")

def test_npipe(capsys):
    try:
        import planckpr4lensing
    except ImportError:
        pytest.skip("Planck PR4 lensing likelihood not found")
    run_cosmosis("examples/npipe.ini")
    check_likelihood(capsys, "-4.22", "-4.23")


def test_desi_dr1(capsys):
    run_cosmosis("examples/desi_dr1.ini")
    check_likelihood(capsys, "-11.25")

def test_desi_dr2(capsys):
    run_cosmosis("examples/desi_dr2.ini")
    check_likelihood(capsys, "-93.02")


def test_candl(capsys):
    try:
        import candl
    except ImportError:
        pytest.skip("Candl not installed")
    run_cosmosis("examples/candl_test.ini")
    check_likelihood(capsys, "-5.83")


def test_hillipop_lollipop(capsys):
    pytest.skip("Skipping Hillipop/Lollipop until we are sure this is working")
    if os.getenv("GITHUB_ACTIONS"):
        pytest.skip("The caching for cobaya is not working on github actions")
    try:
        import planck_2020_lollipop
    except ImportError:
        pytest.skip("Planck 2020 lollipop likelihood not found")
    run_cosmosis("examples/planck-hillipop-lollipop.ini")
    check_likelihood(capsys, "-6476.91", "-6476.90")

def test_decam(capsys):
    run_cosmosis("examples/decam-13k.ini", override={("runtime","sampler"):"test"})
    check_likelihood(capsys, "9442.38", "9442.35", "9442.36")
