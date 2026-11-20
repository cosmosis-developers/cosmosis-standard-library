import cosmosis
import numpy as np
import pytest
from cosmosis.runtime.pipeline import Pipeline

sacc = pytest.importorskip("sacc")

TWO_POINT = "galaxy_shear_cosebi_ee"
ONE_POINT = "galaxy_stellarmassfunction"
PAIR_RANGE = f"angle_range_{TWO_POINT}_source_0_source_1"
COMMON_RANGE = f"angle_range_{TWO_POINT}"
ONE_POINT_RANGE = f"onepoint_range_{ONE_POINT}"
LEGACY_RANGE = f"angle_range_{ONE_POINT}_source_1"


@pytest.fixture
def sacc_file(tmp_path):
    data = sacc.Sacc()
    for tracer in ("source_0", "source_1"):
        data.add_tracer("Misc", tracer)

    for tracers, offset in [
        (("source_0", "source_0"), 0),
        (("source_0", "source_1"), 10),
    ]:
        for n in (1, 2, 3, 4):
            data.add_data_point(TWO_POINT, tracers, offset + n, n=n)
    for tracer, offset in [("source_0", 20), ("source_1", 30)]:
        for mass in (1, 2, 3, 4):
            data.add_data_point(ONE_POINT, (tracer,), offset + mass, mass=mass)

    # Distinct variances and nonzero cross-covariances expose row/column misalignment.
    cov = np.diag(data.mean) + 0.1
    data.add_covariance(cov)
    path = tmp_path / "test.sacc"
    data.save_fits(path)
    return path


@pytest.mark.parametrize(
    "options, expected",
    [
        ({}, [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34]),
        ({PAIR_RANGE: "1.0 2.0"}, [1, 2, 3, 4, 11, 12, 21, 22, 23, 24, 31, 32, 33, 34]),
        ({COMMON_RANGE: "2.0 3.0"}, [2, 3, 12, 13, 21, 22, 23, 24, 31, 32, 33, 34]),
        ({COMMON_RANGE: "2.0 3.0", PAIR_RANGE: "1.0 2.0"},
         [2, 3, 11, 12, 21, 22, 23, 24, 31, 32, 33, 34]),
        ({ONE_POINT_RANGE: "2.0 3.0", f"{ONE_POINT_RANGE}_source_1": "3.0 4.0"},
         [1, 2, 3, 4, 11, 12, 13, 14, 22, 23, 33, 34]),
        ({ONE_POINT_RANGE: "2.0 3.0", LEGACY_RANGE: "3.0 4.0"},
         [1, 2, 3, 4, 11, 12, 13, 14, 22, 23, 33, 34]),
        ({"keep_tracers": "^source_0$", COMMON_RANGE: "2.0 3.0"},
         [2, 3, 21, 22, 23, 24]),
    ],
    ids=["no-cuts", "pair-only", "common", "pair-override", "one-point-override",
         "legacy-one-point-override", "keep-tracers"],
)
def test_scale_cuts(sacc_file, tmp_path, capsys, options, expected):
    ini_path = tmp_path / "pipeline.ini"
    settings = {
        "file": "likelihood/sacc/sacc_like.py",
        "data_file": str(sacc_file),
        **options,
    }
    ini_path.write_text(
        "[pipeline]\nmodules = sacc\n\n[sacc]\n"
        + "".join(f"{key} = {value}\n" for key, value in settings.items())
    )
    pipeline = Pipeline(cosmosis.Inifile(str(ini_path)))
    pipeline.setup()
    likelihood = pipeline.modules[0].data

    original = sacc.Sacc.load_fits(str(sacc_file))
    keep = np.isin(original.mean, expected)
    np.testing.assert_array_equal(likelihood.data_y, original.mean[keep])
    np.testing.assert_array_equal(
        likelihood.cov, original.covariance.dense[np.ix_(keep, keep)]
    )
    if LEGACY_RANGE in options:
        assert f"Use '{ONE_POINT_RANGE}_source_1' instead" in capsys.readouterr().out
