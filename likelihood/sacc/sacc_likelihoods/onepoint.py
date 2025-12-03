import numpy as np
from cosmosis.datablock import BlockError
from scipy.interpolate import interp1d
import pathlib
import sys

def extract_one_point_prediction(sacc_data, block, data_type, section, **kwargs):
    category = kwargs["category"]
    tracer_tuples = sacc_data.get_tracer_combinations(data_type)

    theory_vector = []
    observable_min_vector = []
    observable_max_vector= []
    bins_vector = []

    for t in tracer_tuples:
        assert len(t) == 1, "One-point likelihoods only support single tracer data types"
        # This should be an n(z) tracer
        tracer = t[0]
        b = int(tracer.split("_")[1]) + 1
        
        # Determine the key for accessing block and sacc_data based on category
        # This selection below needs to be generalised more 1pt statistics we implement (cluster richness for instance, ...).
        key = "mass" if category == "one_point_mass" else "luminosity"
        x_theory = block[section, f"{key}_{b}"]
        theory = block[section, f"bin_{b}"]
        theory_spline = interp1d(x_theory, theory, bounds_error=False, fill_value="extrapolate")

        window = None
        for w in sacc_data.get_tag("windows", data_type, t):
            if (window is not None) and (w is not window):
                raise ValueError("Sacc likelihood currently assumes data types share a window object")
            window = w

        # TO-DO: Check if the window thing is ok for 1pt stats and how to do the binning here either way.
        x_nominal = np.array(sacc_data.get_tag(key, data_type, t))
        
        if window is None:
            binned_theory = theory_spline(x_nominal)
        else:
            x_window = window.values
            theory_interpolated = theory_spline(x_window)
            index = sacc_data.get_tag("window_ind", data_type, t)
            weight = window.weight[:, index]

            # TO-DO: Check this for real statistics, but should be ok.
            # We don't automatically renormalize the weights.
            # Some contexts, like the output from NaMaster,
            # use non-unit-sum weights
            binned_theory = (weight @ theory_interpolated)

        theory_vector.append(binned_theory)
        observable_min_vector.append(sacc_data.get_tag(f"{key}_min", data_type, t))
        observable_max_vector.append(sacc_data.get_tag(f"{key}_max", data_type, t))
        bins_vector.append(np.repeat(b, len(binned_theory)))

    theory_vector = np.concatenate(theory_vector)

    metadata = {
        f"{key}_min": np.concatenate(observable_min_vector),
        f"{key}_max": np.concatenate(observable_max_vector),
        f"{key}_bin": np.concatenate(bins_vector),
        f"{key}_theory": theory_vector,
    }
    return theory_vector, metadata





    


