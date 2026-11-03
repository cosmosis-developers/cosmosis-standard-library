"""
Convert sampled combinations of galaxy bias and sigma_8 into the galaxy bias
parameters used by the rest of the pipeline.

This must run after a module that sets cosmological_parameters/sigma_8.
"""
from cosmosis.datablock import names, option_section

BIAS_MODELS = ["linear", "oneloop_eul_bk"]


def setup(options):
    number_lens_bins = options.get_int(option_section, "nbins", 6)
    bias_section = options.get_string(option_section, "bias_section", "bias_lens")
    bias_model = options.get_string(option_section, "bias_model", "oneloop_eul_bk")
    if bias_model not in BIAS_MODELS:
        raise ValueError(f"bias_model should be one of {BIAS_MODELS}, not '{bias_model}'")
    return number_lens_bins, bias_section, bias_model


def execute(block, config):
    number_lens_bins, bias_section, bias_model = config

    sigma_8 = block[names.cosmological_parameters, "sigma_8"]

    # Loop through bins, reading in b1_i * sigma_8 values, where b1_i is the
    # linear bias for bin i, and stopping at the first bin that is not present.
    for i in range(1, number_lens_bins + 1):
        if bias_model == "linear":
            b1_label = f"b{i}"
            b1_sig8_label = f"b{i}_sig8"
        else:
            b1_label = f"b1E_bin{i}"
            b1_sig8_label = f"b1E_sig8_bin{i}"

        if not block.has_value(bias_section, b1_sig8_label):
            print(f"{b1_sig8_label} not found")
            break

        block[bias_section, b1_label] = block[bias_section, b1_sig8_label] / sigma_8

    # The second-order bias is sampled as b2 * sigma_8^2
    if bias_model != "linear":
        for i in range(1, number_lens_bins + 1):
            b2_sig8sq_label = f"b2E_sig8sq_bin{i}"
            if not block.has_value(bias_section, b2_sig8sq_label):
                print(f"{b2_sig8sq_label} not found")
                break
            block[bias_section, f"b2E_bin{i}"] = block[bias_section, b2_sig8sq_label] / sigma_8**2

    return 0


def cleanup(config):
    pass
