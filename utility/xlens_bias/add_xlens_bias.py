"""
Add xlens to the model by modifying the linear galaxy bias.

Based on Shivam Pandey's code for Y3.
"""
from cosmosis.datablock import option_section


def setup(options):
    bias_section = options.get_string(option_section, "bias_section", "bias_lens")
    r_section = options.get_string(option_section, "r_section", "x_lens")
    nbins = options.get_int(option_section, "nbins", 5)
    return bias_section, r_section, nbins


def execute(block, config):
    bias_section, r_section, nbins = config

    rval = block[r_section, "rmean_bin"]

    for i in range(1, nbins + 1):
        b1_in_label = f"b1E_bin{i}"
        if not block.has_value(bias_section, b1_in_label):
            print(f"{b1_in_label} not found")
            break
        b1_in = block[bias_section, b1_in_label]

        # Galaxy clustering uses the original bias, galaxy-galaxy lensing the scaled one
        block[bias_section, f"b1wt_bin{i}"] = b1_in
        block[bias_section, f"b1gt_bin{i}"] = rval * b1_in

    return 0


def cleanup(config):
    pass
