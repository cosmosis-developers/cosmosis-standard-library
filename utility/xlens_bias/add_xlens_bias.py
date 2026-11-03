# Add Xlens to the model modifying the linear gal bias
# Based on Shivam Pandey's code for Y3
from cosmosis.datablock import names, option_section
import numpy as np

def setup(options):
    # Get the name of the bias section
    bias_section = options.get_string(option_section, "bias_section", "bias_lens")
	# Get the name of the xlens section
    r_section = options.get_string(option_section,"r_section", "x_lens")

    return bias_section, r_section

def execute(block, config):

    bias_section, r_section = config

    rval = block[r_section, 'rmean_bin']

    for i in range(1,6):
        b1in_label = "b1E_bin%d"%i
        b1_in = block[bias_section, b1in_label]
        b1out_gt_label = "b1gt_bin%d"%i
        b1out_wt_label = "b1wt_bin%d"%i
        if not block.has_value(bias_section, b1in_label):
            print("%s not found"%b1in_label)
            break
        block[bias_section, b1out_wt_label] = b1_in
        b1_gt = rval*b1_in
        block[bias_section, b1out_gt_label] = b1_gt

    return 0