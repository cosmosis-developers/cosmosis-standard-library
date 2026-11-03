#Sample bias parameters via combinations with sigma_8
from cosmosis.datablock import names, option_section
import numpy as np

BIAS_MODELS = ["linear", "oneloop_eul_bk"]

def setup(options):
	number_lens_bins = options.get_int(option_section,
		"nbins",6)
	#Get the name of the bias section
	#By default look for 'bias_lens'
	bias_section = options.get_string(option_section,
		"bias_section", "bias_lens")
	#Get the name of the bias model
	#By default look for 'oneloop_eul_bk'
	bias_model = options.get_string(option_section,
		"bias_model", "oneloop_eul_bk")
	try:
		assert bias_model in BIAS_MODELS
	except AssertionError:
		print("bias model should be in: "%BIAS_MODELS)
		raise(e)
	return number_lens_bins, bias_section, bias_model

def execute(block, config):

	number_lens_bins, bias_section, bias_model = config

	#Read in sigma_8
	sigma_8 = block[names.cosmological_parameters, "sigma_8"]

	#nbins = 0
	#loop through bins, reading in b1_i * sigma_8 values,
	#where b1_i is the linear bias for bin i
	for i in range(1,number_lens_bins+1):
		if bias_model == "linear":
			b1_label = "b%d"%i
			b1_sig8_label = "b%d_sig8"%i
		else:
			b1_label = "b1E_bin%d"%i
			b1_sig8_label = "b1E_sig8_bin%d"%i
		if not block.has_value(bias_section, b1_sig8_label):
			print("%s not found"%b1_sig8_label)
			break
		b1_i_sig8 = block[bias_section, b1_sig8_label]
		b1_i = b1_i_sig8 / sigma_8
		block[bias_section, b1_label] = b1_i
		#nbins += 1
	# if nbins == 0:
	# 	raise ValueError("we found zero bins, this seems wrong so let's raise an error")

	if bias_model != "linear":
		for i in range(1,number_lens_bins+1):
			b2_label = "b2E_bin%d"%i
			b2_sig8sq_label = "b2E_sig8sq_bin%d"%i
			b2_sig8sq_i = block[bias_section, b2_sig8sq_label]
			b2_i = b2_sig8sq_i / sigma_8 / sigma_8
			block[bias_section, b2_label] = b2_i

	return 0


