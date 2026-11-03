#Sample IA redshift dependence through amplitude*redshift dependence
from cosmosis.datablock import names, option_section
import numpy as np


def execute(block):

	#Read in ia parameters
	A1 = block[names.intrinsic_alignment_parameters, "A1"]
	A2 = block[names.intrinsic_alignment_parameters, "A2"]

	ia_section = "intrinsic_alignment_parameters"

	alphaA1 =  block[ia_section,"alphaA1"]
	alphaA2 =  block[ia_section,"alphaA2"]

	block[ia_section, "alpha1"] = alphaA1/A1
	block[ia_section, "alpha2"] = alphaA2/A2

	return 0


