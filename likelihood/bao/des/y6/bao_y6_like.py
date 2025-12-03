"""
DES Y6 BAO likelihood.

Note that this is not a Gaussian likelihood so we cannot use the standard
cosmosis machinery for those.

"""

from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
import numpy as np
import os

dist = section_names.distances
likes = section_names.likelihoods

ROOT_DIR = os.path.split(os.path.abspath(__file__))[0]


# combined measurement. Not actually used here.
MEAN = 0.9571
SIGMA = 0.0201

REDSHIFT = 0.851

# alpha likelihood file:
default_chi2_dir = ROOT_DIR
default_chi2_file = 'chi2profile_dvdesy6_cosmoplanck18_covcosmolike.csv'

def setup(options):
	section = option_section

	# Determine the path to the chi2(alpha) file and load it
	chi2_dir = options.get_string(section,"chi2_dir", default=default_chi2_dir)
	chi2_file = options.get_string(section,"chi2_file", default=default_chi2_file)
	chi2_path = os.path.join(chi2_dir, chi2_file)
	chi2_data = np.loadtxt(chi2_path, delimiter=',', skiprows=1).T

	# Select the part of the data we will actually use.
	# There are various columns denoting different methods for
	# measuring alpha
	chi2_column = options.get_int(section,"chi2_column", default=1)
	alpha = chi2_data[0]
	chi2_alpha = chi2_data[chi2_column]

	# Read the redshift at which to interpolate theory predictions
	redshift = options.get_double(section, "redshift", default=REDSHIFT)
	feedback = options.get_bool(option_section, "feedback", default=False)

	print("Limiting alpha = D_M / r_s values in interpolation:")
	print("alpha min = ", alpha[0])
	print("alpha max = ", alpha[-1])


	#Return data for later
	return (alpha, chi2_alpha, redshift, feedback)

def execute(block, config):

	alpha, chi2_alpha, redshift, feedback = config

	#Fiducial Planck cosmology rs which was used to compute alpha
	rs_fiducial =  147.8 
	# Angular diameter distance at the fiducial cosmology, d_a(zeff=0.851)
	d_a_fiducial = 1627.70 
	d_m_fiducial = d_a_fiducial * (1 + redshift)


	# load theory distance relations and R_S
	dist_z = block[dist, "z"]
	d_m = block[dist, 'd_m'] # Mpc
	h = block[dist, "h"] # 1/Mpc
	rs = block[dist, "rs_zdrag"] # Mpc

	# In case the redshifts are ordered chronologically instead
	# of numerically, reverse them. This is really a legacy from
	# a very old cosmosis change, but shouldn't hurt to check
	if dist_z[1] < dist_z[0]:
			dist_z = dist_z[::-1]
			d_m = d_m[::-1]
			h = h[::-1]

	# Interpolate the theory at the observed redshift
	d_m_predicted = np.interp(redshift, dist_z, d_m)

	# This ratio is what the likelihood is stored as a function of - the
	# ratio of dm/rs to the fiducial value
	alpha_predicted = (d_m_predicted / d_m_fiducial) * (rs_fiducial / rs)

	# Get the chi2 at the measured alpha, clipped to edges of the range
	chi2_alpha_predicted = np.interp(alpha_predicted, alpha, chi2_alpha, left=chi2_alpha[0], right=chi2_alpha[-1])

	#Get the log likelihood from the chi2
	like = -chi2_alpha_predicted / 2.
	block[likes, 'des_y6_bao_like'] = like

	#If required, print out some info
	if feedback:
		print("redshift = ", redshift)
		print("alpha predicted = ",  alpha_predicted)
		print("rs = ", rs)
		print("rs_fiducial = ", rs_fiducial)

	return 0


def cleanup(config):
    return 0