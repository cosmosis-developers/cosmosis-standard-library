import os
from numpy import log, pi, interp, where, loadtxt,dot, append, linalg
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
from cosmosis.gaussian_likelihood import GaussianLikelihood
from scipy.interpolate import interp2d

dist = section_names.distances

c_km_per_s =  299792.458
default_rd_fiducial = 147.78

ROOT_dir = os.path.split(os.path.abspath(__file__))[0]


class DESY6BAO_likelihood_5bins(GaussianLikelihood):
	like_name = 'des_y6_bao_5bins'
	
	def __init__(self, options):
		
		super(DESY6BAO_likelihood_5bins, self).__init__(options)
		# Allow override of these parameters
		self.rd_fiducial = self.options.get_double('rd_fiducial', default_rd_fiducial)
		self.feedback = self.options.get_bool('feedback', default=False)
	
	def build_data(self):
		
		# Reading data file
		DATA_file = os.path.join(ROOT_dir, 'bao_y6_5bins.txt')
		
		DATA = loadtxt(DATA_file, usecols=(0,1))
		z_eff, data = DATA[:,0], DATA[:,1]
		
		return z_eff, data
	
	def build_covariance(self):
		
		# Reading covariance matrix file
		COV_file = os.path.join(ROOT_dir, 'cov_rescaled_5bins.txt')
		
		cov = loadtxt(COV_file)
		self.inv_cov = linalg.inv(cov)
		
		return cov
	
	def build_inverse_covariance(self):
		return self.inv_cov
	
	def extract_theory_points(self,block):   
		
		# Redshift array
		z = block[dist, 'z']
		# Sound horizon at the drag epoch
		rd = block[dist, 'rs_zdrag']
		# Comoving distance
		Dm = block[dist, 'd_m'] # in Mpc
		
		# z and distance maybe are loaded in chronological order
		# Reverse to start from low z
		if (z[1] < z[0]):
			z = z[::-1]
			Dm = Dm[::-1]
		
		#Find theory Dm at effective redshift by interpolation
		z_eff = self.data_x
		
		# Distances over rd
		Dm_z_rd = interp(z_eff, z, Dm)/rd
		params = Dm_z_rd
		
		if self.feedback:
			print()
			print('             zeff   pred    data')
			print('Dm_over_rd: %.3f  %.3f  %.3f' % (self.data_x[0], Dm_z_rd[0], self.data_y[0]))
			print('Dm_over_rd: %.3f  %.3f  %.3f' % (self.data_x[1], Dm_z_rd[1], self.data_y[1]))
			print('Dm_over_rd: %.3f  %.3f  %.3f' % (self.data_x[2], Dm_z_rd[2], self.data_y[2]))
			print('Dm_over_rd: %.3f  %.3f  %.3f' % (self.data_x[3], Dm_z_rd[3], self.data_y[3]))
			print('Dm_over_rd: %.3f  %.3f  %.3f' % (self.data_x[4], Dm_z_rd[4], self.data_y[4]))
			print()
		
		return params


setup, execute, cleanup = DESY6BAO_likelihood_5bins.build_module()