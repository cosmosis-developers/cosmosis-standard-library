from numpy import log, pi, interp, where, loadtxt,dot, append, linalg
import os
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
from cosmosis.gaussian_likelihood import SingleValueGaussianLikelihood

dist = section_names.distances

c_km_per_s =  299792.458


class Eboss14LRGLikelihood(SingleValueGaussianLikelihood):
    data_type = 'LRG'
    like_name = "eboss14_lrg"
    mean = 2377.0
    # in bautista et al, 2018, sigma_p = 61, sigma_m = 59
    #  here we assume symetric error bars with sigma = 60
    sigma = 60.0
    fiducial_z = 0.72
    fiducial_rd = 147.78


    def __init__(self, options):
        super(Eboss14LRGLikelihood, self).__init__(options)
        # Allow override of these parameters
        if options.has_value("fiducial_z"):
            self.fiducial_z = options['fiducial_z']
        if options.has_value("fiducial_rd"):
            self.fiducial_rd = options['fiducial_rd']

        self.verbose = options.get_bool('verbose', False)


    def extract_theory_points(self, block):
        z = block[dist, 'z']
        rd = block[dist, "rs_zdrag"]
        d_m = block[dist, 'd_m']

        h = c_km_per_s * block[dist, 'h']
        d_h = c_km_per_s / h

        #computation of d_v (see e.g. eq 12 of Bautista et al 2018)
        d_v = (z * d_m**2. * d_h)**(1./3.)

        # Find theory D_v at fiducial redshift by interpolation
        dv_z = interp(self.fiducial_z, z, d_v)

        # Find theory value of observable
        dv_z_rd_predicted = dv_z * (self.fiducial_rd / rd)

        if self.verbose:
            print("For {}, measured D_v*rd_fid/rd at redshift {} = {}".format(
                self.data_type, self.fiducial_z, dv_z_rd_predicted))

        return dv_z_rd_predicted


setup, execute, cleanup = Eboss14LRGLikelihood.build_module()