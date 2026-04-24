from cosmosis.datablock import names, option_section
from cosmosis.datablock import names as section_names
from cosmosis.gaussian_likelihood import SingleValueGaussianLikelihood
import numpy as np

class ThetaStarLikelihood(SingleValueGaussianLikelihood):
    #=====================================================
    # Theta star likelihood
    # Default theta_star value from Planck 2018 constraint 
    # (68 %, TT,TE,EE+lowE)
    #=====================================================

    # Where we should save the likelihood
    like_name = "thetastar"

    def build_data(self):
        # Defaults are currently from Planck 2018.
        theta_star = self.options.get_double("theta_star", default=0.0104109)
        theta_star_err = self.options.get_double("theta_star_err", default=0.000003)
        return theta_star, theta_star_err

    def extract_theory_points(self, block):
        theory_theta = block[section_names.distances,"THETASTAR"]/100
        return theory_theta


setup, execute, cleanup = ThetaStarLikelihood.build_module()