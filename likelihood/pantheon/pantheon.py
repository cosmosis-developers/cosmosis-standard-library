"""
The likelihood module for the Pantheon dataset discards
much of the JLA/CosmoMC SN machinery, which is not needed
for Pantheon because the systematic errors in that dataset
have been subsumed into a single systematic covariance matrix.

In consequence almost all the terms in those codes are zero.

"""

from cosmosis.gaussian_likelihood import GaussianLikelihood
from cosmosis.datablock import names
import os
import numpy as np


# Default is to use the binned version of the data since it's much faster
# You can also downloaded and run the full data if you like, and set the data_file
# and covmat_file parameters in the ini file.
default_data_file = os.path.join(os.path.split(__file__)[0], "lcparam_DS17f.txt")
default_covmat_file = os.path.join(os.path.split(__file__)[0], "sys_DS17f.txt")


class PantheonLikelihood(GaussianLikelihood):
    x_section = names.distances
    x_name = "z"
    y_section = names.distances
    y_name = "mu"
    like_name = "pantheon"

    def __init__(self, options):
        super().__init__(options)
        self.marginalize_m = self.options.get_bool("marginalize_m", default=False)

        if self.marginalize_m:
            self.marginalize_c_term = self.inv_cov.sum()
            self.marginalize_const_term = 0.5 * np.log(2 * np.pi / self.marginalize_c_term)

    def build_data(self):
        """
        Run once at the start to load in the data vectors.

        Returns x, y where x is the independent variable (redshift in this case)
        and y is the Gaussian-distribured measured variable (magnitude in this case).

        """
        filename = self.options.get_string("data_file", default=default_data_file)
        print("Loading Pantheon data from {}".format(filename))

        # The Pantheon data format is mostly zeros.
        # The only columns that we actually need here are the redshift,
        # magnitude, and magnitude error.
        data = np.genfromtxt(filename).T
        z = data[1]
        m_obs = data[4]

        # We will need mag_obs_err later, when building the covariance,
        # so save it for now.
        self.mag_obs_err = data[5]

        # Return this to the parent class, which will use it
        # when working out the likelihood
        print("Found {} Pantheon supernovae (or bins if you used the binned data file)".format(len(z)))
        return z, m_obs

    def build_covariance(self):
        """Run once at the start to build the covariance matrix for the data"""
        filename = self.options.get_string("covmat_file", default=default_covmat_file)
        print("Loading Pantheon covariance from {}".format(filename))
        # The file format for the covariance has the first line as an integer
        # indicating the number of covariance elements, and the the subsequent
        # lines being the elements.
        # This data file is just the systematic component of the covariance - 
        # we also need to add in the statistical error on the magnitudes
        # that we loaded earlier
        f = open(filename)
        line = f.readline()
        n = int(line)
        C = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                C[i,j] = float(f.readline())

        # Now add in the statistical error to the diagonal
        for i in range(n):
            C[i,i] += self.mag_obs_err[i]**2
        f.close()

        # Return the covariance; the parent class knows to invert this
        # later to get the precision matrix that we need for the likelihood.
        return C

    def extract_theory_points(self, block):
        """
        Run once per parameter set to extract the mean vector that our
        data points are compared to.  In this case that means the theory
        prediction for the magnitudes.

        Note that because I'm not a supernova person I will use the absolute
        magnitude to change the *theory*, not to change the *data*.  This
        doesn't make any difference to the mathematics, but does maintain the
        conceptual distinction between "theory" and "observation" better
        than the usual way around.

        """
        import scipy.interpolate

        # Pull out mu and z from the block.
        # self.x_section etc. are defined above - we make them variables
        # so that the user can override them in the ini file.
        # We have to cut off the first element z=0, because mu is not finite
        # there and this confuses the interpolator.
        theory_x = block[self.x_section, self.x_name][1:]
        theory_y = block[self.y_section, self.y_name][1:]

        # This makes an interpolation function
        f = scipy.interpolate.interp1d(theory_x, theory_y, kind=self.kind)

        # Actually do the interpolation at the data redshifts
        theory = np.atleast_1d(f(self.data_x))


        if self.marginalize_m:
            # This is a bit of a hack to avoid having to modify the 
            # gaussian likelihood in cosmosis to support the marginalization
            # In likelihood_only=T mode the theory vector is currently
            # stored anywhere, but we need it for the marginalization
            # below.
            self.marge_theory_tmp = theory
        else:
            # Add the absolute supernova magnitude and return
            theory += block[names.supernova_params, "M"]

        return theory

    def do_likelihood(self, block):
        # Run the basic Gaussian likelihood
        super().do_likelihood(block)

        if not self.marginalize_m:
            return

        # Now apply the marginalization correction
        # log(P) -> log(P) - B^2 / C + D
        # where
        # B = I @ C^{-1} @ (d - mu)
        # C = I @ C^{-1} @ I
        # D = (1/2) log(2 pi / C)
        # where I is a vector of 1s

        B = (self.inv_cov @ (self.marge_theory_tmp - self.data_y)).sum()
        C = self.marginalize_c_term
        D = self.marginalize_const_term

        M_best_fit = - B / C
        change = -B ** 2 / C - 2 * D

        block[names.data_vector, self.like_name+"_CHI2"] += change
        block[names.likelihoods, self.like_name+"_LIKE"] += -0.5 * change
        block[names.supernova_params, "M"] = M_best_fit

        if self.likelihood_only:
            return

        # If we are also saving some additional data we modify that too
        block[names.data_vector, self.like_name + "_theory"] += M_best_fit
        if self.cov is not None:
            block[names.data_vector, self.like_name + "_simulation"] -= M_best_fit




# This takes our class and turns it into 
setup, execute, cleanup = PantheonLikelihood.build_module()
