import numpy as np
from cosmosis.datablock import names
from cosmosis.gaussian_likelihood import SingleValueGaussianLikelihood


class BBNLikelihood(SingleValueGaussianLikelihood):
    # The mean and standard deviation of the BBN measurements.
    # The user can over-ride these in the ini file if desired

    # The value (either chosen by the sampler or computed
    # somewhere in the pipeline) that this Likelihood uses.
    # If we needed to operate on a derived quantity we could
    # have redefined the method block_theory_points instead
    section = names.cosmological_parameters

    # Where we should save the likelihood
    like_name = "bbn"

    def build_data(self):

        paper = self.options.get_string("paper", "beringer_2012")        

        if paper == "beringer_2012":
            print("Using Beringer 2012 (PDG) BBN measurement")
            mean = 0.023
            sigma = 0.002
        elif paper == "cooke_2016_i":
            print("Using Cooke 2016 BBN measurement (version i)")
            mean = 2.260 / 100
            sigma = 0.034 / 100
        elif paper == "cooke_2016_ii":
            print("Using Cooke 2016 BBN measurement (version ii)")
            mean = 2.156 / 100
            sigma = 0.020 / 100
        elif paper == "cooke_2016_combined":
            print("Using Cooke 2016 BBN measurement (combined version)")
            mean = 0.02208
            sigma = 0.00052
        elif paper == "pitrou_cooke_combined":
            print("Using Pitrou 2020 + Cooke 2018 BBN measurement")
            luna_mn = 0.02195
            luna_sd = 0.00022

            cooke18_theory_mn = 0.02166
            cooke18_theory_sd = 0.00019

            invcov_weighted_mn = (luna_mn/luna_sd**2 + cooke18_theory_mn/cooke18_theory_sd**2) / (1/luna_sd**2 + 1/cooke18_theory_sd**2)

            sd_stat = luna_sd
            sd_sys = luna_mn - invcov_weighted_mn
            mean = luna_mn
            sigma = np.sqrt(sd_stat**2 + sd_sys**2)
        else:
            raise ValueError("Unknown 'paper' parameter value in bbn_ombh2. Must be "
                             "one of beringer_2012, cooke_2016_i, cooke_2016_ii, "
                             "cooke_2016_combined, pitrou_cooke_combined.")

        print("Dataset mean = ", mean, " and sigma = ", sigma)

        if self.options.has_value("mean"):
            mean = options.get_float("mean")
            print("** Overriding mean with param file value: ", mean)
        if self.options.has_value("sigma"):
            sigma = options.get_float("sigma")
            print("** Overriding sigma with param file value: ", sigma)

        return mean, sigma        

    def extract_theory_points(self, block):
        omega_b = block[self.section, 'omega_b']
        h = block[self.section, 'h0']
        ombh2 = omega_b * h**2
        return ombh2


setup, execute, cleanup = BBNLikelihood.build_module()
