from cosmosis.datablock import  names
from cosmosis import GaussianLikelihood
cosmo = names.cosmological_parameters
import numpy as np


modes = {
    # Taken from:
    # https://github.com/CosmoLike/DESC_SRD/blob/f3b44070e40cbdc1a459e07a1a8eb31d2608af6f/cosmolike_lite/theory/external_prior.c#L517C8-L517C48

    # Add more entries here from that file if needed,
    # and then you can set the "mode" parameter in the ini file
    # section for this module.

    "Planck18_BAO_Riess18_Pantheon_w0wa": {
        "param_names": [
            "omega_m",
            "sigma_8",
            "n_s",
            "w",
            "wa",
            "omega_b",
            "h0",
        ],
        #setting the fiducial values to the values from Make_Y1_and_Y10_Mocks/baseline-mock-values.ini
        "fiducial": {
                "omega_m":  0.3,
                "sigma_8":  0.8 ,
                "n_s":  0.97 ,
                "w":  -1.0,
                "wa":  0.0,
                "omega_b":  0.0437 ,
                "h0": 0.715,
        },
        "inv_cov": np.array([
                [ 4.30741e+06, -6.29653e+05,  2.30239e+05,  2.40526e+05, 5.75393e+04, -1.09309e+07,  3.63645e+06],
                [-6.29653e+05,  1.84962e+05, -4.30980e+04,  9.10127e+03, 3.25867e+03,  1.38575e+06, -4.94260e+05],
                [ 2.30239e+05, -4.30980e+04,  1.26977e+05, -7.56824e+03, -2.11814e+03, -4.02713e+05,  1.58351e+05],
                [ 2.40526e+05,  9.10127e+03, -7.56824e+03,  4.14100e+04, 1.04261e+04, -9.36525e+05,  2.14609e+05],
                [ 5.75393e+04,  3.25867e+03, -2.11814e+03,  1.04261e+04, 2.64176e+03, -2.28031e+05,  5.15943e+04],
                [-1.09309e+07,  1.38575e+06, -4.02713e+05, -9.36525e+05, -2.28031e+05,  4.94180e+07, -7.29198e+06],
                [ 3.63645e+06, -4.94260e+05,  1.58351e+05,  2.14609e+05, 5.15943e+04, -7.29198e+06,  3.34690e+06]
            ])
    }
}


class Stage3Priors(GaussianLikelihood):
    like_name = "stage3"
    def build_data(self):
        #if we want to include other scenarios we can make this an
        # option in the ini file and select from the variables above.
        mode = self.options.get_string("mode", default="Planck18_BAO_Riess18_Pantheon_w0wa")
        self.prior_data = modes[mode]
        self.param_names = self.prior_data["param_names"]
        #TODO - would make more sense to fix the fiducial values from an input values.ini file instead of having them hardwired?
        fid = self.prior_data["fiducial"]
        p0 = np.array([fid[name] for name in self.param_names])
        return None, p0
    
    def build_inverse_covariance(self):
        return self.prior_data["inv_cov"]

    def build_covariance(self):
        return np.linalg.pinv(self.prior_data["inv_cov"], hermitian=True)

    def extract_theory_points(self, block):
        return np.array([block[cosmo, name] for name in self.param_names])


setup, execute, cleanup = Stage3Priors.build_module()
