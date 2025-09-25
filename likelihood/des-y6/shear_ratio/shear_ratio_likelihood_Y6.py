from cosmosis.datablock import option_section, names
import numpy as np
from scipy.interpolate import interp1d
from cosmosis.gaussian_likelihood import GaussianLikelihood
import pickle



def get_ratio_from_gammat(gammat1, gammat2, inv_cov):
    #Given two gammats, calculate the ratio
    s2 = (1./float(np.matrix(np.ones(len(gammat1)))*np.matrix(inv_cov)*np.matrix(np.ones(len(gammat1))).T))
    ratio = s2*float(np.matrix(gammat1/gammat2)*np.matrix(inv_cov)*np.matrix(np.ones(len(gammat1))).T)
    return ratio

def radians_to_arcmin(r):
    return np.degrees(r) * 60.0

class ShearRatioLikelihood(GaussianLikelihood):
    y_section = "galaxy_shear_xi"
    like_name = "shear_ratio"
    def __init__(self, options):

        filename = options["data_file"]
        with open(filename, "rb") as f:
            self.ratio_data = pickle.load(f)

        # this adds the option to save the theory shear ratio data vector
        self.save_simulated_ratios = options.get_bool("save_simulated_ratios",default=False)
        self.output_name = options.get_string("output_file",default="hello.pkl")

        nbin_source = self.ratio_data['nbin_source']  # [3,4] an array of the exact source bins we have
        nbin_lens = self.ratio_data['nbin_lens'] # 2, because we don't use all the lens bins
        nratios_per_lens = self.ratio_data['nratios_per_lens'] # 2, because there are 1 independent ratios we can construct given 2 source bins, per each lens bin, (3,4) 

        self.theta = self.ratio_data['theta_data']
       
        # Options on masking the different parts
        # theta_min is defined per each lens-source bin ratio. 
        theta_max = options.get_double_array_1d(f"theta_max")
        theta_min = [
            options.get_double_array_1d(f"theta_min_{i+1}") #From Y3 file it change nbin_lens to nratios_per_lens
            for i in range(nratios_per_lens)
        ]


        # Generate scale masks for each bin pair.
        # These are used in the calculation of each ratio from the range of points
        self.masks = {}
        for sc in range(1, nratios_per_lens + 1):
            for l in range(1, nbin_lens + 1):
                t_min = theta_min[sc - 1][l - 1]
                t_max = theta_max[l - 1]
                self.masks[(sc, l)] = (self.theta > t_min) & (self.theta <= t_max)

        self.inv_cov_individual_ratios = {}


        ind_cov_data = self.ratio_data['inv_cov_individual_ratios']

        n = 0
        for l in range(1, nbin_lens + 1):
            for sc in range(1, nratios_per_lens + 1):
                mask = self.masks[sc, l]
                P = ind_cov_data[n][mask][:, mask]
                self.inv_cov_individual_ratios[sc, l] = P
                n += 1

        self.nbin_lens = nbin_lens
        self.nbin_source = nbin_source
        super().__init__(options)


    def build_data(self):
        return self.theta, self.ratio_data['measured_ratios']

    def build_covariance(self):
        return self.ratio_data['ratio_cov']


    def extract_theory_points(self, block):

        bin_avg = block[self.y_section, 'bin_avg']

        s_ref = max(self.nbin_source)
        theory_ratios = []

        # Load the separations at which the gamma_t values are
        # computed and convert to 
        theta = radians_to_arcmin(block[self.y_section, 'theta'])

        # This helper function gets a gamma_t value from the block
        # and then optionally interpolates it, depending on whether
        # it was already calculated at bin-averaged positions or at
        # a wide range of theta values
        def get_gamma_t(s, l):

            gamma_t = block[self.y_section, f'bin_{l}_{s}']

            if not bin_avg:
                gamma_t = interp1d(theta, gamma_t)(self.x) 
            return gamma_t

        
        for l in range(1, self.nbin_lens + 1):
            gammat_ref = get_gamma_t(s_ref, l)

            for s in range(1,len(self.nbin_source[:-1])+1):
                mask = self.masks[s, l]
                mask = self.masks[s, l]
                P = self.inv_cov_individual_ratios[s, l]
                gamma_t = get_gamma_t(self.nbin_source[s-1], l)
                ratio = get_ratio_from_gammat(gamma_t[mask], gammat_ref[mask], P)
                theory_ratios.append(ratio)
                

        if self.save_simulated_ratios:
            # copy the format of the input SR file
            output_data = self.ratio_data

            # overwrite the data bit - assuming we're keeping the cov mat, only measured_ratios needs to change
            output_data['measured_ratios'] = np.array(theory_ratios)


            # now write the file
            with open(self.output_name, 'wb') as f:
                pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)


        return np.array(theory_ratios)

setup, execute, cleanup = ShearRatioLikelihood.build_module()
