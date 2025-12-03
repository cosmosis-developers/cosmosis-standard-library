from cosmosis.datablock import names
from cosmosis.gaussian_likelihood import GaussianLikelihood
import planck_lite_py
import numpy as np
import os

dirname = os.path.split(__file__)[0]

class PlanckPythonLikelihood(GaussianLikelihood):
    x_section = 'cmb_cl'
    x_name = 'ell'
    y_section = 'cmb_cl'
    like_name = 'planck'

    def build_data(self):
        year = self.options['year']
        spectra = self.options['spectra']
        use_low_ell_bins = self.options['use_low_ell_bins']

        # Use the local directory by default
        data_directory = os.path.join(dirname, 'data')
        data_directory = self.options.get_string('data_directory', data_directory)

        ell_max_tt = self.options.get_int('ell_max_tt', -1)
        ell_max_te = self.options.get_int('ell_max_te', -1)
        ell_max_ee = self.options.get_int('ell_max_ee', -1)
        if ell_max_tt == -1:
            ell_max_tt = None
        if ell_max_te == -1:
            ell_max_te = None
        if ell_max_ee == -1:
            ell_max_ee = None

        self.calculator = planck_lite_py.PlanckLitePy(year=year, spectra=spectra, 
            use_low_ell_bins=use_low_ell_bins, data_directory=data_directory,
            ell_max_tt=ell_max_tt, ell_max_te=ell_max_te, ell_max_ee=ell_max_ee)
        x = None

        #We optionally allow the user to provide a cosmosis
        #test sampler output directory here, which will be
        #used to set the CMB datavector. This allows for 
        #generating Planck like constraints at a cosmology 
        #of your choice.
        use_data_from_test = self.options.get_string(
            "use_data_from_test",
            "")
        if use_data_from_test != "":
            print("using cmb data from %s"%use_data_from_test)
            y = self.get_data_points_from_test_output(use_data_from_test)
        else:
            y = self.calculator.data_vector

        self.effective_ell = self.calculator.effective_ells
        self.spectra = self.calculator.spectra
        return x, y


    def get_data_points_from_test_output(self, test_output_dir):
        cmb_dir = os.path.join(test_output_dir, "cmb_cl")
        ell = np.loadtxt(os.path.join(cmb_dir, "ell.txt")).astype(int)
        tt = np.loadtxt(os.path.join(cmb_dir, 'tt.txt'))
        te = np.loadtxt(os.path.join(cmb_dir, 'te.txt'))
        ee = np.loadtxt(os.path.join(cmb_dir, 'ee.txt'))

        y = self.calculator.make_mean_vector(tt, te, ee, ellmin=ell.min())
        return y

    def build_covariance(self):
        return self.calculator.cov

    def build_inverse_covariance(self):
        return self.calculator.fisher

    def extract_theory_points(self, block):
        ell = block[self.x_section, self.x_name]
        ellmin = ell[0]

        Dltt = block[self.y_section, 'TT']
        Dlte = block[self.y_section, 'TE']
        Dlee = block[self.y_section, 'EE']

        try:
            y = self.calculator.make_mean_vector(Dltt, Dlte, Dlee, ellmin=ellmin)
        except ValueError:
            raise ValueError("CMB spectra not calculated to high enough ell for chosen Planck settings")

        block["data_vector", self.like_name + "_ell"] = self.effective_ell
        block["data_vector", self.like_name + "_spectra"] = self.spectra

        return y

setup, execute, cleanup = PlanckPythonLikelihood.build_module()
