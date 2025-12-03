from cosmosis.gaussian_likelihood import GaussianLikelihood
import numpy as np
import scipy.interpolate

# The three different types of measurement
# of BAO used in this data release
KIND_DV = 1
KIND_DM = 2
KIND_DH = 3


# Data from table 1 of https://arxiv.org/pdf/2404.03002
DESI_DATA_SETS = {
    "BGS": {
        "kind": "d_v",
        "z_eff": 0.295,
        "mean": 7.93,
        "sigma": 0.15,
    },
    "LRG1": {
        "kind": "d_m_d_h",
        "z_eff": 0.51,
        "mean": [13.62, 20.98],
        "sigma": [0.25, 0.61],
        "corr": -0.445,
    },
    "LRG2": {
        "kind": "d_m_d_h",
        "z_eff": 0.706,
        "mean": [16.85, 20.08],
        "sigma": [0.32, 0.60],
        "corr": -0.420,
    },
    "LRG3+ELG1": {
        "kind": "d_m_d_h",
        "z_eff": 0.930,
        "mean": [21.71, 17.88],
        "sigma": [0.28, 0.35],
        "corr": -0.389,
    },
    "ELG2": {
        "kind": "d_m_d_h",
        "z_eff": 1.317,
        "mean": [27.79, 13.82],
        "sigma": [0.69, 0.42],
        "corr": -0.444,
    },
    "QSO": {
        "kind": "d_v",
        "z_eff": 1.491,
        "mean": 26.07,
        "sigma": 0.67,
    },
    "Lya QSO": {
        "kind": "d_m_d_h",
        "z_eff": 2.330,
        "mean": [39.71, 8.52],
        "sigma": [0.94, 0.17],
        "corr": -0.477,
    },
}

class DESILikelihood(GaussianLikelihood):
    """
    The 2024 DESI likelihoods from https://arxiv.org/pdf/2404.03002

    We allow the user to specify which data sets to use, and combine
    them all into one. The data sets are:
    - BGS
    - LRG1
    - LRG2
    - LRG3+ELG1
    - ELG2
    - QSO
    - Lya QSO

    """
    # users can override this if they want to use a different name
    # which can be useful if you want to keep the different likelihoods
    # separately.
    like_name = "desi_bao"
    x_section = 'distances'
    x_name = 'z'
    y_section = 'distances'

    def __init__(self, options):
        data_sets = options.get_string("desi_data_sets")
        data_sets = data_sets.split(',')

        allowed = list(DESI_DATA_SETS.keys())
        for data_set in data_sets:
            data_set = data_set.strip()
            if data_set not in allowed:
                raise ValueError(f"Unknown DESI data set {data_set}. Valid options are: {allowed} (comma-separated to use more than one)")
        self.data_sets = data_sets
        super().__init__(options)
    

    def build_data(self):
        z = []
        mu = []
        kinds = []
        for name in self.data_sets:
            ds = DESI_DATA_SETS[name]

            # collect the effective redshfits for the measurements
            z.append(ds["z_eff"])

            # The d_v type measurements are just a single number
            # but the d_m_d_h measurements are two values
            if ds["kind"] == "d_v":
                mu.append(ds["mean"])
                kinds.append(KIND_DV)
            else:
                mu.extend(ds["mean"])
                kinds.append(KIND_DM)
                kinds.append(KIND_DH)
                # This makes the z array the same length
                # as the mu array. But because the D_M and D_H
                # measurements are at the same redshift we only
                # need to store the redshift once, and this should
                # hopefully trigger an error if we mess up later.
                z.append(-1.0)

        kinds = np.array(kinds)
        z = np.array(z)
        mu = np.array(mu)

        # record the indices of the d_v and d_m_d_h measurements
        # for later
        self.dv_index = np.where(kinds==KIND_DV)[0]
        self.dm_index = np.where(kinds==KIND_DM)[0]
        self.dh_index = np.where(kinds==KIND_DH)[0]

        self.any_dv = len(self.dv_index) > 0
        self.any_dmdh = len(self.dm_index) > 0

        return z, mu

    def build_covariance(self):
        n = len(self.data_x)
        C = np.zeros((n, n))
        i = 0
        for name in self.data_sets:
            ds = DESI_DATA_SETS[name]
            if ds["kind"] == "d_v":
                C[i, i] = ds["sigma"]**2
                i += 1
            else:
                C[i, i] = ds["sigma"][0]**2
                C[i+1, i+1] = ds["sigma"][1]**2
                C[i, i+1] = C[i+1, i] = ds["corr"]*ds["sigma"][0]*ds["sigma"][1]
                i += 2
        return C

    def extract_theory_points(self, block):
        z_theory = block[self.x_section, self.x_name]
        y = np.zeros(self.data_x.size)
        r_s = block[self.y_section, "rs_zdrag"]

        block["distances", "h0rd"] = block["cosmological_parameters", "h0"] * r_s

        if self.any_dv:
            d_v = block[self.y_section, "d_v"]
            z_data = self.data_x[self.dv_index]
            f = scipy.interpolate.interp1d(z_theory, d_v/r_s, kind=self.kind)
            y[self.dv_index] = f(z_data)

        if self.any_dmdh:
            z_data = self.data_x[self.dm_index]

            d_m = block[self.y_section, "d_m"]
            f = scipy.interpolate.interp1d(z_theory, d_m/r_s, kind=self.kind)
            y[self.dm_index] = f(z_data)

            d_h = 1.0 / block[self.y_section, "h"]
            f = scipy.interpolate.interp1d(z_theory, d_h/r_s, kind=self.kind)
            y[self.dh_index] = f(z_data)
        return y

setup, execute, cleanup = DESILikelihood.build_module()
