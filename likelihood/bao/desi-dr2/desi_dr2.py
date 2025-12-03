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
        "mean": 7.944,
        "sigma": 0.075,
    },
    "LRG1": {
        "kind": "d_m_d_h",
        "z_eff": 0.51,
        "mean": [13.587, 21.863],
        "sigma": [0.169, 0.427],
        "corr": -0.475,
    },
    "LRG2": {
        "kind": "d_m_d_h",
        "z_eff": 0.706,
        "mean": [17.347, 19.458],
        "sigma": [0.180, 0.332],
        "corr": -0.423,
    },
    "LRG3+ELG1": {
        "kind": "d_m_d_h",
        "z_eff": 0.934,
        "mean": [21.574, 17.641],
        "sigma": [0.153, 0.193],
        "corr": -0.425,
    },
    "ELG2": {
        "kind": "d_m_d_h",
        "z_eff": 1.321,
        "mean": [27.605, 14.178],
        "sigma": [0.320, 0.217],
        "corr": -0.437,
    },
    "QSO": {
        "kind": "d_m_d_h",
        "z_eff": 1.484,
        "mean": [30.519,12.816],
        "sigma": [0.758,0.513],
        "corr": -0.489
    },
    "Lya": {
        "kind": "d_m_d_h",
        "z_eff": 2.330,
        "mean": [38.988, 8.632],
        "sigma": [0.531, 0.101],
        "corr": -0.431,
    },
    # LRG3 and ELG1 are combined in the data release
    # into LRG3+ELG1, above.  We allow them to be separately
    # used below but throw an error if they are both used.
    "LRG3": {
        "kind": "d_m_d_h",
        "zeff": 0.922,
        "mean": [21.649, 17.574],
        "sigma": [0.177,  0.214],
        "corr": -0.408
    },
    "ELG1": {
        "kind": "d_m_d_h",
        "zeff": 0.922,
        "mean": [21.708, 17.811],
        "sigma": [0.337,   0.295],
        "corr": -0.452,

    }


}

class DESILikelihood(GaussianLikelihood):
    """
    The DR2 2025 DESI likelihoods from https://arxiv.org/pdf/2503.14738

    We allow the user to specify which data sets to use, and combine
    them all into one. The data sets are:
    - BGS
    - LRG1
    - LRG2
    - LRG3+ELG1
    - ELG2
    - QSO
    - Lya

    The LRG3 and ELG1 data sets are also available separately, but
    cannot be used jointly with the LRG3+ELG1 data set. The default "all"
    selection leaves out the two individual data sets and just uses the combined one.
    """
    # users can override this if they want to use a different name
    # which can be useful if you want to keep the different likelihoods
    # separately.
    like_name = "desi_bao"
    x_section = 'distances'
    x_name = 'z'
    y_section = 'distances'

    def __init__(self, options):
        data_sets = options.get_string("desi_data_sets", default="all")
        if data_sets == "all":
            data_sets = list(DESI_DATA_SETS.keys())
            data_sets.remove("ELG1")
            data_sets.remove("LRG3")
        else:
            data_sets = data_sets.split(',')

        if ("ELG1" in data_sets or "LRG3" in data_sets) and "LRG3+ELG1" in data_sets:
            raise ValueError("You cannot use both DESI's LRG3+ELG1 and ELG1 or LRG3")

        allowed = list(DESI_DATA_SETS.keys())
        for data_set in data_sets:
            data_set = data_set.strip()
            if data_set not in allowed:
                raise ValueError(f"Unknown DESI data set {data_set}. Valid options are: {allowed} (comma-separated to use more than one) or 'all'")
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
                z.append(ds["z_eff"])

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

        block["distances", "H0rd"] = block["cosmological_parameters", "H0"] * r_s

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
