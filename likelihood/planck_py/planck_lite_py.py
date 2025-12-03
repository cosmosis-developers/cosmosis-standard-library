"""
This is a rewrite by JZ of the Prince & Dunkeley python version of the Planck likelihood code,
to make it a little simpler to apply ell cuts to the data, and to make the data files
more self-descriptive.

Python version of Planck's plik-lite likelihood with the option to include
the low-ell temperature as two Gaussian bins

The official Planck likelihoods are availabe at https://pla.esac.esa.int/
The papers describing the Planck likelihoods are
Planck 2018: https://arxiv.org/abs/1907.12875
Planck 2015: https://arxiv.org/abs/1507.02704

The covariance matrix treatment is based on Zack Li's ACT likelihood code
available at: https://github.com/xzackli/actpols2_like_py

planck calibration is set to 1 by default but this can easily be modified
"""

import numpy as np
from astropy.table import Table
import scipy.linalg


def main():
    TTTEEE2018 = PlanckLitePy(year=2018, spectra="TTTEEE", use_low_ell_bins=False)
    TTTEEE2018.test()

    TTTEEE2018_lowTTbins = PlanckLitePy(year=2018, spectra="TTTEEE", use_low_ell_bins=True)
    TTTEEE2018_lowTTbins.test()

    TT2018 = PlanckLitePy(year=2018, spectra="TT", use_low_ell_bins=False)
    TT2018.test()

    TT2018_lowTTbins = PlanckLitePy(year=2018, spectra="TT", use_low_ell_bins=True)
    TT2018_lowTTbins.test()


class PlanckLitePy:
    def __init__(
        self,
        data_directory="data",
        year=2018,
        spectra="TT",
        use_low_ell_bins=False,
        ell_max_tt=None,
        ell_max_te=None,
        ell_max_ee=None,
    ):
        if str(year) == "2018":
            version = "22"
        elif str(year) == "2015":
            version = "18"
        else:
            raise ValueError("Year must be either 2018 or 2015")

        self.cov = np.load(f"{data_directory}/{year}/planck_lite_{year}_v{version}_cov.npz")["cov"]
        self.data = Table.read(f"{data_directory}/{year}/planck_lite_{year}_v{version}.dat", format="ascii.commented_header")
        self.weight = np.loadtxt(f"{data_directory}/{year}/planck_lite_{year}_v{version}_weights.dat")

        if spectra == "TT":
            mask = self.data["spectrum"] == "TT"
            self.data = self.data[mask]
            self.cov = self.cov[mask][:, mask]
        elif spectra != "TTTEEE":
            # I'm doing this to match the original code, but it would be possible here
            # to do any other combination, but I don't know if it makes sense given the data.
            raise NotImplementedError("Only TT and TTTEEE spectra are implemented")

        # Assume here that no one wants an ell_max that will affect the low ell bins,
        # but we check for it below.
        self._apply_ell_max("TT", ell_max_tt)
        self._apply_ell_max("TE", ell_max_te)
        self._apply_ell_max("EE", ell_max_ee)

        if use_low_ell_bins:
            # Should do this properly and concatenate the data tables,
            # but that would involve re-indexing the weights for the high ell bit and I'm tired.
            cov_low_ell = np.load(f"{data_directory}/{year}_low_ell/planck_lite_{year}_v{version}_cov.npz")["cov"]
            self.data_low_ell = Table.read(
                f"{data_directory}/{year}_low_ell/planck_lite_{year}_v{version}.dat", format="ascii.commented_header"
            )
            self.weight_low_ell = np.loadtxt(f"{data_directory}/{year}_low_ell/planck_lite_{year}_v{version}_weights.dat")

            # Check that the ell_max for TT is not too low, otherwise we can't use the low ell bins.
            if not ((ell_max_tt is None) or (ell_max_tt > self.data_low_ell["band_ell_max"].max())):
                raise ValueError("Ell max for TT must be greater than 30 if using low ell bins")

            # we do have to put the covariances together
            self.cov = scipy.linalg.block_diag(cov_low_ell, self.cov)

        # Generate the inverse covariance matrix (precision matrix), following the original code
        # to ensure identical results.
        self.fisher = scipy.linalg.cho_solve(scipy.linalg.cho_factor(self.cov), np.identity(len(self.cov))).transpose()

        # Record this so we know later to include the low ell bins in the log likelihood.
        self.use_low_ell_bins = use_low_ell_bins

        # these will get constructed when the property is accessed
        self._data_vector = None
        self._effective_ell = None
        self._spectra = None

    def _apply_ell_max(self, spectrum, ell_max):
        if ell_max is None:
            return

        # get rid of anything where the lower end of the bandpass is above the ell_max
        bad = (self.data["band_ell_min"] > ell_max) & (self.data["spectrum"] == spectrum)
        good = ~bad
        self.data = self.data[good]
        self.cov = self.cov[good][:, good]

    def make_mean_vector(self, Dltt, Dlte, Dlee, ellmin=2, calPlanck=1.0):
        """
        Take the model Dltt, Dlte, Dlee (i.e. the ell(ell+1) Cl / 2 pi values)
        and bin them with the appropriate weights to get a predicted theory vector.

        Parameters
        ----------
        Dltt, Dlte, Dlee : array_like
            The model Dl values for TT, TE and EE.
        ellmin : int
            The minimum ell value that the supplied Dls start from.
        calPlanck : float
            A calibration factor for the Planck data, default is 1.0.
        """
        # convert model Dl's to Cls then bin them
        ls = np.arange(len(Dltt)) + ellmin
        fac = ls * (ls + 1) / (2 * np.pi)

        # Avoid an annoying division by zero warning.
        if ellmin == 0:
            fac[0] = 1.0

        # Recale the Dl's to Cl's
        Cltt = Dltt / fac
        Clte = Dlte / fac
        Clee = Dlee / fac

        Cls = {"TT": Cltt, "TE": Clte, "EE": Clee}

        # Extract the mean vector for the main (high ell) bandpowers
        mu = self._get_mean_for_data(self.data, ellmin, Cls, self.weight)

        # If we are using low ell bins, we need to get the mean for those as well
        # and combine them
        if self.use_low_ell_bins:
            mu_low_ell = self._get_mean_for_data(self.data_low_ell, ellmin, Cls, self.weight_low_ell)
            mu = np.concatenate([mu_low_ell, mu])

        return mu / calPlanck**2

    def _get_mean_for_data(self, data, ellmin, Cls, weight):
        """
        Loop through the rows in the table "data" and calculate the mean vector
        for the bandpowers using the provided Cls and weight vectors.

        We separate this into another function so that we can call it again for the low ell bins
        without duplicating code.
        """
        mu = np.zeros(len(data))
        s = data["spectrum"]
        for i, row in enumerate(data):
            b1 = row["band_ell_min"] - ellmin
            b2 = row["band_ell_max"] - ellmin
            w1 = row["weight_row_min"]
            w2 = row["weight_row_max"]
            s = row["spectrum"]
            cl = Cls[s]
            mu[i] = cl[b1:b2] @ weight[w1:w2]
        return mu

    @property
    def effective_ells(self):
        """
        Return the nominal ell values for the bandpowers, mainly useful for plotting.
        """
        if self._effective_ell is not None:
            return self._effective_ell

        self._effective_ell = np.array(self.data["band_ell_nominal"])

        if self.use_low_ell_bins:
            self._effective_ell = np.concatenate([self.data_low_ell["band_ell_nominal"], self._effective_ell])

        return self._effective_ell

    @property
    def data_vector(self):
        """
        Return the data vector, which is the bandpower values, depending on the
        choice of spectra and whether low ell bins are used.
        """
        if self._data_vector is not None:
            return self._data_vector

        d = self.data["bandpower"]

        # append the low-ell bit if used
        if self.use_low_ell_bins:
            d = np.concatenate([self.data_low_ell["bandpower"], d])
        
        # store so next time this is just returned directly
        self._data_vector = np.array(d)
        return self._data_vector

    @property
    def spectra(self):
        """
        Return the spectra names for the data vector, which is useful for seeing which bits
        of data are which.
        """
        if self._spectra is not None:
            return self._spectra

        d = self.data["spectrum"]

        if self.use_low_ell_bins:
            d = np.concatenate([self.data_low_ell["spectrum"], d])

        self._spectra = np.array(d)
        return self._spectra


    def loglike(self, Dltt, Dlte, Dlee, ellmin=2, calPlanck=1.0):
        """
        Compute the log-likelihood of the data chosen when creating this object, given
        the model Dltt, Dlte, Dlee (i.e. the ell(ell+1) Cl / 2 pi values).

        The ellmin parameter specifies the minimum value that the supplied Dls start from.

        Returns the log-likelihood value, without the normalizing |C| factor.

        This isn't used directly in CosmoSIS - instead we get the mean vector directly.

        """
        mu = self.make_mean_vector(Dltt, Dlte, Dlee, ellmin=ellmin, calPlanck=calPlanck)
        data = self.data_vector

        d = data - mu
        like = -0.5 * d @ self.fisher @ d
        return like

    def test(self):
        ls, Dltt, Dlte, Dlee = np.genfromtxt("data/Dl_planck2015fit.dat", unpack=True)
        ellmin = int(ls[0])
        loglikelihood = self.loglike(Dltt, Dlte, Dlee, ellmin)

        if self.year == 2018 and self.spectra == "TTTEEE" and not self.use_low_ell_bins:
            print("Log likelihood for 2018 high-l TT, TE and EE:")
            expected = -291.33481235418026
            # Plik-lite within cobaya gives  -291.33481235418003
        elif self.year == 2018 and self.spectra == "TTTEEE" and self.use_low_ell_bins:
            print("Log likelihood for 2018 high-l TT, TE and EE + low-l TT bins:")
            expected = -293.95586501795134
        elif self.year == 2018 and self.spectra == "TT" and not self.use_low_ell_bins:
            print("Log likelihood for 2018 high-l TT:")
            expected = -101.58123068722583
            # Plik-lite within cobaya gives -101.58123068722568
        elif self.year == 2018 and self.spectra == "TT" and self.use_low_ell_bins:
            print("Log likelihood for 2018 high-l TT + low-l TT bins:")
            expected = -104.20228335099686

        elif self.year == 2015 and self.spectra == "TTTEEE" and not self.use_low_ell_bins:
            print("NB: Don't use 2015 polarization!")
            print("Log likelihood for 2015 high-l TT, TE and EE:")
            expected = -280.9388125627618
            # Plik-lite within cobaya gives  -291.33481235418003
        elif self.year == 2015 and self.spectra == "TTTEEE" and self.use_low_ell_bins:
            print("NB: Don't use 2015 polarization!")
            print("Log likelihood for 2015 high-l TT, TE and EE + low-l TT bins:")
            expected = -283.1905700256343
        elif self.year == 2015 and self.spectra == "TT" and not self.use_low_ell_bins:
            print("Log likelihood for 2015 high-l TT:")
            expected = -102.34403873289027
            # Plik-lite within cobaya gives -101.58123068722568
        elif self.year == 2015 and self.spectra == "TT" and self.use_low_ell_bins:
            print("Log likelihood for 2015 high-l TT + low-l TT bins:")
            expected = -104.59579619576277
        else:
            expected = None

        print("Planck-lite-py:", loglikelihood)
        if expected:
            print("expected:", expected)
            print("difference:", loglikelihood - expected, "\n")
            assert np.isclose(loglikelihood, expected, rtol=1e-5), "Log likelihood does not match expected value"


if __name__ == "__main__":
    main()
