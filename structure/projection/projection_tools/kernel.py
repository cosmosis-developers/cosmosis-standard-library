import numpy as np
import scipy.interpolate as interp
import warnings

class KernelSpline(object):
    def __init__(self, x, y, clip=1.e-6, ymin=1.e-12, 
        norm=True, is_pos=True, xmin_clipped_min=5.0):
        """
        A class for splining Kernels. For use in
        integrals, it's useful to find xmin and
        xmax as finite integration limits that will
        miss only some very small fraction (clip) of the 
        integral over the kernel. For kernels that
        are not strictly positive, we use the integral
        over the absolute value of the kernel
        because this seems conservative.

        Parameters
        ----------
        x: numpy array
            Array of x values
        y: numpy array
            Array of y values
        is_pos: bool (optional)
            assert that all y values are >=0. 
        xmin_clipped_min: float (optional)
            sets min l.o.s. distance for projection
            integral. Currently set to 5 to prevent
            dependence on very high k when n(z) doesn't
            go to zero sufficiently quickly.
        """

        self.x = x
        if is_pos and not np.all(y>=0.):
            warnings.warn("Some of your n(z) or other kernels are negative.")
        self.y = y

        #Find out if we've got zeros or very small values padding
        #the kernel - remove those that are unnecessary.
        quick_norm = np.sum(np.diff(x) * np.abs(y[:-1]))
        too_small = np.abs(self.y)/quick_norm < ymin
        good_inds, = np.where(~too_small)
        start, end = good_inds[0], good_inds[-1]
        if start>0:
            start-=1
            self.y[start]=0.
        if end<len(x)-1:
            end+=1
            self.y[end]=0.

        self.x, self.y = self.x[start:end+1], self.y[start:end+1]
        self.xmin, self.xmax = self.x[0], self.x[-1]

        #Setup spline and normalize if norm=True.
        self.spline = interp.InterpolatedUnivariateSpline(self.x, self.y)
        self.norm = self.spline.integral(self.x[0], self.x[-1])
        if norm==True:
            self.y /= self.norm
            self.norm = 1.
            self.spline = interp.InterpolatedUnivariateSpline(self.x, self.y)

        #Compute a mimumim and maximum chi below and above which
        #a fraction clip of the integral is contained
        abs_y = np.abs(self.y)
        cumsum_y = (np.cumsum(abs_y)-abs_y[0])/(np.sum(abs_y)-abs_y[0]) #cumsum_y goes from 0 to 1
        pos_diff = np.zeros(len(cumsum_y), dtype=bool)
        pos_diff[0] = True
        pos_diff[1:] = np.diff(cumsum_y)>0.
        # We have to use a linear spline here, 
        # because otherwise the points can go a bit crazy
        # near y=0 and y=1, which is exactly the place we care about.
        # I'm not quite clear of the exact cause of this in terms of cosmo
        # parameters triggering it.
        cumsum_of_x_spline = interp.InterpolatedUnivariateSpline(
            cumsum_y[pos_diff], self.x[pos_diff], k=1)
        self.xmin_clipped = cumsum_of_x_spline(clip)
        self.xmax_clipped = cumsum_of_x_spline(1-clip)

        if xmin_clipped_min>0:
            if self.xmin_clipped<xmin_clipped_min:
                self.xmin_clipped = xmin_clipped_min

        #Also compute a mean and width. Again, use the absolute value
        #of y here to avoid strange results when kernels go negatvie
        dx = np.diff(self.x)
        x_mid = 0.5*(self.x[:-1]+self.x[1:])
        abs_y_mid = 0.5*(abs_y[:-1]+abs_y[1:])
        self.mean = np.sum(dx * x_mid * abs_y_mid) / np.sum(dx * abs_y_mid)
        var = np.sum(dx * (x_mid - self.mean)**2 * abs_y_mid) / np.sum(dx*abs_y_mid)
        assert var>0.
        self.sigma = np.sqrt(var)
        #print("Setup kernel with mean(chi), sigma(chi) = %.3f, %.3f"%(self.mean, self.sigma))

    def __call__(self, x, fill_value=0.):
        if fill_value is not None:
            in_range = (x>self.xmin)*(x<self.xmax)
            return np.where(in_range, self.spline(x), fill_value)
        else:
            return self.spline(x)

class TomoNzKernel(object):
    def __init__(self, z, nzs, norm=True, is_cmb_lensing = False):
        # We need to handle CMB lensing separately
        if (not is_cmb_lensing):
            self.z = z
            self.nzs = {}
            self.nbin = 0
            for i,nz in enumerate(nzs):
                self.nbin += 1
                if norm:
                    nz_spline = interp.InterpolatedUnivariateSpline(self.z, nz)
                    norm = nz_spline.integral(self.z[0], self.z[-1])
                    nz = nz/norm
                    self.nzs[i+1] = nz

            # These get set later
            self.nchi_splines = {} #n(chi) splines
            self.wchi_splines = {} #W(chi) splines
            self.wwchi_splines = {} #W_weyl(chi) splines
        else:
            self.z = z
            self.nbin = 1
            self.cmblensing_spline = {} #W_cmb(chi) spline

    @classmethod
    def from_block(cls, block, section_name, norm=True):
        nbin = block[section_name, "nbin"]
        z = block[section_name, "z"]
        nzs = []
        for i in range(1, nbin+1):
            nz = block[section_name, "bin_%d"%i]
            nzs.append(nz)
        return cls(z, nzs, norm=norm)

    def to_block(self, block, section):
        for b, spline in self.nchi_splines.items():
            block[section, f"n_of_chi_chi_{b}"] = spline.x
            block[section, f"n_of_chi_n_{b}"] = spline.y
        for b, spline in self.wchi_splines.items():
            block[section, f"w_of_chi_chi_{b}"] = spline.x
            block[section, f"w_of_chi_n_{b}"] = spline.y
        for b, spline in self.wwchi_splines.items():
            block[section, f"ww_of_chi_chi_{b}"] = spline.x
            block[section, f"ww_of_chi_n_{b}"] = spline.y


    def set_nofchi_splines(self, chi_of_z, dchidz, clip=1.e-6):
        chi = chi_of_z(self.z)
        dchidz = dchidz(self.z)
        self.nchi_splines = {}
        for i in range(1, self.nbin+1):
            self.nchi_splines[i] = KernelSpline(chi, self.nzs[i]/dchidz, clip=clip) 

    def set_wofchi_splines(self, chi_of_z, dchidz, a_of_chi, clip=1.e-6, dchi=1.):
        if len(self.nchi_splines) == 0:
            self.set_nofchi_splines(chi_of_z, dchidz, clip=clip)
        for i in range(1, self.nbin+1):
            nchi_spline = self.nchi_splines[i]
            nchi = int(np.ceil(nchi_spline.xmax/dchi))
            chi_vals = np.linspace(0., nchi_spline.xmax, nchi)
            a_vals = a_of_chi(chi_vals)
            w_of_chi_vals = self.get_wofchi_vals(chi_vals, a_vals, nchi_spline)
            self.wchi_splines[i] = KernelSpline(chi_vals, w_of_chi_vals, 
                norm=False)

    def set_cmblensing_splines(self, chi_of_z, a_of_chi, chi_star, clip=1.e-6):
        chi = chi_of_z(self.z)
        a_vals = 1./(1.+self.z)
        w_of_chi_vals = (chi/a_vals)*(chi_star - chi)/chi_star
        self.cmblensing_spline = KernelSpline(chi, w_of_chi_vals, norm=False)    

    def get_wofchi_vals(self, chi_vals, a_vals, nchi_spline):
        w_of_chi = np.zeros_like(chi_vals)
        for j,chi in enumerate(chi_vals):
            #integral \int_{chi}^{chi_max} dchi' n(chi') * (chi' - chi)/chi'
            chi_start = max(chi, nchi_spline.xmin)
            integrand = np.where(nchi_spline.x>chi, 
                nchi_spline.y * (nchi_spline.x - chi)/nchi_spline.x, 0.)
            # Keep the cubic not-a-knot interpolant while using PPoly integration.
            integrand_spline = interp.CubicSpline(
                nchi_spline.x, integrand, bc_type="not-a-knot")
            w_of_chi[j] = chi * integrand_spline.integrate(
                chi_start, nchi_spline.xmax) / a_vals[j]
        return w_of_chi

    def set_wwofchi_splines(self, chi_of_z, dchidz, a_of_chi, clip=1.e-6, dchi=1.):
        if len(self.nchi_splines) == 0:
            self.set_nofchi_splines(chi_of_z, dchidz, clip=clip)
        for i in range(1, self.nbin+1):
            nchi_spline = self.nchi_splines[i]
            nchi = int(np.ceil(nchi_spline.xmax/dchi))
            chi_vals = np.linspace(0., nchi_spline.xmax, nchi)
            a_vals = a_of_chi(chi_vals)
            ww_of_chi_vals = self.get_wwofchi_vals(chi_vals, a_vals, nchi_spline)
            self.wwchi_splines[i] = KernelSpline(chi_vals, ww_of_chi_vals, 
                norm=False)

    def get_wwofchi_vals(self, chi_vals, a_vals, nchi_spline):
        ww_of_chi = np.zeros_like(chi_vals)
        for j,chi in enumerate(chi_vals):
            #integral \int_{chi}^{chi_max} dchi' n(chi') * (chi' - chi)/chi'
            chi_start = max(chi, nchi_spline.xmin)
            integrand = np.where(nchi_spline.x>chi, 
                nchi_spline.y * (nchi_spline.x - chi)/nchi_spline.x, 0.)
            integrand_spline = interp.InterpolatedUnivariateSpline(
                nchi_spline.x, integrand)
            ww_of_chi[j] = chi * integrand_spline.integral(
                chi_start, nchi_spline.xmax) #/ a_vals[j]
        return ww_of_chi

    def set_combined_shear_ia_splines(self, chi_of_z, dchidz, a_of_chi, 
        F_of_chi_spline, lensing_prefactor, clip=1.e-6, dchi=1.):

        if len(self.nchi_splines) == 0:
            self.set_nofchi_splines(chi_of_z, dchidz, clip=clip)
        self.shear_ia_splines = {}
        for i in range(1, self.nbin+1):
            nchi_spline = self.nchi_splines[i]
            nchi = int(np.ceil(nchi_spline.xmax/dchi))
            chi_vals = np.linspace(0., nchi_spline.xmax, nchi)
            a_vals = a_of_chi(chi_vals)
            w_of_chi_vals = self.get_wofchi_vals(chi_vals, a_vals, nchi_spline)
            #Now add IA part. This is F(chi) * n(chi) / lensing_prefactor
            ia_kernel_vals = (F_of_chi_spline(chi_vals) * 
                nchi_spline(chi_vals) / lensing_prefactor)
            w_of_chi_vals += ia_kernel_vals
            #For a negative IA amplitude, the kernel may 
            #have negative values, so call KernelSpline with
            #is_pos=False.
            self.shear_ia_splines[i] = KernelSpline(chi_vals, 
                w_of_chi_vals, norm=False, is_pos=False)

    def get_kernel_spline(self, kernel_type, bin_number):
        if kernel_type == "N":
            return self.nchi_splines[bin_number]
        elif kernel_type == "W":
            return self.wchi_splines[bin_number]
        elif kernel_type == "K":
            return self.cmblensing_spline
        elif kernel_type == "W_W":
            return self.wwchi_splines[bin_number]
        elif kernel_type == "F":
            return self.shear_ia_splines[bin_number]
        else:
            raise ValueError("""Invalid kernel_type: %s, 
                should be one of N, W, K, W_W or F"""%(kernel_type))
