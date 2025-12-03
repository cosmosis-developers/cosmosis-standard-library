from __future__ import print_function
from builtins import range
from builtins import object
from scipy.interpolate import RectBivariateSpline
import numpy as np


"""
This module calculates the galaxy and intrinsic alignment bias
using the flexible grid parameterisation of Joachimi and Bridle
(2010) p 6-9. 
Outputs both stochastic and systematic terms rI, bI, rg and bg.       

"""


class flexible_grid(object):
    def __init__(self, config):
        self.nz = config['nznodes']
        self.nk = config['nknodes']
        self.galaxy_bias = config['galaxy_bias']
        self.intrinsic_alignments = config['intrinsic_alignments']

        interface = {True: 'yes', False: 'no'}
        print("intrinsic alignments: %s" % interface[self.intrinsic_alignments])
        print("galaxy bias: %s" % interface[self.galaxy_bias])
        print("initialised %d x %d (nz x nk) bias grid." % (self.nz, self.nk))

    def setup_grid_nodes(self, block):
        BI = np.zeros((self.nz, self.nk))
        Bg = np.zeros((self.nz, self.nk))

        for i in range(self.nz):
            for j in range(self.nk):
                if self.intrinsic_alignments:
                    BI[i, j] = block['intrinsic_alignment_parameters',
                                     'node_%d_%d' % (i + 1, j + 1)]
                if self.galaxy_bias:
                    Bg[i, j] = block['bias_parameters',
                                     'node_%d_%d' % (i + 1, j + 1)]

        #import pdb ; pdb.set_trace()
        # Fix the edge nodes to zero
        if self.intrinsic_alignments:
            np.lib.pad(BI, 1, fixed_edge)
            self.BI = BI
        if self.galaxy_bias:
            np.lib.pad(Bg, 1, fixed_edge)
            self.Bg = Bg

        # Load the power spectra required and one free amplitude parameter
        if self.intrinsic_alignments:
            self.AI = block.get_double('intrinsic_alignment_parameters', 'A')
            self.z, self.k, self.b_I_fid = block.get_grid(
                'intrinsic_alignment_parameters', 'z', 'k_h', 'b_I')
            self.z, self.k, self.r_I_fid = block.get_grid(
                'intrinsic_alignment_parameters', 'z', 'k_h', 'r_I')
        if self.galaxy_bias:
            self.Ag = block.get_double_grid('bias_parameters', 'A')
            self.z, self.k, self.b_g_fid = block.get_double_grid(
                'bias_parameters', 'z', 'k_h', 'b_g')
            self.z, self.k, self.r_g_fid = block.get_double_grid(
                'bias_parameters', 'z', 'k_h', 'r_g')

        self.K = np.logspace(np.log10(self.k.min()),
                             np.log10(self.k.max()), self.nk)
        self.Z = np.linspace(self.z.min(), self.z.max(), self.nz)

    def interpolate_grid(self):
        # Use the grid points to get nzxnk free bias parameters for an arbitrary set of k,z coordinates
        if self.intrinsic_alignments:
            ia_interp = RectBivariateSpline(np.log(self.K), self.Z, self.BI)
            self.QI = ia_interp(np.log(self.k), self.z)
        if self.galaxy_bias:
            gb_interp = RectBivariateSpline(np.log(self.K), self.Z, self.Bg)
            self.Qg = gb_interp(np.log(self.k), self.z)

    def evaluate_and_save_bias(self, block):
        # Use the interpolated grid to evaluate a bias at each k,z
        if self.intrinsic_alignments:
            b_I = self.AI * self.QI * self.b_I_fid
            r_I = self.AI * self.QI * self.r_I_fid
            block.replace_grid('intrinsic_alignments_parameters',
                               'z', self.z, 'k_h', self.k, 'b_I', b_I)
            block.replace_grid('intrinsic_alignments_parameters',
                               'z', self.z, 'k_h', self.k, 'r_I', r_I)
        if self.galaxy_bias:
            b_g = self.Ag * self.Qg * self.b_g_fid
            r_g = self.Ag * self.Qg * self.r_g_fid
            block.replace_grid('intrinsic_alignments_parameters',
                               'z', self.z, 'k_h', self.k, 'b_g', b_g)
            block.replace_grid('intrinsic_alignments_parameters',
                               'z', self.z, 'k_h', self.k, 'r_g', r_g)


def fixed_edge(v, width, i, kw):
    v[:width[0]] = 0.
    v[-width[1]:] = 0.
    return v
