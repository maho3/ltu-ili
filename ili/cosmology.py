"""
Cosmology module for the ILI project.

This version is based on CAMB. Eventually, we could add support for other Boltzmann solvers such as CLASS
"""

import camb


class Cosmology(object):
    def __init__(
        self,
        h=0.67,
        ombh2=0.022,
        omch2=0.12,
        omk=0.0,
        tau=0.06,
        ns=0.96,
        As=2.1e-9,
        w=-1.0,
        wa=0.0,
    ):
        """
        :param h: Hubble constant
        :param ombh2: Baryon density
        :param omch2: CDM density
        :param omk: Curvature density
        :param tau: Optical depth
        :param ns: Scalar spectral index
        :param As: Scalar amplitude
        :param w: Dark energy equation of state
        :param wa: Dark energy equation of state time derivative
        """
        self.h = h
        self.ombh2 = ombh2
        self.omch2 = omch2
        self.omk = omk
        self.tau = tau
        self.ns = ns
        self.As = As
        self.w = w
        self.wa = wa

    def get_cosmology(self):
        """
        :return: CAMB cosmology object
        """
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.h * 100,
            ombh2=self.ombh2,
            omch2=self.omch2,
            omk=self.omk,
            tau=self.tau,
        )
        pars.InitPower.set_params(ns=self.ns, As=self.As)
        pars.set_dark_energy(w=self.w, wa=self.wa)
        pars.set_for_lmax(2500, lens_potential_accuracy=0)
        results = camb.get_results(pars)
        return results

    def get_matter_power_spectrum(self, minkh=1e-4, maxkh=1, npoints=200):
        """
        :param minkh: Minimum k
        :param maxkh: Maximum k
        :param npoints: Number of points
        :return: k, z and Matter power spectrum
        """
        results = self.get_cosmology()
        kh, z, pk = results.get_matter_power_spectrum(
            minkh=minkh, maxkh=maxkh, npoints=npoints
        )
        return kh, z, pk
