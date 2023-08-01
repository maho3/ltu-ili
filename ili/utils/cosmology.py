"""
Cosmology module for the ILI project.

This version is based on CAMB. Eventually, we could add support for other
Boltzmann solvers such as CLASS
"""

import camb
from typing import Tuple
import numpy as np


class Cosmology(object):
    """Cosmology object to provide CAMB cosmology object and matter power
    spectrum.

    Args:
        h (float): Hubble constant.
        ombh2 (float): Baryon density.
        omch2 (float): CDM density.
        omk (float): Curvature density.
        tau (float): Optical depth.
        ns (float): Scalar spectral index.
        As (float): Scalar amplitude.
        w (float): Dark energy equation of state.
        wa (float): Dark energy equation of state time derivative.
    """

    def __init__(
        self,
        h: float = 0.67,
        ombh2: float = 0.022,
        omch2: float = 0.12,
        omk: float = 0.0,
        tau: float = 0.06,
        ns: float = 0.96,
        As: float = 2.1e-9,
        w: float = -1.0,
        wa: float = 0.0,
    ) -> None:
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

    def get_matter_power_spectrum(
        self,
        minkh: float = 1e-4,
        maxkh: float = 1,
        npoints: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the matter power spectrum.

        Args:
            minkh (float): Minimum k.
            maxkh (float): Maximum k.
            npoints (int): Number of points.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: k, z and Matter power
                spectrum.
        """
        results = self.get_cosmology()
        kh, z, pk = results.get_matter_power_spectrum(
            minkh=minkh, maxkh=maxkh, npoints=npoints
        )
        return kh, z, pk
