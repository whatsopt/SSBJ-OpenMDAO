"""
SSBJ test case - http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Python implementation and OpenMDAO integration developed by
Sylvain Dubreuil and Remi Lafage of ONERA, the French Aerospace Lab.
"""
from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent
from common import PolynomialFunction


def dpdx_constraint(pf, Z0):
    dpdx = pf([Z0], [1], [.25], "dpdx")

    return dpdx


class DpdxCalc(ExplicitComponent):

    def __init__(self, scalers):
        super(DpdxCalc, self).__init__()
        self.scalers = scalers
        self.pf = PolynomialFunction()

    def setup(self):
        # Global Design Variable z=(t/c,h,M,AR,Lambda,Sref)
        self.add_input('z0', val=1.0)
        self.add_output('dpdx', val=1.0)
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        Z0 = inputs['z0']*self.scalers['z'][0]
        dpdx = dpdx_constraint(self.pf, Z0)
        outputs['dpdx'] = dpdx/self.scalers['dpdx']

    def compute_partials(self, inputs, partials):

        Z0 = inputs['z0']*self.scalers['z'][0]

        # dpdx ################################################################
        # partials['dpdx', 'z0'] = np.zeros((1, 1))
        S_shifted, Ai, Aij = self.pf([Z0], [1], [.25], "dpdx", deriv=True)
        if Z0/self.pf.d['dpdx'][0]>=0.75 and Z0/self.pf.d['dpdx'][0]<=1.25:
            dStcdtc = 1.0/self.pf.d['dpdx'][0]
        else:
            dStcdtc = 0.0
        dStcdtc2 = 2.0*S_shifted[0, 0]*dStcdtc
        ddpdxdtc = Ai[0]*dStcdtc+0.5*Aij[0, 0]*dStcdtc2
        partials['dpdx', 'z0'] = ddpdxdtc*self.scalers['z'][0]/self.scalers['dpdx']
