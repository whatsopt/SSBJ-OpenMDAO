"""
SSBJ test case - http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Python implementation and OpenMDAO integration developed by
Sylvain Dubreuil and Remi Lafage of ONERA, the French Aerospace Lab.
"""
from __future__ import print_function
from six import iterkeys
import numpy as np

from openmdao.api import ExecComp, IndepVarComp
from openmdao.api import Group, Problem
from openmdao.api import NonlinearBlockGS, ScipyKrylov

from ssbj_disciplines.aerodynamics import Aerodynamics
from ssbj_disciplines.performance import Performance
from ssbj_disciplines.propulsion import Propulsion
from ssbj_disciplines.structure import Structure
from ssbj_disciplines.common import PolynomialFunction

class SSBJ_MDA(Group):
    """
    SSBJ Analysis with aerodynamics, performance, propulsion and structure disciplines.
    """
    def __init__(self, scalers):
        super(SSBJ_MDA, self).__init__()
        self.scalers = scalers

    def setup(self):
        #Design variables
        self.add_subsystem('z_ini',
                 IndepVarComp('z', np.array([1.0,1.0,1.0,1.0,1.0,1.0])),
                 promotes=['*'])
        self.add_subsystem('x_aer_ini', IndepVarComp('x_aer', 1.0), promotes=['*'])
        self.add_subsystem('x_str_ini',
                 IndepVarComp('x_str', np.array([1.0,1.0])),
                 promotes=['*'])
        self.add_subsystem('x_pro_ini', IndepVarComp('x_pro', 1.0), promotes=['*'])

        #Disciplines
        sap_group = Group()
        sap_group.add_subsystem('Structure', Structure(self.scalers), promotes=['*'])
        sap_group.add_subsystem('Aerodynamics', Aerodynamics(self.scalers), promotes=['*'])
        sap_group.add_subsystem('Propulsion', Propulsion(self.scalers),promotes=['*'])

        sap_group.nonlinear_solver = NonlinearBlockGS()
        sap_group.nonlinear_solver.options['atol'] = 1.0e-3
        sap_group.linear_solver = ScipyKrylov()
        self.add_subsystem('Mda', sap_group, promotes=['*'])

        self.add_subsystem('Performance', Performance(self.scalers), promotes=['*'])

        #Constraints
        cstrs = ['con_theta_up = Theta*'+str(self.scalers['Theta'])+'-1.04',
                 'con_theta_low = 0.96-Theta*'+ str(self.scalers['Theta']),
                 'con_dpdx = dpdx*'+str(self.scalers['dpdx'])+'-1.04',
                 'con1_esf = ESF*'+str(self.scalers['ESF'])+'-1.5',
                 'con2_esf = 0.5-ESF*'+str(self.scalers['ESF']),
                 'con_temp = Temp*'+str(self.scalers['Temp'])+'-1.02',
                 'con_dt=DT'
                 ]
        for i in range(5):
            cstrs.append('con_sigma'+str(i+1)+' = sigma['+str(i)+']*'+ str(self.scalers['sigma'][i])+'-1.09')
        self.add_subsystem('Constraints', ExecComp(cstrs, sigma=np.zeros(5)), promotes=['*'])

def init_ssbj_mda():
    """
    Runs the analysis once.
    """
    prob = Problem()

    # Mean point is chosen for the design variables
    scalers = {}
    #scalers['z'] = np.array([0.06, 60000., 1.4, 2.475, 69.85, 1500.0])  # optimum
    scalers['z'] = np.array([0.05, 45000., 1.6, 5.500, 55.00, 1000.0])  # start point
    scalers['x_aer']=1.#0.75
    scalers['x_str']=np.array([.25, 1.])#np.array([0.28959593,0.75])
    scalers['x_pro']=.5#0.15621093

    # Others variables are unknowns for the moment so Scale=1.0
    scalers['WT']=1.0
    scalers['Theta']=1.0
    scalers['L']=1.0
    scalers['WF']=1.0
    scalers['D']=1.0
    scalers['ESF']=1.0
    scalers['WE']=1.0
    scalers['fin']=1.0
    scalers['SFC']=1.0
    scalers['R']=1.0
    scalers['DT']=1.0
    scalers['Temp']=1.0
    scalers['dpdx']=1.0
    scalers['sigma']=np.array([1.0,1.0,1.0,1.0,1.0])

    prob.model = SSBJ_MDA(scalers)
    prob.setup()

    #Initialization of acceptable values as initial values for the polynomial functions
    Z = prob['z']*scalers['z']
    Wfo = 2000
    Wo = 25000
    We = 3*4360.0*(1.0**1.05)
    t = Z[0]*Z[5]/np.sqrt(Z[5]*Z[3])
    Wfw = (5.*Z[5]/18.)*(2.0/3.0*t)*(42.5)
    Fo = 1.0
    Wtotal = 80000.
    Wtot=1.1*Wtotal
    while abs(Wtot - Wtotal) > Wtotal*0.0001:
        Wtot = Wtotal
        Ww  = Fo*(.0051*((Wtot*6.0)**0.557)*Z[5]**.649*Z[3]**.5*Z[0]**-.4*((1.0+0.25)**.1)*\
                  ((np.cos(Z[4]*np.pi/180))**-1)*((.1875*Z[5])**.1))
        Wtotal = Wo + Ww + Wfo + Wfw + We

    prob['WT'] = Wtotal
    # prob['sap']['Aero.WT'] = Wtotal
    # prob['sap']['Struc.L'] = Wtotal
    prob['L'] = Wtotal

    prob.run_driver()

    #Update the scalers dictionary
    for key in iterkeys(scalers):
        if key not in ['z', 'x_str', 'x_aer', 'x_pro']:
            scalers[key] = prob[key]
    return scalers

def get_initial_state():
    return init_ssbj_mda(), PolynomialFunction().d

if __name__ == "__main__":
    scalers = init_ssbj_mda()
    print(scalers)
