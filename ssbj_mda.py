"""
SSBJ test case - http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Python implementation and OpenMDAO integration developed by
Sylvain Dubreuil and Remi Lafage of ONERA, the French Aerospace Lab.
"""
from __future__ import print_function
import numpy as np

from openmdao.api import ExecComp, IndepVarComp
from openmdao.api import Group, Problem
from openmdao.api import NonlinearBlockGS, ScipyKrylov

from disciplines.aerodynamics import Aerodynamics
from disciplines.performance import Performance
from disciplines.propulsion import Propulsion
from disciplines.structure import Structure
# pylint: disable=C0103

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
        sap_group.add_subsystem('Struc', Structure(self.scalers), promotes=['*'])
        sap_group.add_subsystem('Aero', Aerodynamics(self.scalers), promotes=['*'])
        sap_group.add_subsystem('Propu',Propulsion(self.scalers),promotes=['*'])

        sap_group.nonlinear_solver = NonlinearBlockGS()
        sap_group.nonlinear_solver.options['atol'] = 1.0e-3
        sap_group.linear_solver = ScipyKrylov()
        self.add_subsystem('sap', sap_group, promotes=['*'])

        self.add_subsystem('Perfo', Performance(self.scalers), promotes=['*'])

        #Constraints
        self.add_subsystem('con_Theta_sup', ExecComp('con_Theta_up = Theta*'+\
                                           str(self.scalers['Theta'])+'-1.04'), promotes=['*'])
        self.add_subsystem('con_Theta_inf', ExecComp('con_Theta_low = 0.96-Theta*'+\
                                           str(self.scalers['Theta'])), promotes=['*'])
        for i in range(5):
            self.add_subsystem('con_Sigma'+str(i+1), ExecComp('con_sigma'+str(i+1)+'=sigma['+str(i)+']*'+\
                                                    str(self.scalers['sigma'][i])+'-1.9',
                                                    sigma=np.zeros(5)), promotes=['*'])
        self.add_subsystem('con_Dpdx', ExecComp('con_dpdx=dpdx*'+str(self.scalers['dpdx'])+'-1.04'),
                 promotes=['*'])
        self.add_subsystem('con1_ESF', ExecComp('con1_esf=ESF*'+str(self.scalers['ESF'])+'-1.5'),
                 promotes=['*'])
        self.add_subsystem('con2_ESF', ExecComp('con2_esf=0.5-ESF*'+str(self.scalers['ESF'])),
                 promotes=['*'])
        self.add_subsystem('con_Temp', ExecComp('con_temp=Temp*'+str(self.scalers['Temp'])+'-1.02'),
                 promotes=['*'])

        self.add_subsystem('con_DT', ExecComp('con_dt=DT'), promotes=['*'])

def init_ssbj_mda():
    """
    Runs the analysis once.
    """
    prob = Problem()

    # Mean point is chosen for the design variables
    scalers = {}
    scalers['z']=np.array([0.05,45000.,1.6,5.5,55.0,1000.0])
    scalers['x_aer']=1.0
    scalers['x_str']=np.array([0.25,1.0])
    scalers['x_pro']=0.5

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

    #Uptade the scalers dictionary
    for key in scalers.iterkeys():
        if key not in ['z', 'x_str', 'x_pro']:
            scalers[key] = prob[key]

    return scalers

if __name__ == "__main__":
    scalers = init_ssbj_mda()
    print(scalers)
