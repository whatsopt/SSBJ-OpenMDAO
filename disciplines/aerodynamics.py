"""
SSBJ test case - http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Python implementation and OpenMDAO integration developed by
Sylvain Dubreuil and Remi Lafage of ONERA, the French Aerospace Lab.
"""
from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent
from .common import PolynomialFunction, CDMIN
# pylint: disable=C0103

def aerodynamics(pf, x_aer, Z, WT, ESF, Theta):
    if Z[1] <= 36089.0:
        V = 1116.39 * Z[2] * np.sqrt(abs(1.0 - 6.875E-6*Z[1]))
        rho = 2.377E-3 * (1. - 6.875E-6*Z[1])**4.2561
    else:
        V = 968.1 * abs(Z[2])
        rho = 2.377E-3 * 0.2971 * np.exp((36089.0 - Z[1]) / 20806.7)
    CL = WT / (0.5*rho*(V**2)*Z[5])
    Fo2 = pf([ESF, abs(x_aer)], [1, 1], [.25]*2, "Fo2")

    CDmin = CDMIN*Fo2 + 3.05*abs(Z[0])**(5.0/3.0) \
            * abs(np.cos(Z[4]*np.pi/180.0))**1.5
    if Z[2] >= 1:
        k = abs(Z[3]) * (abs(Z[2])**2-1.0) * np.cos(Z[4]*np.pi/180.) \
        / (4.* abs(Z[3])* np.sqrt(abs(Z[4]**2 - 1.) - 2.))
    else:
        k = (0.8 * np.pi * abs(Z[3]))**-1

    Fo3 = pf([Theta], [5], [.25], "Fo3")
    CD = (CDmin + k * CL**2) * Fo3
    D = CD * 0.5 * rho * V**2 * Z[5]
    fin = WT/D
    L = WT
    dpdx = pf([Z[0]], [1], [.25], "dpdx")

    return L, D, fin, dpdx 

class Aerodynamics(ExplicitComponent):

    def __init__(self, scalers):
        super(Aerodynamics, self).__init__()
        self.scalers = scalers
        self.pf = PolynomialFunction()

    def setup(self):
        # Global Design Variable z=(t/c,h,M,AR,Lambda,Sref)
        self.add_input('z', val=np.ones(6))
        # Local Design Variable x_aer=Cf
        self.add_input('x_aer', val=1.0)
        # Coupling parameters
        self.add_input('WT', val=1.0)
        self.add_input('Theta', val=1.0)
        self.add_input('ESF', val=1.0)
        # Coupling output
        self.add_output('L', val=1.0)
        self.add_output('D', val=1.0)
        self.add_output('fin', val=1.0)
        self.add_output('dpdx', val=1.0)
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        Z = inputs['z']*self.scalers['z']
        x_aer = inputs['x_aer']*self.scalers['x_aer']
        WT = inputs['WT']*self.scalers['WT']
        ESF = inputs['ESF']*self.scalers['ESF']
        Theta = inputs['Theta']*self.scalers['Theta']

        L, D, fin, dpdx = aerodynamics(self.pf, x_aer, Z, WT, ESF, Theta)

        outputs['L'] = L/self.scalers['L']
        outputs['D'] = D/self.scalers['D']
        outputs['fin'] = fin/self.scalers['fin']
        outputs['dpdx'] = dpdx/self.scalers['dpdx']


    def compute_partials(self, inputs, partials):

        Z = inputs['z']*self.scalers['z']
        WT = inputs['WT']*self.scalers['WT']
        ESF = inputs['ESF']*self.scalers['ESF']
        Theta = inputs['Theta']*self.scalers['Theta']

        # auxiliary computations
        if Z[1] <= 36089.0:
            V = 1116.39 * Z[2] * np.sqrt(abs(1.0 - 6.875E-6 * Z[1]))
            rho = 2.377E-3 * (1. - 6.875E-6*Z[1])**4.2561
        else:
            V = 968.1 * abs(Z[2])
            rho = 2.377E-3*0.2971*np.exp((36089.0 - Z[1])/20806.7)
        CL = WT / (0.5*rho*(V**2)*Z[5])
        s_new = [ESF, abs(inputs['x_aer'])]
        Fo2 = self.pf(s_new, [1, 1], [.25]*2, "Fo2")

        CDmin = CDMIN * Fo2 + 3.05 * abs(Z[0])**(5.0/3.0) \
                * abs(np.cos(Z[4]*np.pi/180.0))**1.5
        if Z[2] >= 1.:
            k = abs(Z[3]) * (abs(Z[2])**2-1.0) * np.cos(Z[4]*np.pi/180.) \
                / (4. * abs(Z[3])* np.sqrt(abs(Z[4]**2 - 1.) - 2.))
        else:
            k = (0.8 * np.pi * abs(Z[3]))**-1

        Fo3 = self.pf([Theta], [5], [.25], "Fo3")
        CD = (CDmin + k * CL**2) * Fo3
        D = CD * 0.5 * rho * V**2 * Z[5]

        # dL #################################################################
        partials['L', 'x_aer'] = np.array([[0.0]])
        partials['L', 'z'] = np.zeros((1, 6))
        partials['L', 'WT'] = np.array([[1.0]])
        partials['L', 'Theta'] = np.array([[0.0]])
        partials['L', 'ESF'] = np.array([[0.0]])

        # dD #################################################################
        S_shifted, Ai, Aij = self.pf(s_new,
                                          [1, 1], [.25]*2, "Fo2", deriv=True)
        if abs(inputs['x_aer'])/self.pf.d['Fo2'][1]>=0.75 and \
           abs(inputs['x_aer'])/self.pf.d['Fo2'][1]<=1.25:	  
            dSCfdCf = 1.0/self.pf.d['Fo2'][1]
        else:
            dSCfdCf = 0.0
        dSCfdCf2 = 2.0*S_shifted[0, 1]*dSCfdCf
        dFo1dCf = Ai[1]*dSCfdCf+0.5*Aij[1, 1]*dSCfdCf2+Aij[0, 1]*S_shifted[0, 1]*dSCfdCf
        dDdCf = 0.5*rho*V**2*Z[5]*Fo3*CDMIN*dFo1dCf
        partials['D', 'x_aer'] = np.array([[dDdCf/self.scalers['D']]]).reshape((1, 1))
        dDdtc = 0.5*rho*V**2*Z[5]*5.0/3.0*3.05*Fo3*Z[0]**(2./3.)*np.cos(Z[4]*np.pi/180.)**(3./2.)
        if Z[1] <= 36089.0:
            drhodh = 2.377E-3 * 4.2561 * 6.875E-6* (1. - 6.875E-6 * Z[1])**3.2561
            dVdh = 6.875E-6*1116.39*Z[2]/2* (1.0 - 6.875E-6 * Z[1])**-0.5
        else:
            drhodh = 2.377E-3 * 0.2971 * (-1.0)/20806.7 *np.exp((36089.0 - Z[1]) / 20806.7)
            dVdh = 0.0
        dVdh2 = 2.0*dVdh*V
        dCDdh = -k*Fo3*CL*WT/(0.5*Z[5])*(V**-2*rho**-2*drhodh+rho**-1*V**-3*dVdh)
        dDdh = 0.5*Z[5]*(drhodh*CDmin*V**2+rho*dCDdh*V**2+rho*CDmin*dVdh2)
        if Z[1] <= 36089.0:
            dVdM = 1116.39*(1.0 - 6.875E-6 * Z[1])**-0.5
        else:
            dVdM = 968.1
        if Z[2] >= 1:
            dkdM = abs(Z[3]) * (2.0*abs(Z[2])) * np.cos(Z[4]*np.pi/180.) \
                / (4. * abs(Z[3])* np.sqrt(abs(Z[4]**2 - 1.) - 2.))
        else:
            dkdM = 0.0
        dVdM2 = 2.0*V*dVdM
        dCLdM = -2.0*WT/(0.5*Z[5])*rho**-1*V**-3*dVdM
        dCDdM = Fo3*(2.0*k*CL*dCLdM+CL**2*dkdM)
        dDdM = 0.5*rho*Z[5]*(CD*dVdM2+V**2*dCDdM)
        if Z[2] >= 1:
            dkdAR = 0.0
        else:
            dkdAR = -1.0/(0.8 * np.pi * abs(Z[3])**2)
        dCDdAR = Fo3*CL**2*dkdAR
        dDdAR = 0.5*rho*Z[5]*V**2*dCDdAR
        dCDmindLambda = -3.05*3.0/2.0*Z[0]**(5.0/3.0)\
            *np.cos(Z[4]*np.pi/180.)**0.5*np.pi/180.*np.sin(Z[4]*np.pi/180.)
        if Z[2] >= 1:
            u = (Z[2]**2-1.)*np.cos(Z[4]*np.pi/180.)
            up = -np.pi/180.0*(Z[2]**2-1.)*np.sin(Z[4]*np.pi/180.)
            v = 4.0*np.sqrt(Z[4]**2-1.0)-2.0
            vp = 4.0*Z[4]*(Z[4]**2-1.0)**-0.5
            dkdLambda = (up*v-u*vp)/v**2
        else:
            dkdLambda = 0.0
        dCDdLambda = Fo3*(dCDmindLambda+CL**2*dkdLambda)
        dDdLambda = 0.5*rho*Z[5]*V**2*dCDdLambda
        dCLdSref2 = 2.0*CL*-WT/(0.5*rho*V**2*Z[5]**2)
        dCDdSref = Fo3*k*dCLdSref2
        dDdSref = 0.5*rho*V**2*(CD+Z[5]*dCDdSref)
        partials['D', 'z'] = np.array([np.append(dDdtc/self.scalers['D'], [dDdh/self.scalers['D'],
                                 dDdM/self.scalers['D'], dDdAR/self.scalers['D'],
                                 dDdLambda/self.scalers['D'],
                                 dDdSref/self.scalers['D']])])*self.scalers['z']
        dDdWT = Fo3*k*2.0*WT/(0.5*rho*V**2*Z[5])
        partials['D', 'WT'] = np.array([[dDdWT/self.scalers['D']*self.scalers['WT']]])
        S_shifted, Ai, Aij = self.pf([Theta], [5], [.25], "Fo3", deriv=True)
        if Theta/self.pf.d['Fo3'][0]>=0.75 and Theta/self.pf.d['Fo3'][0]<=1.25: 
            dSThetadTheta = 1.0/self.pf.d['Fo3'][0]
        else:
            dSThetadTheta = 0.0
        dSThetadTheta2 = 2.0*S_shifted[0, 0]*dSThetadTheta
        dFo3dTheta = Ai[0]*dSThetadTheta + 0.5*Aij[0, 0]*dSThetadTheta2
        dCDdTheta = dFo3dTheta*(CDmin+k*CL**2)
        dDdTheta = 0.5*rho*V**2*Z[5]*dCDdTheta
        partials['D', 'Theta'] = np.array(
            [[dDdTheta/self.scalers['D']*self.scalers['Theta']]]).reshape((1, 1))
        S_shifted, Ai, Aij = self.pf(s_new,
                                          [1, 1], [.25]*2, "Fo2", deriv=True)
        if ESF/self.pf.d['Fo2'][0]>=0.75 and ESF/self.pf.d['Fo2'][0]<=1.25: 							  
            dSESFdESF = 1.0/self.pf.d['Fo2'][0]
        else:
            dSESFdESF = 0.0
        dSESFdESF2 = 2.0*S_shifted[0, 0]*dSESFdESF
        dFo2dESF = Ai[0]*dSESFdESF+0.5*Aij[0, 0]*dSESFdESF2 \
                   + Aij[1, 0]*S_shifted[0, 1]*dSESFdESF
        dCDdESF = Fo3*CDMIN*dFo2dESF
        dDdESF = 0.5*rho*V**2*Z[5]*dCDdESF
        partials['D', 'ESF'] = np.array(
            [[dDdESF/self.scalers['D']*self.scalers['ESF']]]).reshape((1, 1))

        # dpdx ################################################################
        partials['dpdx', 'x_aer'] = np.array([[0.0]])
        partials['dpdx', 'z'] = np.zeros((1, 6))
        S_shifted, Ai, Aij = self.pf([Z[0]], [1], [.25], "dpdx", deriv=True)
        if Z[0]/self.pf.d['dpdx'][0]>=0.75 and Z[0]/self.pf.d['dpdx'][0]<=1.25:
            dStcdtc = 1.0/self.pf.d['dpdx'][0]
        else:
            dStcdtc = 0.0
        dStcdtc2 = 2.0*S_shifted[0, 0]*dStcdtc
        ddpdxdtc = Ai[0]*dStcdtc+0.5*Aij[0, 0]*dStcdtc2
        partials['dpdx', 'z'][0, 0] = ddpdxdtc*self.scalers['z'][0]/self.scalers['dpdx']
        partials['dpdx', 'WT'] = np.array([[0.0]])
        partials['dpdx', 'Theta'] = np.array([[0.0]])
        partials['dpdx', 'ESF'] = np.array([[0.0]])
        
        # dfin ###############################################################
        partials['fin', 'x_aer'] = np.array(
            [[-dDdCf*WT/D**2/self.scalers['WT']*self.scalers['D']]]).reshape((1, 1))
        partials['fin', 'z'] = np.array(
            [-partials['D', 'z'][0]*WT/self.scalers['WT']/D**2*self.scalers['D']**2])
        partials['fin', 'WT'] = np.array(
            [[(D-dDdWT*WT)/D**2/self.scalers['WT']*self.scalers['D'] \
              *self.scalers['WT']]]).reshape((1, 1))
        partials['fin', 'Theta'] = np.array(
            [[(-dDdTheta*WT)/D**2/self.scalers['WT']*self.scalers['D'] \
              *self.scalers['Theta']]]).reshape((1, 1))
        partials['fin', 'ESF'] = np.array(
            [[(-dDdESF*WT)/D**2/self.scalers['WT']\
              *self.scalers['D']*self.scalers['ESF']]]).reshape((1, 1))

if __name__ == "__main__": # pragma: no cover

    from openmdao.api import Problem, Group, IndepVarComp 
    scalers = {}
    scalers['z'] = np.array([0.05, 45000., 1.6, 5.5, 55.0, 1000.0])
    scalers['WT'] = 49909.58578
    scalers['ESF'] = 1.0
    scalers['Theta'] = 0.950978
    scalers['D'] = 12193.7018
    scalers['fin'] = 4.093062
    scalers['dpdx'] = 1.0

    top = Problem()
    top.model.add_subsystem('z_in', IndepVarComp('z', np.array([1.2  ,  1.333,  0.875,  0.45 ,  1.27 ,  1.5])),
                            promotes=['*'])
    top.model.add_subsystem('x_aer_in', IndepVarComp('x_aer', 0.75), promotes=['*'])
    top.model.add_subsystem('WT_in', IndepVarComp('WT', 0.89), promotes=['*'])
    top.model.add_subsystem('Theta_in', IndepVarComp('Theta', 0.9975), promotes=['*'])
    top.model.add_subsystem('ESF_in', IndepVarComp('ESF', 1.463), promotes=['*'])
    top.model.add_subsystem('Aer1', Aerodynamics(scalers, PolynomialFunction()), promotes=['*'])
    top.setup()
    top.check_partials(compact_print=True)
