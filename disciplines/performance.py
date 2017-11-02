"""
SSBJ test case - http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Python implementation and OpenMDAO integration developed by
Sylvain Dubreuil and Remi Lafage of ONERA, the French Aerospace Lab.
"""
from __future__ import print_function
import numpy as np
from openmdao.api import ExplicitComponent
from common import PolynomialFunction
# pylint: disable=C0103

class Performance(ExplicitComponent):

    def __init__(self, scalers):
        super(Performance, self).__init__()
        # scalers values
        self.scalers = scalers

    def setup(self):
        # Global Design Variable z=(t/c,h,M,AR,Lambda,Sref)
        self.add_input('z', val=np.ones(6))
        # Local Design Variable x_per=null
        # Coupling parameters
        self.add_input('WT', val=1.0)
        self.add_input('WF', val=1.0)
        self.add_input('fin', val=1.0)
        self.add_input('SFC', val=1.0)
        # Coupling output
        self.add_output('R', val=1.0)
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        #Variables scaling
        Z = inputs['z']*self.scalers['z']
        fin = inputs['fin']*self.scalers['fin']
        SFC = inputs['SFC']*self.scalers['SFC']
        WT = inputs['WT']*self.scalers['WT']
        WF = inputs['WF']*self.scalers['WF']
        if Z[1] <= 36089.:
            theta = 1.0-6.875E-6*Z[1]
        else:
            theta = 0.7519
        R = 661.0*np.sqrt(theta)*Z[2]*fin/SFC*np.log(abs(WT/(WT-WF)))
        outputs['R'] = R/self.scalers['R']

    def compute_partials(self, inputs, J):
        Z = inputs['z']*self.scalers['z']
        fin = inputs['fin']*self.scalers['fin']
        SFC = inputs['SFC']*self.scalers['SFC']
        WT = inputs['WT']*self.scalers['WT']
        WF = inputs['WF']*self.scalers['WF']
        
        if Z[1] <= 36089:
            theta = 1.0-6.875E-6*Z[1]
            dRdh = -0.5*661.0*theta**-0.5*6.875e-6*Z[2]*fin \
                   /SFC*np.log(abs(WT/(WT-WF)))
        else:
            theta = 0.7519
            dRdh = 0.0

        dRdM = 661.0*np.sqrt(theta)*fin/SFC*np.log(abs(WT/(WT-WF)))

        J['R', 'z'] = np.zeros((1, 6))
        J['R', 'z'][0, 1] = np.array([dRdh/self.scalers['R'] *45000.0])
        J['R', 'z'][0, 2] = np.array([dRdM/self.scalers['R'] *1.6])
        dRdfin = 661.0*np.sqrt(theta)*Z[2]/SFC*np.log(abs(WT/(WT-WF)))
        J['R', 'fin'] = np.array([[dRdfin/self.scalers['R']*self.scalers['fin']]])
        dRdSFC = -661.0*np.sqrt(theta)*Z[2]*fin/SFC**2*np.log(abs(WT/(WT-WF)))
        J['R', 'SFC'] = np.array([[dRdSFC/self.scalers['R']*self.scalers['SFC']]])
        dRdWT = 661.0*np.sqrt(theta)*Z[2]*fin/SFC*-WF/(WT*(WT-WF))
        J['R', 'WT'] = np.array([[dRdWT/self.scalers['R']*self.scalers['WT']]])
        dRdWF = 661.0*np.sqrt(theta)*Z[2]*fin/SFC*1.0/(WT-WF)
        J['R', 'WF'] = np.array([[dRdWF/self.scalers['R']*self.scalers['WF']]])

if __name__ == "__main__": # pragma: no cover

    from openmdao.api import Problem, Group, IndepVarComp
    scalers = {}
    scalers['z'] = np.array([0.05, 45000., 1.6, 5.5, 55.0, 1000.0])
    scalers['fin'] = 4.093062
    scalers['SFC'] = 1.12328
    scalers['WF'] = 7306.20261
    scalers['WT'] = 49909.58578
    scalers['R'] = 528.91363
    top = Problem()
    top.root = Group()
    top.root.add('z_in', IndepVarComp('z', np.array([1.2  ,  1.333,  0.875,  0.45 ,  1.27 ,  1.5])),
                 promotes=['*'])
    top.root.add('WT_in', IndepVarComp('WT', 0.888), promotes=['*'])
    top.root.add('WF_in', IndepVarComp('WF', 2.66), promotes=['*'])
    top.root.add('fin_in', IndepVarComp('fin', 1.943), promotes=['*'])
    top.root.add('SFC_in', IndepVarComp('SFC', 0.8345), promotes=['*'])
    top.root.add('Per1', Performance(scalers), promotes=['*'])
    top.setup()
    top.check_partials(compact_print=True)
