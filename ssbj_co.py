import math

from openmdao.api import *

from disciplines.aerodynamics import Aerodynamics
from disciplines.performance import Performance
from disciplines.propulsion import Propulsion
from disciplines.structure import Structure
from ssbj_mda import init_ssbj_mda

import numpy as np


class SubOpt(ExplicitComponent):
    """Suboptimization component for the CO approach."""
    def initialize(self):
        self.options.declare('discipline')
        self.options.declare('scalers')

    def setup(self):
        if self.options['discipline'] == 'structures':
            # Add system-level inputs
            self.add_input('z', val=np.array([1.0,1.0,1.0,1.0,1.0,1.0]))
            self.add_input('L_hat', val=1.0)
            self.add_input('WE_hat', val=1.0)
            self.add_input('WF_hat', val=1.0)
            self.add_input('Theta_hat', val=1.0)
            self.add_input('WT_hat', val=1.0)

            # Add system-level outputs
            self.add_output('z_hat', val=np.ones(6))
            self.add_output('WF', val=1.0)
            self.add_output('Theta', val=1.0)
            self.add_output('WT', val=1.0)

            # Declare partials
            self.declare_partials('z_hat', ['z', 'L_hat', 'WE_hat', 'WF_hat', 'Theta_hat', 'WT_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('WF', ['z', 'L_hat', 'WE_hat', 'WF_hat', 'Theta_hat', 'WT_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('Theta', ['z', 'L_hat', 'WE_hat', 'WF_hat', 'Theta_hat', 'WT_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('WT', ['z', 'L_hat', 'WE_hat', 'WF_hat', 'Theta_hat', 'WT_hat'],
                                  method='fd', step=1e-4, step_calc='abs')

            # Set subproblem
            self.prob = p = Problem()

            # Define the copies so that OpenMDAO can compute derivs w.r.t. these variables
            params = p.model.add_subsystem('params', IndepVarComp(), promotes=['*'])
            params.add_output('z', np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
            params.add_output('L_hat', val=1.0)
            params.add_output('WE_hat', val=1.0)
            params.add_output('WF_hat', val=1.0)
            params.add_output('Theta_hat', val=1.0)
            params.add_output('WT_hat', val=1.0)

            # Define design variables of subproblem
            des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
            des_vars.add_output('z_hat', val=np.ones(6))
            des_vars.add_output('x_str', val=np.ones(2))

            # Define components
            # Disciplinary analysis
            p.model.add_subsystem('structures', Structure(self.options['scalers']))

            # Local constraint functions
            cstrs = ['con_theta_up = Theta*' + str(self.options['scalers']['Theta']) + '-1.04',
                     'con_theta_low = 0.96-Theta*' + str(self.options['scalers']['Theta'])]
            for i in range(5):
                cstrs.append(
                    'con_sigma' + str(i + 1) + ' = sigma[' + str(i) + ']*' + str(self.options['scalers']['sigma'][i]) + '-1.09')
            p.model.add_subsystem('constraints', ExecComp(cstrs, sigma=np.zeros(5)), promotes_outputs=['*'])

            # Local objective
            # p.model.add_subsystem('J', ExecComp('J_struc = ((z[0]-z_hat[0])**2 + (z[3]-z_hat[3])**2 +'
            #                                     '(z[4]-z_hat[4])**2 + (z[5]-z_hat[5])**2 + (WF_hat-WF)**2 + '
            #                                     '(Theta_hat-Theta)**2 + (WT_hat-WT)**2)**.5',
            #                                     z=np.ones(6), z_hat=np.ones(6)))
            p.model.add_subsystem('J', Jcalc(input_var_sets=[['z', 'z_hat', np.ones(6) + 0.1, [0, 3, 4, 5]],
                                                             ['WF_hat', 'WF', 1.1],
                                                             ['Theta_hat', 'Theta', 1.1],
                                                             ['WT_hat', 'WT', 1.1]],
                                             take_square_root=True))

            # Connect variables in sub-problem
            # Structures component inputs
            p.model.connect('z_hat', 'structures.z')
            p.model.connect('x_str', 'structures.x_str')
            p.model.connect('L_hat', 'structures.L')
            p.model.connect('WE_hat', 'structures.WE')

            # Constraints inputs
            p.model.connect('structures.sigma', 'constraints.sigma')
            p.model.connect('structures.Theta', 'constraints.Theta')

            # Objective
            p.model.connect('z', 'J.z')
            p.model.connect('z_hat', 'J.z_hat')
            p.model.connect('WF_hat', 'J.WF_hat')
            p.model.connect('structures.WF', 'J.WF')
            p.model.connect('Theta_hat', 'J.Theta_hat')
            p.model.connect('structures.Theta', 'J.Theta')
            p.model.connect('WT_hat', 'J.WT_hat')
            p.model.connect('structures.WT', 'J.WT')

            # Set subproblem optimizer
            p.driver = ScipyOptimizeDriver()
            p.driver.options['optimizer'] = 'SLSQP'
            p.driver.options['maxiter'] = 100
            p.driver.options['tol'] = 1e-8
            #p.driver.options['debug_print'] = ['desvars', 'ln_cons', 'nl_cons', 'objs']

            # Add design variables
            p.model.add_design_var('x_str', lower=np.array([0.4, 0.75]), upper=np.array([1.6, 1.25]))
            p.model.add_design_var('z_hat', lower=np.array([0.2, 0.666, 0.875, 0.45, 0.72, 0.5]),
                                   upper=np.array([1.8, 1.333, 1.125, 1.45, 1.27, 1.5]),
                                   indices=[0,3,4,5])

            # Add objective
            p.model.add_objective('J.J')

            # Add constraints
            p.model.add_constraint('con_theta_up', upper=0.0)
            p.model.add_constraint('con_theta_low', upper=0.0)
            p.model.add_constraint('con_sigma1', upper=0.0)
            p.model.add_constraint('con_sigma2', upper=0.0)
            p.model.add_constraint('con_sigma3', upper=0.0)
            p.model.add_constraint('con_sigma4', upper=0.0)
            p.model.add_constraint('con_sigma5', upper=0.0)

            # Final setup
            p.setup()
            p.final_setup()

            # View model
            # view_model(p, outfile='co_struc.html')

        elif self.options['discipline'] == 'aerodynamics':
            # Add system-level inputs
            self.add_input('z', val=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
            self.add_input('ESF_hat', val=1.0)
            self.add_input('WT_hat', val=1.0)
            self.add_input('Theta_hat', val=1.0)
            self.add_input('L_hat', val=1.0)
            self.add_input('D_hat', val=1.0)
            self.add_input('fin_hat', val=1.0)

            # Add system-level outputs
            self.add_output('z_hat', val=np.ones(6))
            self.add_output('L', val=1.0)
            self.add_output('fin', val=1.0)
            self.add_output('D', val=1.0)

            # Declare partials
            self.declare_partials('z_hat', ['z', 'ESF_hat', 'WT_hat', 'Theta_hat', 'L_hat', 'D_hat', 'fin_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('L', ['z', 'ESF_hat', 'WT_hat', 'Theta_hat', 'L_hat', 'D_hat', 'fin_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('fin', ['z', 'ESF_hat', 'WT_hat', 'Theta_hat', 'L_hat', 'D_hat', 'fin_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('D', ['z', 'ESF_hat', 'WT_hat', 'Theta_hat', 'L_hat', 'D_hat', 'fin_hat'],
                                  method='fd', step=1e-4, step_calc='abs')

            # Set subproblem
            self.prob = p = Problem()

            # Define the copies so that OpenMDAO can compute derivs w.r.t. these variables
            params = p.model.add_subsystem('params', IndepVarComp(), promotes=['*'])
            params.add_output('z', np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
            params.add_output('ESF_hat', val=1.0)
            params.add_output('WT_hat', val=1.0)
            params.add_output('Theta_hat', val=1.0)
            params.add_output('L_hat', val=1.0)
            params.add_output('D_hat', val=1.0)
            params.add_output('fin_hat', val=1.0)

            # Define design variables of subproblem
            des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
            des_vars.add_output('z_hat', val=np.ones(6))
            des_vars.add_output('x_aer', val=1.0)

            # Define components
            # Disciplinary analysis
            p.model.add_subsystem('aerodynamics', Aerodynamics(self.options['scalers']))

            # Local constraint functions
            p.model.add_subsystem('constraints',
                                  ExecComp('con_dpdx = dpdx*' + str(self.options['scalers']['dpdx']) + '-1.04'))

            # Local objective
            # p.model.add_subsystem('J', ExecComp('J_aero = (sum((z-z_hat)**2) + (fin_hat-fin)**2 + '
            #                                     '(D_hat-D)**2 + (L_hat-L)**2)**.5',
            #                                     z=np.zeros(6), z_hat=np.zeros(6)))
            p.model.add_subsystem('J', Jcalc(input_var_sets=[['z', 'z_hat', np.ones(6) + .1],
                                                             ['fin_hat', 'fin', 1.1],
                                                             ['D_hat', 'D', 1.1],
                                                             ['L_hat', 'L', 1.1]],
                                             take_square_root=True))

            # Connect variables in sub-problem
            # Aerodynamics component inputs
            p.model.connect('z_hat', 'aerodynamics.z')
            p.model.connect('x_aer', 'aerodynamics.x_aer')
            p.model.connect('ESF_hat', 'aerodynamics.ESF')
            p.model.connect('WT_hat', 'aerodynamics.WT')
            p.model.connect('Theta_hat', 'aerodynamics.Theta')

            # Constraints inputs
            p.model.connect('aerodynamics.dpdx', 'constraints.dpdx')

            # Objective
            p.model.connect('z', 'J.z')
            p.model.connect('z_hat', 'J.z_hat')
            p.model.connect('fin_hat', 'J.fin_hat')
            p.model.connect('aerodynamics.fin', 'J.fin')
            p.model.connect('D_hat', 'J.D_hat')
            p.model.connect('aerodynamics.D', 'J.D')
            p.model.connect('L_hat', 'J.L_hat')
            p.model.connect('aerodynamics.L', 'J.L')

            # Set subproblem optimizer
            p.driver = ScipyOptimizeDriver()
            p.driver.options['optimizer'] = 'SLSQP'
            p.driver.options['maxiter'] = 100
            p.driver.options['tol'] = 1e-8
            #p.driver.options['debug_print'] = ['desvars', 'ln_cons', 'nl_cons', 'objs']

            # Add design variables
            p.model.add_design_var('x_aer', lower=0.75, upper=1.25)
            p.model.add_design_var('z_hat', lower=np.array([0.2, 0.666, 0.875, 0.45, 0.72, 0.5]),
                                   upper=np.array([1.8, 1.333, 1.125, 1.45, 1.27, 1.5]))

            # Add objective
            p.model.add_objective('J.J')

            # Add constraints
            p.model.add_constraint('constraints.con_dpdx', upper=0.0)

            # Final setup
            p.setup()
            p.final_setup()

            # View model
            # view_model(p, outfile='co_aero.html')
        elif self.options['discipline'] == 'propulsion':
            # Add system-level inputs
            self.add_input('z', val=np.ones(6))
            self.add_input('D_hat', val=1.0)
            self.add_input('ESF_hat', val=1.0)
            self.add_input('SFC_hat', val=1.0)
            self.add_input('WE_hat', val=1.0)

            # Add system-level outputs
            self.add_output('z_hat', val=np.ones(6))
            self.add_output('ESF', val=1.0)
            self.add_output('SFC', val=1.0)
            self.add_output('WE', val=1.0)

            # Declare partials
            self.declare_partials('z_hat', ['z', 'D_hat', 'ESF_hat', 'SFC_hat', 'WE_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('ESF', ['z', 'D_hat', 'ESF_hat', 'SFC_hat', 'WE_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('SFC', ['z', 'D_hat', 'ESF_hat', 'SFC_hat', 'WE_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('WE', ['z', 'D_hat', 'ESF_hat', 'SFC_hat', 'WE_hat'],
                                  method='fd', step=1e-4, step_calc='abs')

            # Set subproblem
            self.prob = p = Problem()

            # Define the copies so that OpenMDAO can compute derivs w.r.t. these variables
            params = p.model.add_subsystem('params', IndepVarComp(), promotes=['*'])
            params.add_output('z', np.ones(6))
            params.add_output('D_hat', val=1.0)
            params.add_output('ESF_hat', val=1.0)
            params.add_output('SFC_hat', val=1.0)
            params.add_output('WE_hat', val=1.0)

            # Define design variables of subproblem
            des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
            des_vars.add_output('z_hat', val=np.ones(6))
            des_vars.add_output('x_pro', val=1.0)

            # Define components
            # Disciplinary analysis
            p.model.add_subsystem('propulsion', Propulsion(self.options['scalers']))

            # Local constraint functions
            cnstrnts = ['con1_esf = ESF*' + str(self.options['scalers']['ESF']) + '-1.5',
                        'con2_esf = 0.5-ESF*' + str(self.options['scalers']['ESF']),
                        'con_temp = Temp*' + str(self.options['scalers']['Temp']) + '-1.02',
                        'con_dt=DT']

            p.model.add_subsystem('constraints', ExecComp(cnstrnts))

            # Local objective
            # p.model.add_subsystem('J', ExecComp('J_prop = ((z[1]-z_hat[1])**2 + (z[2]-z_hat[2])**2 + '
            #                                     '(ESF_hat-ESF)**2 + (WE_hat-WE)**2 + (SFC_hat-SFC)**2)**.5',
            #                                     z=np.zeros(6), z_hat=np.zeros(6)))
            p.model.add_subsystem('J', Jcalc(input_var_sets=[['z', 'z_hat', np.ones(6) + .1, [1, 2]],
                                                             ['ESF_hat', 'ESF', 1.1],
                                                             ['WE_hat', 'WE', 1.1],
                                                             ['SFC_hat', 'SFC', 1.1]],
                                             take_square_root=True))

            # Connect variables in sub-problem
            # Aerodynamics component inputs
            p.model.connect('z_hat', 'propulsion.z')
            p.model.connect('x_pro', 'propulsion.x_pro')
            p.model.connect('D_hat', 'propulsion.D')

            # Constraints inputs
            p.model.connect('propulsion.ESF', 'constraints.ESF')
            p.model.connect('propulsion.Temp', 'constraints.Temp')
            p.model.connect('propulsion.DT', 'constraints.DT')

            # Objective
            p.model.connect('z', 'J.z')
            p.model.connect('z_hat', 'J.z_hat')
            p.model.connect('ESF_hat', 'J.ESF_hat')
            p.model.connect('propulsion.ESF', 'J.ESF')
            p.model.connect('WE_hat', 'J.WE_hat')
            p.model.connect('propulsion.WE', 'J.WE')
            p.model.connect('SFC_hat', 'J.SFC_hat')
            p.model.connect('propulsion.SFC', 'J.SFC')

            # Set subproblem optimizer
            p.driver = ScipyOptimizeDriver()
            p.driver.options['optimizer'] = 'SLSQP'
            p.driver.options['maxiter'] = 100
            p.driver.options['tol'] = 1e-8
            #p.driver.options['debug_print'] = ['desvars', 'ln_cons', 'nl_cons', 'objs']

            # Add design variables
            p.model.add_design_var('x_pro', lower=0.18, upper=1.81)
            p.model.add_design_var('z_hat', lower=np.array([0.2, 0.666, 0.875, 0.45, 0.72, 0.5]),
                                   upper=np.array([1.8, 1.333, 1.125, 1.45, 1.27, 1.5]),
                                   indices=[1,2])

            # Add objective
            p.model.add_objective('J.J')

            # Add constraints
            p.model.add_constraint('constraints.con1_esf', upper=0.0)
            p.model.add_constraint('constraints.con2_esf', upper=0.0)
            p.model.add_constraint('constraints.con_temp', upper=0.0)
            p.model.add_constraint('constraints.con_dt', upper=0.0)

            # Final setup
            p.setup()
            p.final_setup()

            # View model
            # view_model(p, outfile='co_prop.html')
        else:
            raise IOError('Unknown discipline {} provided in setup function.'.format(self.options['discipline']))

    def compute(self, inputs, outputs):
        p = self.prob
        if self.options['discipline'] == 'structures':
            # Push any global inputs down
            p['z'] = inputs['z']
            p['L_hat'] = inputs['L_hat']
            p['WE_hat'] = inputs['WE_hat']
            p['WF_hat'] = inputs['WF_hat']
            p['Theta_hat'] = inputs['Theta_hat']
            p['WT_hat'] = inputs['WT_hat']

            # Run the optimization
            print('{} discipline suboptimization'.format(self.options['discipline']))
            p.run_driver()

            # Pull the values back up to the output array
            outputs['z_hat'] = p['z_hat']
            outputs['WF'] = p['structures.WF']
            outputs['Theta'] = p['structures.Theta']
            outputs['WT'] = p['structures.WT']
        elif self.options['discipline'] == 'aerodynamics':
            # Push any global inputs down
            p['z'] = inputs['z']
            p['ESF_hat'] = inputs['ESF_hat']
            p['WT_hat'] = inputs['WT_hat']
            p['Theta_hat'] = inputs['Theta_hat']
            p['L_hat'] = inputs['L_hat']
            p['D_hat'] = inputs['D_hat']
            p['fin_hat'] = inputs['fin_hat']

            # Run the optimization
            print('{} discipline suboptimization'.format(self.options['discipline']))
            p.run_driver()

            # Pull the values back up to the output array
            outputs['z_hat'] = p['z_hat']
            outputs['L'] = p['aerodynamics.L']
            outputs['fin'] = p['aerodynamics.fin']
            outputs['D'] = p['aerodynamics.D']
        elif self.options['discipline'] == 'propulsion':
            # Push any global inputs down
            p['z'] = inputs['z']
            p['D_hat'] = inputs['D_hat']
            p['ESF_hat'] = inputs['ESF_hat']
            p['SFC_hat'] = inputs['SFC_hat']
            p['WE_hat'] = inputs['WE_hat']

            # Run the optimization
            print('{} discipline suboptimization'.format(self.options['discipline']))
            p.run_driver()

            # Pull the values back up to the output array
            outputs['z_hat'] = p['z_hat']
            outputs['ESF'] = p['propulsion.ESF']
            outputs['SFC'] = p['propulsion.SFC']
            outputs['WE'] = p['propulsion.WE']
        else:
            raise IOError('Unknown discipline {} provided in setup function.'.format(self.options['discipline']))


class Jcalc(ExplicitComponent):

    def initialize(self):
        self.options.declare('input_var_sets')
        self.options.declare('take_square_root', default=True)

    def setup(self):
        defined_inputs = []
        for input_set in self.options['input_var_sets']:
            if input_set[0] not in defined_inputs:
                self.add_input(input_set[0], val=input_set[2])
                defined_inputs.append(input_set[0])
            if input_set[1] not in defined_inputs:
                self.add_input(input_set[1], val=input_set[2])

        self.add_output('J', val=1.)

        for input_set in self.options['input_var_sets']:
            self.declare_partials('J', input_set[0])
            self.declare_partials('J', input_set[1])

    def compute(self, inputs, outputs):
        J = 0.0
        for input_set in self.options['input_var_sets']:
            if len(input_set) == 4:  # indices are provided
                input_set0 = np.take(inputs[input_set[0]], input_set[3])
                input_set1 = np.take(inputs[input_set[1]], input_set[3])
            else:
                input_set0 = inputs[input_set[0]]
                input_set1 = inputs[input_set[1]]
            J += (sum((input_set0-input_set1)**2))
        if self.options['take_square_root']:
            outputs['J'] = J**.5
        else:
            outputs['J'] = J

    def compute_partials(self, inputs, partials):
        denom_term = 0.0
        if self.options['take_square_root']:
            for input_set in self.options['input_var_sets']:
                if len(input_set) == 4:  # indices are provided
                    input_set0 = np.take(inputs[input_set[0]], input_set[3])
                    input_set1 = np.take(inputs[input_set[1]], input_set[3])
                else:
                    input_set0 = inputs[input_set[0]]
                    input_set1 = inputs[input_set[1]]
                denom_term += (sum((input_set0 - input_set1) ** 2))
            denom_term = denom_term**.5
        else:
            denom_term = 0.5
        for input_set in self.options['input_var_sets']:
            if denom_term == 0.0:
                partials['J', input_set[0]] = np.zeros(len(inputs[input_set[0]]))
                partials['J', input_set[1]] = np.zeros(len(inputs[input_set[1]]))
            else:
                if len(input_set) == 4: # indices are provided
                    input_set0 = np.take(inputs[input_set[0]], input_set[3])
                    input_set1 = np.take(inputs[input_set[1]], input_set[3])
                else:
                    input_set0 = inputs[input_set[0]]
                    input_set1 = inputs[input_set[1]]
                partial0 = (input_set0-input_set1)/denom_term
                partial1 = (input_set1-input_set0)/denom_term
                if len(input_set) == 4:
                    partials['J', input_set[0]] = np.zeros(len(inputs[input_set[0]]))
                    partials['J', input_set[1]] = np.zeros(len(inputs[input_set[1]]))
                    for i, j in enumerate(input_set[3]):
                        partials['J', input_set[0]][0,j] = partial0[i]
                        partials['J', input_set[1]][0,j] = partial1[i]
                else:
                    partials['J', input_set[0]] = partial0
                    partials['J', input_set[1]] = partial1


class SsbjCO(Group):
    """Main group for the SSBJ case to run it using Collaborative Optimization."""
    def __init__(self, scalers):
        super(SsbjCO, self).__init__()
        self.scalers = scalers

    def setup(self):

        # Define system-level design variables
        des_vars = self.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        # Add global design variables
        des_vars.add_output('z', val=np.array([1.0,1.0,1.0,1.0,1.0,1.0]))
        # Add coupling copies for interdisciplinary couplings
        des_vars.add_output('L_hat', val=1.0)
        des_vars.add_output('D_hat', val=1.0)
        des_vars.add_output('WE_hat', val=1.0)
        des_vars.add_output('WT_hat', val=1.0)
        des_vars.add_output('Theta_hat', val=1.0)
        des_vars.add_output('WF_hat', val=1.0)
        des_vars.add_output('ESF_hat', val=1.0)
        des_vars.add_output('fin_hat', val=1.0)
        des_vars.add_output('SFC_hat', val=1.0)

        # Add suboptimizations
        self.add_subsystem('subopt_struc', SubOpt(discipline='structures', scalers=self.scalers))
        self.add_subsystem('subopt_aero', SubOpt(discipline='aerodynamics', scalers=self.scalers))
        self.add_subsystem('subopt_prop', SubOpt(discipline='propulsion', scalers=self.scalers))

        # Add system-level analyses
        self.add_subsystem('performance', Performance(self.scalers))
        # J_tot_expr = 'J = ((z[0]-z_hat_struc[0])**2 + (z[3]-z_hat_struc[3])**2 + (z[4]-z_hat_struc[4])**2 ' \
        #              '+ (z[5]-z_hat_struc[5])**2 + (WF_hat-WF_struc)**2 + (Theta_hat-Theta_struc)**2 ' \
        #              '+ (WT_hat-WT_struc)**2 + sum((z-z_hat_aero)**2) + (fin_hat-fin_aero)**2 + (D_hat-D_aero)**2 ' \
        #              '+ (L_hat-L_aero)**2 + (z[1]-z_hat_prop[1])**2 + (z[2]-z_hat_prop[2])**2 + (ESF_hat-ESF_prop)**2' \
        #              ' + (WE_hat-WE_prop)**2 + (SFC_hat-SFC_prop)**2)**0.5'
        # self.add_subsystem('J', ExecComp(J_tot_expr, z=np.ones(6), z_hat_struc=np.ones(6),
        #                                  z_hat_aero=np.ones(6), z_hat_prop=np.ones(6)))
        self.add_subsystem('J', Jcalc(input_var_sets = [['z', 'z_hat_struc', np.ones(6), [0, 3, 4, 5]],
                                                        ['WF_hat', 'WF_struc', 1.],
                                                        ['Theta_hat', 'Theta_struc', 1.],
                                                        ['WT_hat', 'WT_struc', 1.],
                                                        ['z', 'z_hat_aero', np.ones(6)],
                                                        ['fin_hat', 'fin_aero', 1.],
                                                        ['D_hat', 'D_aero', 1.],
                                                        ['L_hat', 'L_aero', 1.],
                                                        ['z', 'z_hat_prop', np.ones(6), [1, 2]],
                                                        ['ESF_hat', 'ESF_prop', 1.],
                                                        ['WE_hat', 'WE_prop', 1.],
                                                        ['SFC_hat', 'SFC_prop', 1.]],
                                      take_square_root=True))

        # Connect variables
        self.connect('z', ['subopt_struc.z', 'subopt_aero.z', 'subopt_prop.z', 'performance.z', 'J.z'])
        self.connect('L_hat', ['subopt_struc.L_hat', 'subopt_aero.L_hat', 'J.L_hat'])
        self.connect('D_hat', ['subopt_aero.D_hat', 'subopt_prop.D_hat', 'J.D_hat'])
        self.connect('WE_hat', ['subopt_struc.WE_hat', 'subopt_prop.WE_hat', 'J.WE_hat'])
        self.connect('WT_hat', ['performance.WT', 'subopt_struc.WT_hat', 'subopt_aero.WT_hat', 'J.WT_hat'])
        self.connect('Theta_hat', ['subopt_struc.Theta_hat', 'subopt_aero.Theta_hat', 'J.Theta_hat'])
        self.connect('WF_hat', ['performance.WF', 'subopt_struc.WF_hat', 'J.WF_hat'])
        self.connect('ESF_hat', ['subopt_aero.ESF_hat', 'subopt_prop.ESF_hat', 'J.ESF_hat'])
        self.connect('fin_hat', ['performance.fin', 'subopt_aero.fin_hat', 'J.fin_hat'])
        self.connect('SFC_hat', ['performance.SFC', 'subopt_prop.SFC_hat', 'J.SFC_hat'])
        self.connect('subopt_struc.z_hat', ['J.z_hat_struc'])
        self.connect('subopt_struc.WF', ['J.WF_struc'])
        self.connect('subopt_struc.Theta', ['J.Theta_struc'])
        self.connect('subopt_struc.WT', ['J.WT_struc'])
        self.connect('subopt_aero.z_hat', ['J.z_hat_aero'])
        self.connect('subopt_aero.fin', ['J.fin_aero'])
        self.connect('subopt_aero.D', ['J.D_aero'])
        self.connect('subopt_aero.L', ['J.L_aero'])
        self.connect('subopt_prop.z_hat', ['J.z_hat_prop'])
        self.connect('subopt_prop.ESF', ['J.ESF_prop'])
        self.connect('subopt_prop.WE', ['J.WE_prop'])
        self.connect('subopt_prop.SFC', ['J.SFC_prop'])


if __name__ == '__main__':

    scalers = init_ssbj_mda()
    prob = Problem()

    prob.model = model = SsbjCO(scalers=scalers)

    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    # prob.driver.opt_settings['Major optimality tolerance'] = 1e-1
    # prob.driver.opt_settings['Major feasibility tolerance'] = 1e-3
    #prob.driver.options['debug_print'] = ['desvars', 'ln_cons', 'nl_cons', 'objs']
    prob.driver.opt_settings['ACC'] = 1e-4
    prob.set_solver_print(level=2)

    # Set design variables
    prob.model.add_design_var('z', lower=np.array([0.2, 0.666, 0.875, 0.45, 0.72, 0.5]),
                              upper=np.array([1.8, 1.333, 1.125, 1.45, 1.27, 1.5]))

    # Add some logical, though conservative, bounds to the coupling variables that have become design variables
    prob.model.add_design_var('L_hat', lower=0.40, upper=1.58)                         # 20000-80000 (w.r.t. 50606.68)
    prob.model.add_design_var('D_hat', lower=0.082, upper=3.28)                        # 1000-40000 (w.r.t 12193.80)
    prob.model.add_design_var('WE_hat', lower=0.0, upper=3.148)                        # 0-20000 (w.r.t 6354.07)
    prob.model.add_design_var('WT_hat', lower=0.40, upper=1.58)                        # 20000-80000 (w.r.t 50606.68)
    prob.model.add_design_var('Theta_hat', lower=0.935280535448, upper=1.01322058007)  # 0.96-1.04 (w.r.t 1.02643)
    prob.model.add_design_var('WF_hat', lower=0.0, upper=5.47480025858)                # 0-40000 (w.r.t 7306.20262124)
    prob.model.add_design_var('ESF_hat', lower=0.994476478742, upper=2.98342943623)    # 0.5-1.5 (w.r.t 0.5027771)
    prob.model.add_design_var('fin_hat', lower=0.48190485642, upper=3.85523885136)     # 2-16 (w.r.t 4.15019681449)
    prob.model.add_design_var('SFC_hat', lower=0.451432768154, upper=1.35429830446)    # 0.5-1.5 (w.r.t 1.10758464)

    # Set objective
    prob.model.add_objective('performance.R', scaler=-1.)

    # Set constraints
    prob.model.add_constraint('J.J', equals=0.0)

    # Setup
    prob.setup(mode='fwd')

    # view_model(prob, outfile='co_ssbj.html')

    #prob.check_partials(compact_print=True)

    prob.run_driver()

    print('Outcome of analysis:')
    print('objective=', prob['performance.R'] * scalers['R'])
    print('Z_opt=', prob['z'] * scalers['z'])
    print('X_str_opt=', prob.model.subopt_struc.prob['x_str'] * scalers['x_str'])
    print('X_aer_opt=', prob.model.subopt_aero.prob['x_aer'])
    print('X_pro_opt=', prob.model.subopt_prop.prob['x_pro'] * scalers['x_pro'])
    print('R_opt=', prob['performance.R'] * scalers['R'])

    # prob = Problem()
    #
    # prob.model = model = Group()
    #
    # prob.model.add_subsystem('inputs', IndepVarComp)