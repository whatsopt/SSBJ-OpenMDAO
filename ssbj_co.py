"""
SSBJ test case - http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Python implementation and OpenMDAO integration developed by
Sylvain Dubreuil and Remi Lafage of ONERA, the French Aerospace Lab.
Collaborative Optimization (CO) strategy optimization and postprocessing scripts
developed by Imco van Gent of TU Delft, Faculty of Aerospace Engineering.
"""
import datetime
import math

from openmdao.api import *

from ssbj_disciplines.aerodynamics import Aerodynamics
from ssbj_disciplines.performance import Performance
from ssbj_disciplines.propulsion import Propulsion
from ssbj_disciplines.structure import Structure
from ssbj_mda import init_ssbj_mda

import numpy as np


# Set keyword for case reader files (to be used in postprocessing script)
cr_files_key_word = 'results'  # or use: str(datetime.datetime.now())


class SubOpt(ExplicitComponent):
    """Suboptimization component for the CO approach."""

    def initialize(self):
        self.options.declare('discipline')
        self.options.declare('scalers')
        self.options.declare('driver')

    def setup(self):
        if self.options['discipline'] == 'structures':
            # Add system-level inputs
            self.add_input('z', val=np.array([1.0,1.0,1.0,1.0,1.0,1.0]))
            self.add_input('WE_hat', val=1.0)
            self.add_input('WF_hat', val=1.0)
            self.add_input('Theta_hat', val=1.0)
            self.add_input('WT_hat', val=1.0)

            # Add system-level outputs
            self.add_output('z_hat_str', val=np.array([1.0,1.0,1.0,1.0,1.0,1.0]))
            self.add_output('WF', val=1.0)
            self.add_output('Theta', val=1.0)
            self.add_output('WT', val=1.0)

            # Declare partials
            self.declare_partials('z_hat_str', ['z', 'WE_hat', 'WF_hat', 'Theta_hat', 'WT_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('WF', ['z', 'WE_hat', 'WF_hat', 'Theta_hat', 'WT_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('Theta', ['z', 'WE_hat', 'WF_hat', 'Theta_hat', 'WT_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('WT', ['z', 'WE_hat', 'WF_hat', 'Theta_hat', 'WT_hat'],
                                  method='fd', step=1e-4, step_calc='abs')

            # Set subproblem
            self.prob = p = Problem()

            # Define the copies so that OpenMDAO can compute derivs w.r.t. these variables
            params = p.model.add_subsystem('params', IndepVarComp(), promotes=['*'])
            params.add_output('z', np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
            #params.add_output('L_hat', val=1.0)
            params.add_output('WE_hat', val=1.0)
            params.add_output('WF_hat', val=1.0)
            params.add_output('Theta_hat', val=1.0)
            params.add_output('WT_hat', val=1.0)

            # Define design variables of subproblem
            des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
            des_vars.add_output('z_hat_str', val=np.array([1.0,1.0,1.0,1.0,1.0,1.0]))
            des_vars.add_output('x_str', val=np.array([1.6, 0.75]))

            # Define components
            # Disciplinary analysis
            p.model.add_subsystem('structures', Structure(self.options['scalers']))

            # Local constraint functions
            cstrs = ['con_theta = Theta*' + str(self.options['scalers']['Theta']) + '-1.04']
            for i in range(5):
                cstrs.append('con_sigma' + str(i + 1) + ' = sigma[' + str(i) + ']*' +
                             str(self.options['scalers']['sigma'][i]) + '-1.09')
            p.model.add_subsystem('constraints', ExecComp(cstrs, sigma=np.zeros(5)), promotes_outputs=['*'])

            # Local objective
            p.model.add_subsystem('J', ExecComp('J = ((z[0]-z_hat_str[0])**2 + (z[3]-z_hat_str[3])**2 +'
                                                '(z[4]-z_hat_str[4])**2 + (z[5]-z_hat_str[5])**2 + (WF_hat-WF)**2 + '
                                                '(Theta_hat-Theta)**2 + (WT_hat-WT)**2)**.5',
                                                z=np.ones(6), z_hat_str=np.ones(6)))

            # Connect variables in sub-problem
            # Structures component inputs
            p.model.connect('z_hat_str', 'structures.z')
            p.model.connect('x_str', 'structures.x_str')
            p.model.connect('WT_hat', 'structures.L')
            p.model.connect('WE_hat', 'structures.WE')

            # Constraints inputs
            p.model.connect('structures.sigma', 'constraints.sigma')
            p.model.connect('structures.Theta', 'constraints.Theta')

            # Objective
            p.model.connect('z', 'J.z')
            p.model.connect('z_hat_str', 'J.z_hat_str')
            p.model.connect('WF_hat', 'J.WF_hat')
            p.model.connect('structures.WF', 'J.WF')
            p.model.connect('Theta_hat', 'J.Theta_hat')
            p.model.connect('structures.Theta', 'J.Theta')
            p.model.connect('WT_hat', 'J.WT_hat')
            p.model.connect('structures.WT', 'J.WT')

            # Set subproblem optimizer
            if isinstance(self.options['driver'], ScipyOptimizeDriver):
                p.driver = ScipyOptimizeDriver()
                p.driver.options['optimizer'] = 'SLSQP'
                p.driver.options['maxiter'] = 100
                p.driver.options['tol'] = 1e-6
            elif isinstance(self.options['driver'], pyOptSparseDriver):
                p.driver = pyOptSparseDriver()
                p.driver.options['optimizer'] = 'SLSQP'
                p.driver.opt_settings['MAXIT'] = 100
                p.driver.opt_settings['ACC'] = 1e-6
            #p.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']

            # Set recording options
            recorder = SqliteRecorder(os.path.join('files', 'ssbj_cr_{}_subsystem_str.sql'.format(cr_files_key_word)))
            p.driver.add_recorder(recorder)
            p.driver.recording_options['includes'] = []
            p.driver.recording_options['record_objectives'] = True
            p.driver.recording_options['record_constraints'] = True
            p.driver.recording_options['record_desvars'] = True
            p.driver.recording_options['record_metadata'] = True

            # Add design variables
            p.model.add_design_var('x_str', lower=np.array([0.4, 0.75]), upper=np.array([1.6, 1.25]))
            p.model.add_design_var('z_hat_str', lower=np.array([0.2, 0.45, 0.72, 0.5]),
                                   upper=np.array([1.8, 1.45, 1.27, 1.5]),
                                   indices=[0, 3, 4, 5])

            # Add objective
            p.model.add_objective('J.J')

            # Add constraints
            p.model.add_constraint('con_theta', upper=0.0, lower=-0.04)
            p.model.add_constraint('con_sigma1', upper=0.0)
            p.model.add_constraint('con_sigma2', upper=0.0)
            p.model.add_constraint('con_sigma3', upper=0.0)
            p.model.add_constraint('con_sigma4', upper=0.0)
            p.model.add_constraint('con_sigma5', upper=0.0)

            # Final setup
            p.setup()
            p.final_setup()

            # View model
            view_model(p, outfile=os.path.join('files', 'co_n2_struc.html'), show_browser=False)

        elif self.options['discipline'] == 'aerodynamics':
            # Add system-level inputs (N.B. L_hat is not used, instead L_hat = W_hat is assumed)
            self.add_input('z', val=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
            self.add_input('ESF_hat', val=1.0)
            self.add_input('WT_hat', val=1.0)
            self.add_input('Theta_hat', val=1.0)
            self.add_input('D_hat', val=1.0)
            self.add_input('fin_hat', val=1.0)

            # Add system-level outputs
            self.add_output('z_hat_aer', val=np.ones(6))
            self.add_output('L', val=1.0)
            self.add_output('fin', val=1.0)
            self.add_output('D', val=1.0)

            # Declare partials
            self.declare_partials('z_hat_aer', ['z', 'ESF_hat', 'WT_hat', 'Theta_hat', 'D_hat', 'fin_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('L', ['z', 'ESF_hat', 'WT_hat', 'Theta_hat', 'D_hat', 'fin_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('fin', ['z', 'ESF_hat', 'WT_hat', 'Theta_hat', 'D_hat', 'fin_hat'],
                                  method='fd', step=1e-4, step_calc='abs')
            self.declare_partials('D', ['z', 'ESF_hat', 'WT_hat', 'Theta_hat', 'D_hat', 'fin_hat'],
                                  method='fd', step=1e-4, step_calc='abs')

            # Set subproblem
            self.prob = p = Problem()

            # Define the copies so that OpenMDAO can compute derivs w.r.t. these variables
            params = p.model.add_subsystem('params', IndepVarComp(), promotes=['*'])
            params.add_output('z', np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
            params.add_output('ESF_hat', val=1.0)
            params.add_output('WT_hat', val=1.0)
            params.add_output('Theta_hat', val=1.0)
            params.add_output('D_hat', val=1.0)
            params.add_output('fin_hat', val=1.0)

            # Define design variables of subproblem
            des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
            des_vars.add_output('z_hat_aer', val=np.ones(6))
            des_vars.add_output('x_aer', val=1.0)

            # Define components
            # Disciplinary analysis
            p.model.add_subsystem('aerodynamics', Aerodynamics(self.options['scalers']))

            # Local constraint functions
            p.model.add_subsystem('constraints',
                                  ExecComp('con_dpdx = dpdx*' + str(self.options['scalers']['dpdx']) + '-1.04'))

            # Local objective
            p.model.add_subsystem('J', ExecComp('J = (sum((z-z_hat_aer)**2) + (fin_hat-fin)**2 + (D_hat-D)**2 + '
                                                '(WT_hat-L)**2)**.5', z=np.zeros(6), z_hat_aer=np.zeros(6)))

            # Connect variables in sub-problem
            # Aerodynamics component inputs
            p.model.connect('z_hat_aer', 'aerodynamics.z')
            p.model.connect('x_aer', 'aerodynamics.x_aer')
            p.model.connect('ESF_hat', 'aerodynamics.ESF')
            p.model.connect('WT_hat', 'aerodynamics.WT')
            p.model.connect('Theta_hat', 'aerodynamics.Theta')

            # Constraints inputs
            p.model.connect('aerodynamics.dpdx', 'constraints.dpdx')

            # Objective
            p.model.connect('z', 'J.z')
            p.model.connect('z_hat_aer', 'J.z_hat_aer')
            p.model.connect('fin_hat', 'J.fin_hat')
            p.model.connect('aerodynamics.fin', 'J.fin')
            p.model.connect('D_hat', 'J.D_hat')
            p.model.connect('aerodynamics.D', 'J.D')
            p.model.connect('WT_hat', 'J.WT_hat')
            p.model.connect('aerodynamics.L', 'J.L')

            # Set subproblem optimizer
            if isinstance(self.options['driver'], ScipyOptimizeDriver):
                p.driver = ScipyOptimizeDriver()
                p.driver.options['optimizer'] = 'SLSQP'
                p.driver.options['maxiter'] = 100
                p.driver.options['tol'] = 1e-6
            elif isinstance(self.options['driver'], pyOptSparseDriver):
                p.driver = pyOptSparseDriver()
                p.driver.options['optimizer'] = 'SLSQP'
                p.driver.opt_settings['MAXIT'] = 100
                p.driver.opt_settings['ACC'] = 1e-6
            #p.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']

            # Set recording options
            recorder = SqliteRecorder(os.path.join('files', 'ssbj_cr_{}_subsystem_aer.sql'.format(cr_files_key_word)))
            p.driver.add_recorder(recorder)
            p.driver.recording_options['includes'] = []
            p.driver.recording_options['record_objectives'] = True
            p.driver.recording_options['record_constraints'] = True
            p.driver.recording_options['record_desvars'] = True
            p.driver.recording_options['record_metadata'] = True

            # Add design variables
            p.model.add_design_var('x_aer', lower=0.75, upper=1.25)
            p.model.add_design_var('z_hat_aer', lower=np.array([0.2, 0.666, 0.875, 0.45, 0.72, 0.5]),
                                   upper=np.array([1.8, 1.333, 1.125, 1.45, 1.27, 1.5]))

            # Add objective
            p.model.add_objective('J.J')

            # Add constraints
            p.model.add_constraint('constraints.con_dpdx', upper=0.0)

            # Final setup
            p.setup()
            p.final_setup()

            # View model
            view_model(p, outfile=os.path.join('files', 'co_n2_aero.html'), show_browser=False)
        elif self.options['discipline'] == 'propulsion':
            # Add system-level inputs
            self.add_input('z', val=np.ones(6))
            self.add_input('D_hat', val=1.0)
            self.add_input('ESF_hat', val=1.0)
            self.add_input('SFC_hat', val=1.0)
            self.add_input('WE_hat', val=1.0)

            # Add system-level outputs
            self.add_output('z_hat_pro', val=np.ones(6))
            self.add_output('ESF', val=1.0)
            self.add_output('SFC', val=1.0)
            self.add_output('WE', val=1.0)

            # Declare partials
            self.declare_partials('z_hat_pro', ['z', 'D_hat', 'ESF_hat', 'SFC_hat', 'WE_hat'],
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
            des_vars.add_output('z_hat_pro', val=np.ones(6))
            des_vars.add_output('x_pro', val=.5)

            # Define components
            # Disciplinary analysis
            p.model.add_subsystem('propulsion', Propulsion(self.options['scalers']))

            # Local constraint functions
            cnstrnts = ['con_esf = ESF*' + str(self.options['scalers']['ESF']) + '-1.5',
                        'con_temp = Temp*' + str(self.options['scalers']['Temp']) + '-1.02',
                        'con_dt=DT']

            p.model.add_subsystem('constraints', ExecComp(cnstrnts))

            # Local objective
            p.model.add_subsystem('J', ExecComp('J = ((z[1]-z_hat_pro[1])**2 + (z[2]-z_hat_pro[2])**2 + '
                                                '(ESF_hat-ESF)**2 + (WE_hat-WE)**2 + (SFC_hat-SFC)**2)**.5',
                                                z=np.zeros(6), z_hat_pro=np.zeros(6)))

            # Connect variables in sub-problem
            # Aerodynamics component inputs
            p.model.connect('z_hat_pro', 'propulsion.z')
            p.model.connect('x_pro', 'propulsion.x_pro')
            p.model.connect('D_hat', 'propulsion.D')

            # Constraints inputs
            p.model.connect('propulsion.ESF', 'constraints.ESF')
            p.model.connect('propulsion.Temp', 'constraints.Temp')
            p.model.connect('propulsion.DT', 'constraints.DT')

            # Objective
            p.model.connect('z', 'J.z')
            p.model.connect('z_hat_pro', 'J.z_hat_pro')
            p.model.connect('ESF_hat', 'J.ESF_hat')
            p.model.connect('propulsion.ESF', 'J.ESF')
            p.model.connect('WE_hat', 'J.WE_hat')
            p.model.connect('propulsion.WE', 'J.WE')
            p.model.connect('SFC_hat', 'J.SFC_hat')
            p.model.connect('propulsion.SFC', 'J.SFC')

            # Set subproblem optimizer
            if isinstance(self.options['driver'], ScipyOptimizeDriver):
                p.driver = ScipyOptimizeDriver()
                p.driver.options['optimizer'] = 'SLSQP'
                p.driver.options['maxiter'] = 100
                p.driver.options['tol'] = 1e-6
            elif isinstance(self.options['driver'], pyOptSparseDriver):
                p.driver = pyOptSparseDriver()
                p.driver.options['optimizer'] = 'SLSQP'
                p.driver.opt_settings['MAXIT'] = 100
                p.driver.opt_settings['ACC'] = 1e-6
            #p.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']

            # Set recording options
            recorder = SqliteRecorder(os.path.join('files', 'ssbj_cr_{}_subsystem_pro.sql'.format(cr_files_key_word)))
            p.driver.add_recorder(recorder)
            p.driver.recording_options['includes'] = []
            p.driver.recording_options['record_objectives'] = True
            p.driver.recording_options['record_constraints'] = True
            p.driver.recording_options['record_desvars'] = True
            p.driver.recording_options['record_metadata'] = True

            # Add design variables
            p.model.add_design_var('x_pro', lower=0.18, upper=1.81)
            p.model.add_design_var('z_hat_pro', lower=np.array([0.666, 0.875]),
                                   upper=np.array([1.333, 1.125]),
                                   indices=[1,2])

            # Add objective
            p.model.add_objective('J.J')

            # Add constraints
            p.model.add_constraint('constraints.con_esf', upper=0.0, lower=-1.)  # TODO: Turn off to avoid issues
            p.model.add_constraint('constraints.con_temp', upper=0.0)
            p.model.add_constraint('constraints.con_dt', upper=0.0)

            # Final setup
            p.setup()
            p.final_setup()

            # View model
            view_model(p, outfile=os.path.join('files', 'co_n2_prop.html'), show_browser=False)
        else:
            raise IOError('Unknown discipline {} provided in setup function.'.format(self.options['discipline']))

    def compute(self, inputs, outputs):
        p = self.prob
        if self.options['discipline'] == 'structures':
            # Push any global inputs down
            p['z'] = inputs['z']
            p['WE_hat'] = inputs['WE_hat']
            p['WF_hat'] = inputs['WF_hat']
            p['Theta_hat'] = inputs['Theta_hat']
            p['WT_hat'] = inputs['WT_hat']

            # Run the optimization
            print('{} discipline suboptimization'.format(self.options['discipline']))
            p.run_driver()

            # Pull the values back up to the output array
            outputs['z_hat_str'] = p['z_hat_str']
            outputs['WF'] = p['structures.WF']
            outputs['Theta'] = p['structures.Theta']
            outputs['WT'] = p['structures.WT']
        elif self.options['discipline'] == 'aerodynamics':
            # Push any global inputs down
            p['z'] = inputs['z']
            p['ESF_hat'] = inputs['ESF_hat']
            p['WT_hat'] = inputs['WT_hat']
            p['Theta_hat'] = inputs['Theta_hat']
            p['D_hat'] = inputs['D_hat']
            p['fin_hat'] = inputs['fin_hat']

            # Run the optimization
            print('{} discipline suboptimization'.format(self.options['discipline']))
            p.run_driver()

            # Pull the values back up to the output array
            outputs['z_hat_aer'] = p['z_hat_aer']
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
            outputs['z_hat_pro'] = p['z_hat_pro']
            outputs['ESF'] = p['propulsion.ESF']
            outputs['SFC'] = p['propulsion.SFC']
            outputs['WE'] = p['propulsion.WE']
        else:
            raise IOError('Unknown discipline {} provided in setup function.'.format(self.options['discipline']))


class SsbjCO(Group):
    """Main group for the SSBJ case to run it using Collaborative Optimization."""
    def initialize(self):
        self.options.declare('scalers')
        self.options.declare('subopt_driver')

    def setup(self):
        # Define system-level design variables
        des_vars = self.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        # Add global design variables
        des_vars.add_output('z', val=np.ones(6))
        # Add coupling copies for interdisciplinary couplings (N.B. L_hat is not used, but L_hat = W_hat at this level)
        des_vars.add_output('D_hat', val=1.)
        des_vars.add_output('WE_hat', val=1.)
        des_vars.add_output('WT_hat', val=1.)
        des_vars.add_output('Theta_hat', val=1.)
        des_vars.add_output('WF_hat', val=1.)
        des_vars.add_output('ESF_hat', val=1.)
        des_vars.add_output('fin_hat', val=1.)
        des_vars.add_output('SFC_hat', val=1.)

        # Add suboptimizations
        self.add_subsystem('subopt_struc', SubOpt(discipline='structures',
                                                  scalers=self.options['scalers'],
                                                  driver=self.options['subopt_driver']))
        self.add_subsystem('subopt_aero', SubOpt(discipline='aerodynamics',
                                                 scalers=self.options['scalers'],
                                                 driver=self.options['subopt_driver']))
        self.add_subsystem('subopt_prop', SubOpt(discipline='propulsion',
                                                 scalers=self.options['scalers'],
                                                 driver=self.options['subopt_driver']))

        # Add system-level analyses
        self.add_subsystem('performance', Performance(self.options['scalers']))
        J_tot_expr = 'J = ((z[0]-z_hat_struc[0])**2 + (z[3]-z_hat_struc[3])**2 + (z[4]-z_hat_struc[4])**2 ' \
                     '+ (z[5]-z_hat_struc[5])**2 + (WF_hat-WF_struc)**2 + (Theta_hat-Theta_struc)**2 ' \
                     '+ (WT_hat-WT_struc)**2 + sum((z-z_hat_aero)**2) + (fin_hat-fin_aero)**2 + (D_hat-D_aero)**2 ' \
                     '+ (WT_hat-L_aero)**2 + (z[1]-z_hat_prop[1])**2 + (z[2]-z_hat_prop[2])**2 + (ESF_hat-ESF_prop)**2' \
                     ' + (WE_hat-WE_prop)**2 + (SFC_hat-SFC_prop)**2)**.5'
        self.add_subsystem('J', ExecComp(J_tot_expr, z=np.ones(6), z_hat_struc=np.ones(6),
                                         z_hat_aero=np.ones(6), z_hat_prop=np.ones(6)))

        # Connect variables
        self.connect('z', ['subopt_struc.z', 'subopt_aero.z', 'subopt_prop.z', 'performance.z', 'J.z'])
        self.connect('D_hat', ['subopt_aero.D_hat', 'subopt_prop.D_hat', 'J.D_hat'])
        self.connect('WE_hat', ['subopt_struc.WE_hat', 'subopt_prop.WE_hat', 'J.WE_hat'])
        self.connect('WT_hat', ['performance.WT', 'subopt_struc.WT_hat', 'subopt_aero.WT_hat', 'J.WT_hat'])
        self.connect('Theta_hat', ['subopt_struc.Theta_hat', 'subopt_aero.Theta_hat', 'J.Theta_hat'])
        self.connect('WF_hat', ['performance.WF', 'subopt_struc.WF_hat', 'J.WF_hat'])
        self.connect('ESF_hat', ['subopt_aero.ESF_hat', 'subopt_prop.ESF_hat', 'J.ESF_hat'])
        self.connect('fin_hat', ['performance.fin', 'subopt_aero.fin_hat', 'J.fin_hat'])
        self.connect('SFC_hat', ['performance.SFC', 'subopt_prop.SFC_hat', 'J.SFC_hat'])
        self.connect('subopt_struc.z_hat_str', ['J.z_hat_struc'])
        self.connect('subopt_struc.WF', ['J.WF_struc'])
        self.connect('subopt_struc.Theta', ['J.Theta_struc'])
        self.connect('subopt_struc.WT', ['J.WT_struc'])
        self.connect('subopt_aero.z_hat_aer', ['J.z_hat_aero'])
        self.connect('subopt_aero.fin', ['J.fin_aero'])
        self.connect('subopt_aero.D', ['J.D_aero'])
        self.connect('subopt_aero.L', ['J.L_aero'])
        self.connect('subopt_prop.z_hat_pro', ['J.z_hat_prop'])
        self.connect('subopt_prop.ESF', ['J.ESF_prop'])
        self.connect('subopt_prop.WE', ['J.WE_prop'])
        self.connect('subopt_prop.SFC', ['J.SFC_prop'])


if __name__ == '__main__':

    # Initialize problem
    scalers = init_ssbj_mda()
    prob = Problem()

    subopt_driver = ScipyOptimizeDriver()

    prob.model = model = SsbjCO(scalers=scalers, subopt_driver=subopt_driver)

    if isinstance(subopt_driver, ScipyOptimizeDriver):
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.opt_settings['ACC'] = 1e-3
        prob.driver.opt_settings['MAXIT'] = 200
    elif isinstance(subopt_driver, pyOptSparseDriver):
        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['maxiter'] = 200
        prob.driver.options['tol'] = 1e-3
    prob.driver.options['debug_print'] = ['desvars', 'ln_cons', 'nl_cons', 'objs']

    # Set design variables
    prob.model.add_design_var('z', lower=np.array([0.2, 0.666, 0.875, 0.45, 0.72, 0.5]),
                              upper=np.array([1.8, 1.333, 1.125, 1.45, 1.27, 1.5]))

    # Add some logical, though conservative, bounds to the coupling variables that have become design variables
    des_vars_def = {'D_hat':[2500, 15000],      # The lower bound on drag is essential for problems in the optimizer
                    'WE_hat': [0, 20000],
                    'WT_hat': [20000, 60000],
                    'Theta_hat': [0.96, 1.04],
                    'WF_hat': [5000, 25000],
                    'ESF_hat': [0.5, 1.5],
                    'fin_hat': [2, 12],
                    'SFC_hat': [0.5, 1.5]}

    for key, item in des_vars_def.items():
        prob.model.add_design_var(key, lower=item[0]/scalers[key.replace('_hat', '')],
                                  upper=item[1]/scalers[key.replace('_hat', '')])

    # Set objective
    prob.model.add_objective('performance.R', scaler=-1.)

    # Set constraints
    prob.model.add_constraint('J.J', equals=0.0)

    # Add recorder
    recorder = SqliteRecorder(os.path.join('files', 'ssbj_cr_{}_co_system.sql'.format(cr_files_key_word)))
    prob.driver.add_recorder(recorder)
    prob.driver.recording_options['includes'] = []
    prob.driver.recording_options['record_objectives'] = True
    prob.driver.recording_options['record_constraints'] = True
    prob.driver.recording_options['record_desvars'] = True
    prob.driver.recording_options['record_metadata'] = True

    # Setup
    prob.setup(mode='rev')

    # View model
    view_model(prob, outfile=os.path.join('files', 'co_sys_ssbj.html'), show_browser=False)

    # Check partials
    # prob.check_partials(compact_print=True)

    # Run problem (either once (run_model) or full optimization (run_driver))
    # prob.run_model()
    prob.run_driver()

    # Report result in the log
    print('Outcome of analysis:')
    print('\nDesign variables:')
    print('Z_opt=', prob['z'] * scalers['z'])
    print('Z_opt_c=', [[key, float(prob[key] * scalers[key.replace('_hat', '')])] for key in des_vars_def.keys()])
    print('X_str_opt=', prob.model.subopt_struc.prob['x_str'])
    print('X_aer_opt=', prob.model.subopt_aero.prob['x_aer'])
    print('X_pro_opt=', prob.model.subopt_prop.prob['x_pro'])

    print('\nObjectives')
    print('R_opt=', prob['performance.R'] * scalers['R'])
    print('J_opt_str=', prob.model.subopt_struc.prob['J.J'])
    print('J_opt_aer=', prob.model.subopt_aero.prob['J.J'])
    print('J_opt_pro=', prob.model.subopt_prop.prob['J.J'])

    print('\nConstraints')
    print('J_sys=', prob['J.J'])
    for i in range(1, 6):
        print('con_sigma{}='.format(i), prob.model.subopt_struc.prob['con_sigma{}'.format(i)])
    print('con_theta=', prob.model.subopt_struc.prob['con_theta'])
    print('con_dpdx=', prob.model.subopt_aero.prob['constraints.con_dpdx'])
    print('con_dt=', prob.model.subopt_prop.prob['constraints.con_dt'])
    print('con_esf=', prob.model.subopt_prop.prob['constraints.con_esf'])
    print('con_temp=', prob.model.subopt_prop.prob['constraints.con_temp'])
