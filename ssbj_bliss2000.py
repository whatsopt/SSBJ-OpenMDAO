"""
SSBJ test case - http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Python implementation and OpenMDAO integration developed by
Sylvain Dubreuil and Remi Lafage of ONERA, the French Aerospace Lab.
BLISS-2000 strategy optimization and postprocessing scripts
developed by Imco van Gent of TU Delft, Faculty of Aerospace Engineering.
"""
import copy
import pickle
import math
import warnings
from collections import OrderedDict

from openmdao.api import *

import numpy as np

from disciplines.aerodynamics import Aerodynamics
from disciplines.performance import Performance
from disciplines.propulsion import Propulsion
from disciplines.structure import Structure
from disciplines.dpdxcalc import DpdxCalc

from ssbj_mda import init_ssbj_mda

# Main execution settings
# Save settings for output files
cr_files_keyword = 'bliss_run'   # keyword for files to be saved
cr_files_folder = 'files'        # name of folder to save execution files

# BLISS algorithm settings
F_SAMPLES = 15                   # LHS sample factor (N_SAMPLES = F_SAMPLES*number_of_variables)
MAX_LOOPS = 30                   # maximum number of BLISS iteration loops
CONV_ABS_TOL = 1e-3              # Absolute convergence tolerance for BLISS iterations
CONV_REL_TOL = 1e-3              # Relative convergence tolerance for BLISS iterations
LHS_SEED = 4                     # Seed of the Latin Hypercube Sampling algorithm

# BLISS design variables interval adjustment settings
F_K_RED = 2.0                    # K_bound_reduction: K-factor reduction
F_INT_INC = 0.25                 # interval increase: percentage of interval increase if bound is hit
F_INT_INC_ABS = 0.1              # absolute interval increase: minimum increase if percentual increase is too low
F_INT_RANGE = 1.e-3              # minimal range of the design variable interval

# Restart settings (last three are only needed when START_TYPE == 'restart')
START_TYPE = 'fresh'             # start based on previous (restart) or afresh (fresh)
RESTART_FOLDER = 'files'         # folder name to look for previous results
RESTART_KEYWORD = 'bliss_run'    # keyword of file names for previous results (see cr_files_keyword above)
RESTART_FROM_LOOP = 7            # Loop number to restart the run from


class SubOpt(ExplicitComponent):
    """Suboptimization component for the BLISS approach. In this class the suboptimizations of the three disciplines
    are carried out w.r.t. the local design variables."""

    def initialize(self):
        self.options.declare('discipline')
        self.options.declare('scalers')
        self.options.declare('driver')

    def setup(self):
        if self.options['discipline'] == 'structures':
            # Add system-level inputs
            self.add_input('tc_hat', val=1.0)
            self.add_input('AR_hat', val=1.0)
            self.add_input('Lambda_hat', val=1.0)
            self.add_input('Sref_hat', val=1.0)
            self.add_input('WE_hat', val=1.0)
            self.add_input('L_hat', val=1.0)
            self.add_input('w_Theta', val=1.0)
            self.add_input('w_WT', val=1.0)

            # Add system-level outputs
            self.add_output('WF', val=1.0)
            self.add_output('Theta', val=1.0)
            self.add_output('WT', val=1.0)

            # Declare partials
            self.declare_partials('*', '*', method='fd')

            # Set subproblem
            self.prob = p = Problem()

            # Define the copies so that OpenMDAO can compute derivatives w.r.t. these variables
            params = p.model.add_subsystem('params', IndepVarComp(), promotes=['*'])
            params.add_output('tc_hat', val=1.0)
            params.add_output('AR_hat', val=1.0)
            params.add_output('Lambda_hat', val=1.0)
            params.add_output('Sref_hat', val=1.0)
            params.add_output('WE_hat', val=1.0)
            params.add_output('L_hat', val=1.0)
            params.add_output('w_Theta', val=1.0)
            params.add_output('w_WT', val=1.0)

            # Define design variables of subproblem
            des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
            des_vars.add_output('x_str', val=np.array([1.6, 0.75]))

            # Define components
            # Map inputs from z-vector to a single z-vector for the Structure component
            z_mappings = ['z[0] = tc_hat',
                          'z[1] = 0.0',
                          'z[2] = 0.0',
                          'z[3] = AR_hat',
                          'z[4] = Lambda_hat',
                          'z[5] = Sref_hat']
            p.model.add_subsystem('map_design_vector', ExecComp(z_mappings, z=np.ones(6)))

            # Disciplinary analysis
            p.model.add_subsystem('structures', Structure(self.options['scalers']))

            # Local constraint functions
            cstrs = ['con_theta = Theta*' + str(self.options['scalers']['Theta']) + '-1.04']
            for i in range(5):
                cstrs.append('con_sigma' + str(i + 1) + ' = sigma[' + str(i) + ']*' +
                             str(self.options['scalers']['sigma'][i]) + '-1.09')
            p.model.add_subsystem('constraints', ExecComp(cstrs, sigma=np.zeros(5)), promotes_outputs=['*'])

            # Local objective
            p.model.add_subsystem('WCF', ExecComp('WCF = w_Theta*Theta + w_WT*WT'))

            # Connect variables in sub-problem
            # Mapping component inputs
            p.model.connect('tc_hat', 'map_design_vector.tc_hat')
            p.model.connect('AR_hat', 'map_design_vector.AR_hat')
            p.model.connect('Lambda_hat', 'map_design_vector.Lambda_hat')
            p.model.connect('Sref_hat', 'map_design_vector.Sref_hat')

            # Structures component inputs
            p.model.connect('map_design_vector.z', 'structures.z')
            p.model.connect('x_str', 'structures.x_str')
            p.model.connect('L_hat', 'structures.L')
            p.model.connect('WE_hat', 'structures.WE')

            # Constraints inputs
            p.model.connect('structures.Theta', 'constraints.Theta')
            p.model.connect('structures.sigma', 'constraints.sigma')

            # Objective
            p.model.connect('w_Theta', 'WCF.w_Theta')
            p.model.connect('structures.Theta', 'WCF.Theta')
            p.model.connect('w_WT', 'WCF.w_WT')
            p.model.connect('structures.WT', 'WCF.WT')

            # Set subproblem optimizer
            if isinstance(self.options['driver'], ScipyOptimizeDriver):
                p.driver = ScipyOptimizeDriver()
                p.driver.options['optimizer'] = 'SLSQP'
                p.driver.options['maxiter'] = 25
                p.driver.options['tol'] = 1e-6
                p.driver.options['disp'] = False
            elif isinstance(self.options['driver'], pyOptSparseDriver):
                p.driver = pyOptSparseDriver()
                p.driver.options['optimizer'] = 'SLSQP'
                p.driver.opt_settings['MAXIT'] = 25
                p.driver.opt_settings['ACC'] = 1e-6

            # Set recording options
            recorder = SqliteRecorder(os.path.join(cr_files_folder, 'ssbj_cr_{}_subsystem_str.sql'.format(cr_files_keyword)))
            p.driver.add_recorder(recorder)
            p.driver.recording_options['record_objectives'] = True
            p.driver.recording_options['record_constraints'] = True
            p.driver.recording_options['record_desvars'] = True
            p.driver.recording_options['record_metadata'] = True

            # Add design variables
            p.model.add_design_var('x_str', lower=np.array([0.4, 0.75]), upper=np.array([1.6, 1.25]))

            # Add objective
            p.model.add_objective('WCF.WCF')

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
            view_model(p, outfile=os.path.join(cr_files_folder, 'bliss2000_subopt_n2_str.html'), show_browser=False)

        elif self.options['discipline'] == 'aerodynamics':
            self.add_input('tc_hat', val=1.0)
            self.add_input('h_hat', val=1.0)
            self.add_input('M_hat', val=1.0)
            self.add_input('AR_hat', val=1.0)
            self.add_input('Lambda_hat', val=1.0)
            self.add_input('Sref_hat', val=1.0)
            self.add_input('WT_hat', val=1.0)
            self.add_input('ESF_hat', val=1.0)
            self.add_input('Theta_hat', val=1.0)
            self.add_input('w_D', val=1.0)
            self.add_input('w_L', val=1.0)

            # Add system-level outputs
            self.add_output('D', val=1.0)
            self.add_output('L', val=1.0)
            self.add_output('fin', val=1.0)

            # Declare partials
            self.declare_partials('*', '*', method='fd')

            # Set subproblem
            self.prob = p = Problem()

            # Define the copies so that OpenMDAO can compute derivatives w.r.t. these variables
            params = p.model.add_subsystem('params', IndepVarComp(), promotes=['*'])
            params.add_output('tc_hat', val=1.0)
            params.add_output('h_hat', val=1.0)
            params.add_output('M_hat', val=1.0)
            params.add_output('AR_hat', val=1.0)
            params.add_output('Lambda_hat', val=1.0)
            params.add_output('Sref_hat', val=1.0)
            params.add_output('WT_hat', val=1.0)
            params.add_output('ESF_hat', val=1.0)
            params.add_output('Theta_hat', val=1.0)
            params.add_output('w_D', val=1.0)
            params.add_output('w_L', val=1.0)

            # Define design variables of subproblem
            des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
            des_vars.add_output('x_aer', val=1.0)

            # Define components
            # Map inputs from z-vector to single z-vector for the Aerodynamics component
            z_mappings = ['z[0] = tc_hat',
                          'z[1] = h_hat',
                          'z[2] = M_hat',
                          'z[3] = AR_hat',
                          'z[4] = Lambda_hat',
                          'z[5] = Sref_hat']
            p.model.add_subsystem('map_design_vector', ExecComp(z_mappings, z=np.ones(6)))

            # Disciplinary analysis
            p.model.add_subsystem('aerodynamics', Aerodynamics(self.options['scalers']))

            # Local constraint functions -> N.B. The dpdx constraint is moved to the system-level
            #p.model.add_subsystem('constraints',
            #                      ExecComp('con_dpdx = dpdx*' + str(self.options['scalers']['dpdx']) + '-1.04'))

            # Local objective
            p.model.add_subsystem('WCF', ExecComp('WCF = w_D*D + w_L*L'))

            # Connect variables in sub-problem
            # Mapping component inputs
            p.model.connect('tc_hat', 'map_design_vector.tc_hat')
            p.model.connect('h_hat', 'map_design_vector.h_hat')
            p.model.connect('M_hat', 'map_design_vector.M_hat')
            p.model.connect('AR_hat', 'map_design_vector.AR_hat')
            p.model.connect('Lambda_hat', 'map_design_vector.Lambda_hat')
            p.model.connect('Sref_hat', 'map_design_vector.Sref_hat')

            # Aerodynamics component inputs
            p.model.connect('map_design_vector.z', 'aerodynamics.z')
            p.model.connect('x_aer', 'aerodynamics.x_aer')
            p.model.connect('ESF_hat', 'aerodynamics.ESF')
            p.model.connect('WT_hat', 'aerodynamics.WT')
            p.model.connect('Theta_hat', 'aerodynamics.Theta')

            # Constraints inputs -> N.B. The dpdx constraint is moved to the system-level
            # p.model.connect('aerodynamics.dpdx', 'constraints.dpdx')

            # Objective
            p.model.connect('w_D', 'WCF.w_D')
            p.model.connect('aerodynamics.D', 'WCF.D')
            p.model.connect('w_L', 'WCF.w_L')
            p.model.connect('aerodynamics.L', 'WCF.L')

            # Set subproblem optimizer
            if isinstance(self.options['driver'], ScipyOptimizeDriver):
                p.driver = ScipyOptimizeDriver()
                p.driver.options['optimizer'] = 'SLSQP'
                p.driver.options['maxiter'] = 25
                p.driver.options['tol'] = 1e-6
                p.driver.options['disp'] = False
            elif isinstance(self.options['driver'], pyOptSparseDriver):
                p.driver = pyOptSparseDriver()
                p.driver.options['optimizer'] = 'SLSQP'
                p.driver.opt_settings['MAXIT'] = 25
                p.driver.opt_settings['ACC'] = 1e-6

            # Set recording options
            recorder = SqliteRecorder(os.path.join(cr_files_folder,
                                                   'ssbj_cr_{}_subsystem_aer.sql'.format(cr_files_keyword)))
            p.driver.add_recorder(recorder)
            p.driver.recording_options['record_objectives'] = True
            p.driver.recording_options['record_constraints'] = True
            p.driver.recording_options['record_desvars'] = True
            p.driver.recording_options['record_metadata'] = True

            # Add design variables
            p.model.add_design_var('x_aer', lower=0.75, upper=1.25)

            # Add objective
            p.model.add_objective('WCF.WCF')

            # Add constraints -> N.B. The dpdx constraint is moved to the system-level
            # p.model.add_constraint('constraints.con_dpdx', upper=0.0)

            # Final setup
            p.setup()
            p.final_setup()

            # View model
            view_model(p, outfile=os.path.join(cr_files_folder, 'bliss2000_subopt_n2_aer.html'), show_browser=False)
        elif self.options['discipline'] == 'propulsion':
            # Add system-level inputs
            self.add_input('h_hat', val=1.0)
            self.add_input('M_hat', val=1.0)
            self.add_input('D_hat', val=1.0)
            self.add_input('w_WE', val=1.0)
            self.add_input('w_ESF', val=1.0)

            # Add system-level outputs
            self.add_output('ESF', val=1.0)
            self.add_output('SFC', val=1.0)
            self.add_output('WE', val=1.0)

            # Declare partials
            self.declare_partials('*', '*', method='fd')

            # Set subproblem
            self.prob = p = Problem()

            # Define the copies so that OpenMDAO can compute derivs w.r.t. these variables
            params = p.model.add_subsystem('params', IndepVarComp(), promotes=['*'])
            params.add_output('h_hat', val=1.0)
            params.add_output('M_hat', val=1.0)
            params.add_output('D_hat', val=1.0)
            params.add_output('w_WE', val=1.0)
            params.add_output('w_ESF', val=1.0)

            # Define design variables of subproblem
            des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
            des_vars.add_output('x_pro', val=.5)

            # Define components
            # Map inputs from z-vector to single z-vector for the Propulsion component
            z_mappings = ['z[0] = 0.',
                          'z[1] = h_hat',
                          'z[2] = M_hat',
                          'z[3] = 0.',
                          'z[4] = 0.',
                          'z[5] = 0.']
            p.model.add_subsystem('map_design_vector', ExecComp(z_mappings, z=np.ones(6)))

            # Disciplinary analysis
            p.model.add_subsystem('propulsion', Propulsion(self.options['scalers']))

            # Local constraint functions
            cnstrnts = ['con_esf = ESF*' + str(self.options['scalers']['ESF']) + '-1.5',
                        'con_temp = Temp*' + str(self.options['scalers']['Temp']) + '-1.02',
                        'con_dt=DT']

            p.model.add_subsystem('constraints', ExecComp(cnstrnts))

            # Local objective
            p.model.add_subsystem('WCF', ExecComp('WCF = w_WE*WE + w_ESF*ESF'))

            # Connect variables in sub-problem
            # Mapping component inputs
            p.model.connect('h_hat', 'map_design_vector.h_hat')
            p.model.connect('M_hat', 'map_design_vector.M_hat')

            # Aerodynamics component inputs
            p.model.connect('map_design_vector.z', 'propulsion.z')
            p.model.connect('x_pro', 'propulsion.x_pro')
            p.model.connect('D_hat', 'propulsion.D')

            # Constraints inputs
            p.model.connect('propulsion.ESF', 'constraints.ESF')
            p.model.connect('propulsion.Temp', 'constraints.Temp')
            p.model.connect('propulsion.DT', 'constraints.DT')

            # Objective
            p.model.connect('w_WE', 'WCF.w_WE')
            p.model.connect('propulsion.WE', 'WCF.WE')
            p.model.connect('w_ESF', 'WCF.w_ESF')
            p.model.connect('propulsion.ESF', 'WCF.ESF')

            # Set subproblem optimizer
            if isinstance(self.options['driver'], ScipyOptimizeDriver):
                p.driver = ScipyOptimizeDriver()
                p.driver.options['optimizer'] = 'SLSQP'
                p.driver.options['maxiter'] = 25
                p.driver.options['tol'] = 1e-6
                p.driver.options['disp'] = False
            elif isinstance(self.options['driver'], pyOptSparseDriver):
                p.driver = pyOptSparseDriver()
                p.driver.options['optimizer'] = 'SLSQP'
                p.driver.opt_settings['MAXIT'] = 25
                p.driver.opt_settings['ACC'] = 1e-6

            # Set recording options
            recorder = SqliteRecorder(os.path.join(cr_files_folder,
                                                   'ssbj_cr_{}_subsystem_pro.sql'.format(cr_files_keyword)))
            p.driver.add_recorder(recorder)
            p.driver.recording_options['record_objectives'] = True
            p.driver.recording_options['record_constraints'] = True
            p.driver.recording_options['record_desvars'] = True
            p.driver.recording_options['record_metadata'] = True

            # Add design variables
            p.model.add_design_var('x_pro', lower=0.18, upper=1.81)

            # Add objective
            p.model.add_objective('WCF.WCF')

            # Add constraints
            p.model.add_constraint('constraints.con_esf', upper=0.0, lower=-1.)
            p.model.add_constraint('constraints.con_temp', upper=0.0)
            p.model.add_constraint('constraints.con_dt', upper=0.0)

            # Final setup
            p.setup()
            p.final_setup()

            # View model
            view_model(p, outfile=os.path.join(cr_files_folder, 'bliss2000_subopt_n2_pro.html'), show_browser=False)
        else:
            raise IOError('Unknown discipline {} provided in setup function.'.format(self.options['discipline']))

    def compute(self, inputs, outputs):
        p = self.prob
        if self.options['discipline'] == 'structures':
            # Push any global inputs down
            p['tc_hat'] = inputs['tc_hat']
            p['AR_hat'] = inputs['AR_hat']
            p['Lambda_hat'] = inputs['Lambda_hat']
            p['Sref_hat'] = inputs['Sref_hat']
            p['WE_hat'] = inputs['WE_hat']
            p['L_hat'] = inputs['L_hat']
            p['w_Theta'] = inputs['w_Theta']
            p['w_WT'] = inputs['w_WT']

            # Run the optimization
            p.run_driver()

            # Pull the values back up to the output array
            if not p.driver.fail:
                outputs['WF'] = p['structures.WF']
                outputs['Theta'] = p['structures.Theta']
                outputs['WT'] = p['structures.WT']
            else:
                outputs['WF'] = float('nan')
                outputs['Theta'] = float('nan')
                outputs['WT'] = float('nan')
                clean_driver_for_next_run(p)

        elif self.options['discipline'] == 'aerodynamics':
            # Push any global inputs down
            p['tc_hat'] = inputs['tc_hat']
            p['h_hat'] = inputs['h_hat']
            p['M_hat'] = inputs['M_hat']
            p['AR_hat'] = inputs['AR_hat']
            p['Lambda_hat'] = inputs['Lambda_hat']
            p['Sref_hat'] = inputs['Sref_hat']
            p['WT_hat'] = inputs['WT_hat']
            p['ESF_hat'] = inputs['ESF_hat']
            p['Theta_hat'] = inputs['Theta_hat']
            p['w_D'] = inputs['w_D']
            p['w_L'] = inputs['w_L']

            # Run the optimization
            p.run_driver()

            # Pull the values back up to the output array
            if not p.driver.fail:
                outputs['L'] = p['aerodynamics.L']
                outputs['fin'] = p['aerodynamics.fin']
                outputs['D'] = p['aerodynamics.D']
            else:
                outputs['L'] = float('nan')
                outputs['fin'] = float('nan')
                outputs['D'] = float('nan')
                clean_driver_for_next_run(p)
        elif self.options['discipline'] == 'propulsion':
            # Push any global inputs down
            p['h_hat'] = inputs['h_hat']
            p['M_hat'] = inputs['M_hat']
            p['D_hat'] = inputs['D_hat']
            p['w_WE'] = inputs['w_WE']
            p['w_ESF'] = inputs['w_ESF']

            # Run the optimization
            p.run_driver()

            # Pull the values back up to the output array
            if not p.driver.fail:
                outputs['ESF'] = p['propulsion.ESF']
                outputs['SFC'] = p['propulsion.SFC']
                outputs['WE'] = p['propulsion.WE']
            else:
                outputs['ESF'] = float('nan')
                outputs['SFC'] = float('nan')
                outputs['WE'] = float('nan')
                clean_driver_for_next_run(p)
        else:
            raise IOError('Unknown discipline {} provided in setup function.'.format(self.options['discipline']))


class SsbjBLISS2000(Group):
    """Main group for the SSBJ case to run it using the BLISS-2000 strategy. In this group the overall system is
    optimized using the surrogates with optimized disciplines."""
    def initialize(self):
        self.options.declare('des_vars')
        self.options.declare('subsystems')
        self.options.declare('scalers')
        self.options.declare('loop_number')

    def setup(self):
        # Define system-level design variables
        des_vars = self.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        for des_var, details in self.options['des_vars'].items():
            des_vars.add_output(des_var, val=details['nominal'])

        # Add surrogate models for each subsystem
        for discipline, details in self.options['subsystems'].items():
            # Get the relevant design variables
            relevant_des_vars, relevant_qois = get_des_vars_and_qois(discipline)
            x_len = sum([len(relevant_des_vars[key]) for key in relevant_des_vars.keys()])

            # First add mapping between design variables vector and surrogate models
            mapping_eqs = []
            x_ind = 0
            for des_var, indices in relevant_des_vars.items():
                for ind in indices:
                    mapping_eqs.append('x_sm[{}] = {}[{}]'.format(x_ind, des_var, ind))
                    x_ind += 1
            self.add_subsystem('vector_mapping_{}'.format(discipline),
                               ExecComp(mapping_eqs,
                                        x_sm=np.zeros(x_len),
                                        z_sh=np.zeros(len(self.options['des_vars']['z_sh']['nominal'])),
                                        z_c=np.zeros(len(self.options['des_vars']['z_c']['nominal'])),
                                        z_w=np.zeros(len(self.options['des_vars']['z_w']['nominal']))),
                               promotes_outputs=[('x_sm', 'x_sm_{}'.format(discipline[:3]))], promotes_inputs=['*'])

            # Then add the surrogate model
            self.add_subsystem('sm_{}'.format(discipline),
                               details['surrogate_model'][self.options['loop_number']],
                               promotes_inputs=[('x', 'x_sm_{}'.format(discipline[:3]))])

        # Add system-level analyses
        # Performance analysis
        self.add_subsystem('performance', Performance(self.options['scalers']),
                           promotes_inputs=[('z', 'z_sh')])
        # Consistency constraints
        cons_cons_eqs = ['gc_WE = WE_opt - WE_sm',
                         'gc_D = D_opt - D_sm',
                         'gc_WT = WT_opt - WT_sm',
                         'gc_L = L_opt - L_sm',
                         'gc_Theta = Theta_opt - Theta_sm',
                         'gc_ESF = ESF_opt - ESF_sm',
                         'gc_WT_L = WT_opt - L_opt']
        self.add_subsystem('consistency_constraints', ExecComp(cons_cons_eqs))

        # dpdx constraint at system level
        self.add_subsystem('dpdxcalc', DpdxCalc(self.options['scalers']))
        self.add_subsystem('constraints', ExecComp('con_dpdx = dpdx*' + str(self.options['scalers']['dpdx']) + '-1.04'))
        self.connect('z_sh', 'dpdxcalc.z0', src_indices=[0])
        self.connect('dpdxcalc.dpdx', 'constraints.dpdx')

        # Connect variables correctly
        # between surrogate models and performance block
        self.connect('sm_structures.y', 'performance.WT', src_indices=[0])
        self.connect('sm_structures.y', 'performance.WF', src_indices=[1])
        self.connect('sm_aerodynamics.y', 'performance.fin', src_indices=[2])
        self.connect('sm_propulsion.y', 'performance.SFC', src_indices=[2])

        # between design variables and consistency constraint block
        self.connect('z_c', 'consistency_constraints.D_opt', src_indices=[0])
        self.connect('z_c', 'consistency_constraints.WE_opt', src_indices=[1])
        self.connect('z_c', 'consistency_constraints.WT_opt', src_indices=[2])
        self.connect('z_c', 'consistency_constraints.Theta_opt', src_indices=[3])
        self.connect('z_c', 'consistency_constraints.ESF_opt', src_indices=[4])
        self.connect('z_c', 'consistency_constraints.L_opt', src_indices=[5])

        # between surrogate models and consistency constraint block
        self.connect('sm_aerodynamics.y', 'consistency_constraints.L_sm', src_indices=[0])
        self.connect('sm_aerodynamics.y', 'consistency_constraints.D_sm', src_indices=[1])
        self.connect('sm_structures.y', 'consistency_constraints.WT_sm', src_indices=[0])
        self.connect('sm_structures.y', 'consistency_constraints.Theta_sm', src_indices=[2])
        self.connect('sm_propulsion.y', 'consistency_constraints.WE_sm', src_indices=[0])
        self.connect('sm_propulsion.y', 'consistency_constraints.ESF_sm', src_indices=[1])


def clean_driver_for_next_run(p):
    """Method to clean the driver of an OpenMDAO Probem() object. This is done if the driver (optimization) has failed
    and nan (not a number) values are stored in the inputs and outputs.

    :param p: OpenMDAO problem object with a driver
    :type p: Problem
    """
    for inp in p.model.list_inputs(out_stream=None):
        if np.isnan(np.min(inp[1]['value'])):
            p[inp[0]] = np.ones(len(inp[1]['value']))
    for out in p.model.list_outputs(out_stream=None):
        if np.isnan(np.min(out[1]['value'])):
            p[out[0]] = np.ones(len(out[1]['value']))


def set_initial_values(start_type, cr_file_folder_name='files', cr_file_key_word='bliss_removews3', n_loop=24):
    """
    Method to set the initial values for the BLISS iterations. Here either the values can be specified for a fresh run
    or the values and bounds from a previous run can be used when a restart should be done.

    :param start_type: specify a fresh start or a restart based on previous results
    :type start_type: basestring
    :param cr_file_folder_name: name of the folder containing restart files
    :type cr_file_folder_name: basestring
    :param cr_file_key_word: keyword of the restart files
    :type cr_file_key_word: basestring
    :param n_loop: loop at which the restart should be taken from in the restart files
    :type n_loop: int
    :return: dictionary containing the design variable definitions
    :rtype: dict
    """
    if start_type == 'fresh':
        # Set initial shared design variables and its bounds
        # Variables:    z = (t/c, h, M, AR, Lambda, Sref)
        z_sh = np.ones(6)
        # Bounds: z_lower = [0.2, 0.666, 0.875, 0.45, 0.72, 0.5]
        z_sh_lower = np.array([0.2, 0.666, 0.875, 0.45, 0.72, 0.5])
        #         z_upper = [1.8, 1.333, 1.125, 1.45, 1.27, 1.5]
        z_sh_upper = np.array([1.8, 1.333, 1.125, 1.45, 1.27, 1.5])
        # Also set absolute minimum and maximum values of design variables
        z_sh_min = copy.deepcopy(z_sh_lower)
        z_sh_max = copy.deepcopy(z_sh_upper)

        # Set coupling values and bounds
        # Add some logi cal, though conservative, bounds to the coupling variables that have become design variables
        z_c_def = OrderedDict()
        z_c_def['D'] = [1000, 15000, 0, np.inf]  # 0
        z_c_def['WE'] = [0, 20000, 0, np.inf]  # 1
        z_c_def['WT'] = [20000, 60000, 0, np.inf]  # 2
        z_c_def['Theta'] = [0.96, 1.04, 0.96, 1.04]  # 3
        z_c_def['ESF'] = [0.5, 1.5, 0.5, 1.5]  # 4
        z_c_def['L'] = [20000, 60000, 0, np.inf]  # 5
        z_c = np.array([0.4, 1.5, 1., 1., 1.5, 1.])
        z_c_lower = np.reshape(np.asarray([item[0] / scalers[key] for key, item in z_c_def.items()]), (len(z_c_def)))
        z_c_upper = np.reshape(np.asarray([item[1] / scalers[key] for key, item in z_c_def.items()]), (len(z_c_def)))
        z_c_min = np.reshape(np.asarray([item[2] / scalers[key] for key, item in z_c_def.items()]), (len(z_c_def)))
        z_c_max = np.reshape(np.asarray([item[3] / scalers[key] for key, item in z_c_def.items()]), (len(z_c_def)))

        # Set weights and bounds
        z_w_def = ['w_D',  # 0
                   'w_WE',  # 1
                   'w_WT',  # 2
                   'w_Theta',  # 3
                   'w_ESF',  # 4
                   'w_L']  # 5
        # Variables
        z_w = np.array([1., 1., 1., 1., 1., 1.])
        z_w_lower = np.array([-2., -2., -2., -2., -2., -2.])
        z_w_upper = np.array([2., 2., 2., 2., 2., 2.])

        # Also set absolute minimum and maximum values of design variables
        z_w_min = -np.ones(len(z_w_def)) * np.inf
        z_w_max = np.ones(len(z_w_def)) * np.inf

    elif start_type == 'restart':
        # Pick up values from recorder
        des_vars_list = pickle.load(open(os.path.join(cr_file_folder_name,
                                                      'ssbj_des_vars_{}_system_loops.p'.format(cr_file_key_word)), 'rb')
                                    )
        cr_sys = CaseReader(
            os.path.join(cr_file_folder_name, 'ssbj_cr_{}_system_loop{:02d}.sql'.format(cr_file_key_word, n_loop)))
        # Get last case
        case = cr_sys.driver_cases.get_case(-1)
        des_vars_sh = case.outputs['z_sh']
        des_vars_c = case.outputs['z_c']
        des_vars_w = case.outputs['z_w']

        # Set initial shared design variables and its bounds
        # Variables:    z = (t/c, h, M, AR, Lambda, Sref)
        z_sh = des_vars_sh
        z_sh_lower = des_vars_list[n_loop]['z_sh']['lower']
        z_sh_upper = des_vars_list[n_loop]['z_sh']['upper']
        # Also set absolute minimum and maximum values of design variables
        z_sh_min = np.array([0.2, 0.666, 0.875, 0.45, 0.72, 0.5])
        z_sh_max = np.array([1.8, 1.333, 1.125, 1.45, 1.27, 1.5])

        # Set coupling values and bounds
        # Add some logical, though conservative, bounds to the coupling variables that have become design variables
        z_c_def = OrderedDict()
        z_c_def['D'] = [1000., 15000., 0., np.inf]  # 0
        z_c_def['WE'] = [0., 20000., 0., np.inf]  # 1
        z_c_def['WT'] = [20000., 60000., 0., np.inf]  # 2
        z_c_def['Theta'] = [0.96, 1.04, 0.96, 1.04]  # 3
        z_c_def['ESF'] = [0.5, 1.5, 0.5, 1.5]  # 4
        z_c_def['L'] = [20000., 60000., 0., np.inf]  # 5
        z_c = des_vars_c
        z_c_lower = des_vars_list[n_loop]['z_c']['lower']
        z_c_upper = des_vars_list[n_loop]['z_c']['upper']
        z_c_min = np.reshape(np.asarray([item[2] / scalers[key] for key, item in z_c_def.items()]), (len(z_c_def)))
        z_c_max = np.reshape(np.asarray([item[3] / scalers[key] for key, item in z_c_def.items()]), (len(z_c_def)))

        # Set weights and bounds
        z_w_def = ['w_D',  # 0
                   'w_WE',  # 1
                   'w_WT',  # 2
                   'w_Theta',  # 3
                   'w_ESF',  # 4
                   'w_L']  # 5
        # Variables
        z_w = des_vars_w
        z_w_lower = des_vars_list[n_loop]['z_w']['lower']
        z_w_upper = des_vars_list[n_loop]['z_w']['upper']
        # Also set absolute minimum and maximum values of design variables
        z_w_min = -np.ones(len(z_w_def)) * np.inf
        z_w_max = np.ones(len(z_w_def)) * np.inf
    else:
        raise AssertionError('Unknown start_type selected ({}).'.format(start_type))

    # All this is stored in a design variables object
    z_ini = dict()
    z_ini['z_sh'] = {'nominal': z_sh, 'lower': z_sh_lower, 'upper': z_sh_upper, 'min': z_sh_min, 'max': z_sh_max}
    z_ini['z_c'] = {'nominal': z_c, 'lower': z_c_lower, 'upper': z_c_upper, 'min': z_c_min, 'max': z_c_max}
    z_ini['z_w'] = {'nominal': z_w, 'lower': z_w_lower, 'upper': z_w_upper, 'min': z_w_min, 'max': z_w_max}

    return z_ini


def get_des_vars_and_qois(discipline):
    """Method to get the relevant design variables and quantities of interest for a specific subsystem.

    :param discipline: name of the discipline (structures, aerodynamics, propulsion)
    :type discipline: basestring
    :return: ordered dictionary with relevant design variables and list of QOIs
    :rtype: tuple
    """
    if discipline == 'structures':
        relevant_des_vars = OrderedDict()
        relevant_des_vars['z_sh'] = [0, 3, 4, 5]  # t/c, AR, Lambda, Sref
        relevant_des_vars['z_c'] = [1, 5]         # WE, L
        relevant_des_vars['z_w'] = [2, 3]         # w_WT, w_Theta
        relevant_qois = ['WT', 'WF', 'Theta']
    elif discipline == 'aerodynamics':
        relevant_des_vars = OrderedDict()
        relevant_des_vars['z_sh'] = [0, 1, 2, 3, 4, 5]  # t/c, h, M, AR, Lambda, Sref
        relevant_des_vars['z_c'] = [2, 3, 4]            # WT, Theta, ESF
        relevant_des_vars['z_w'] = [0, 5]               # w_D, w_L
        relevant_qois = ['L', 'D', 'fin']
    elif discipline == 'propulsion':
        relevant_des_vars = OrderedDict()
        relevant_des_vars['z_sh'] = [1, 2]  # h, M
        relevant_des_vars['z_c'] = [0]      # D
        relevant_des_vars['z_w'] = [1, 4]   # w_WE, w_ESF
        relevant_qois = ['WE', 'ESF', 'SFC']
    else:
        raise IOError('Invalid discipline {} specified.'.format(discipline))
    return relevant_des_vars, relevant_qois


def get_optimized_subsystem(discipline, des_vars, scalers, opt_driver):
    """Method to run the optimizations of a subsystem based on a DOE.

    :param discipline: name of the discipline (structures, aerodynamics, propulsion)
    :type discipline: basestring
    :param des_vars: definition of all design variables
    :type des_vars: dict
    :param scalers: scalers of all the system values
    :type scalers: dict
    :param opt_driver: type of optimization driver
    :type opt_driver: Driver
    :return: tuple with the sample and result values for all optimized subsystems
    :rtype:
    """
    # Get list of relevant design variables and QOIs
    relevant_des_vars, relevant_qois = get_des_vars_and_qois(discipline)

    # Set up Problem() with the relevant design variables
    p = Problem()
    m = p.model

    # Add component with design variables to the model
    des_vars_comp = m.add_subsystem('des_vars', IndepVarComp(), promotes_outputs=['*'])
    for des_var, indices in relevant_des_vars.items():
        for ind in indices:
            des_vars_comp.add_output('{}_{}'.format(des_var, ind), val=des_vars[des_var]['nominal'][ind])

    # Add SubOpt() group to the problem
    p.model.add_subsystem('sub_opt',
                          SubOpt(discipline=discipline, scalers=scalers, driver=opt_driver), promotes_outputs=['*'])

    # Define connections
    if discipline == 'structures':
        m.connect('z_sh_0', 'sub_opt.tc_hat')
        m.connect('z_sh_3', 'sub_opt.AR_hat')
        m.connect('z_sh_4', 'sub_opt.Lambda_hat')
        m.connect('z_sh_5', 'sub_opt.Sref_hat')
        m.connect('z_c_1', 'sub_opt.WE_hat')
        m.connect('z_c_5', 'sub_opt.L_hat')
        m.connect('z_w_2', 'sub_opt.w_WT')
        m.connect('z_w_3', 'sub_opt.w_Theta')
        n_x = 8
    elif discipline == 'aerodynamics':
        m.connect('z_sh_0', 'sub_opt.tc_hat')
        m.connect('z_sh_1', 'sub_opt.h_hat')
        m.connect('z_sh_2', 'sub_opt.M_hat')
        m.connect('z_sh_3', 'sub_opt.AR_hat')
        m.connect('z_sh_4', 'sub_opt.Lambda_hat')
        m.connect('z_sh_5', 'sub_opt.Sref_hat')
        m.connect('z_c_2', 'sub_opt.WT_hat')
        m.connect('z_c_3', 'sub_opt.Theta_hat')
        m.connect('z_c_4', 'sub_opt.ESF_hat')
        m.connect('z_w_0', 'sub_opt.w_D')
        m.connect('z_w_5', 'sub_opt.w_L')
        n_x = 11
    elif discipline == 'propulsion':
        m.connect('z_sh_1', 'sub_opt.h_hat')
        m.connect('z_sh_2', 'sub_opt.M_hat')
        m.connect('z_c_0', 'sub_opt.D_hat')
        m.connect('z_w_1', 'sub_opt.w_WE')
        m.connect('z_w_4', 'sub_opt.w_ESF')
        n_x = 5
    else:
        raise AssertionError('Unknown discipline "{}" provided.')

    # Define design variables and objectives
    sample_keys = []
    qoi_keys = []
    for des_var, indices in relevant_des_vars.items():
        for ind in indices:
            m.add_design_var('{}_{}'.format(des_var, ind),
                             lower=des_vars[des_var]['lower'][ind],
                             upper=des_vars[des_var]['upper'][ind])
            sample_keys.append('{}_{}'.format(des_var, ind))
    for qoi in relevant_qois:
        m.add_objective('{}'.format(qoi))
        qoi_keys.append('{}'.format(qoi))

    # Define DOE driver + sampler
    d = p.driver = DOEDriver(LatinHypercubeGenerator(samples=F_SAMPLES*n_x, criterion='maximin', seed=LHS_SEED))

    # Set-up and run driver
    d.add_recorder(SqliteRecorder(os.path.join(cr_files_folder, 'doe_subsystem_{}.sql'.format(discipline))))
    p.setup()
    p.set_solver_print(level=1)

    # Store (and optionally view) the model
    view_model(p, outfile=os.path.join(cr_files_folder, 'bliss2000_subdoe_n2_{}.html'.format(discipline[0:4])),
               show_browser=False)

    # Run the driver
    # p.run_model()
    p.run_driver()

    # Print results
    cases = CaseReader(os.path.join(cr_files_folder, 'doe_subsystem_{}.sql'.format(discipline))).driver_cases
    print('Number of cases analyzed: {}'.format(cases.num_cases))
    sample_values = []
    result_values = []
    fail_cases = 0.
    for n in range(cases.num_cases):
        outputs = cases.get_case(n).outputs
        if not math.isnan(float(outputs[qoi_keys[0]])):
            sample_values.append([float(outputs[sample_key]) for sample_key in sample_keys])
            result_values.append([float(outputs[result_key]) for result_key in qoi_keys])
        else:
            fail_cases += 1.
    fail_percentage = fail_cases / cases.num_cases * 100.
    print('Fail percentage: {:.1f}% for discipline {} in BLISS loop {}/{}.'.format(fail_percentage, discipline, l,
                                                                                   MAX_LOOPS-1))
    if fail_percentage > 50.:
        warnings.warn('ATTENTION! More than 50% of the DOE samples (actually {:.1f}%) were not optimized for discipline'
                      ' {} in BLISS loop {}/{}.'.format(fail_percentage, discipline, l, MAX_LOOPS-1))
    elif fail_percentage > 20.:
        warnings.warn('More than 20% of the DOE samples (actually {:.1f}%) were not optimized for discipline {} '
                      'in BLISS loop {}/{}.'.format(fail_percentage, discipline, l, MAX_LOOPS-1))

    # Return samples and results
    return sample_values, result_values


def run_system_optimization(des_vars, subsystems, scalers, loop_number):
    """Method to run the top-level system optimization based on the disciplinary surrogate models.

    :param des_vars: definition of design variables
    :type des_vars: dict
    :param subsystems: definition of the disciplinary surrogate models
    :type subsystems: dict
    :param scalers: scalers of all the system values
    :type scalers: dict
    :param loop_number: number of the BLISS iteration
    :type loop_number: int
    :return: tuple with Problem object and driver status
    :rtype: tuple
    """

    # Set up problem and model
    prob = Problem()
    prob.model = model = SsbjBLISS2000(des_vars=des_vars, subsystems=subsystems,
                                       scalers=scalers, loop_number=loop_number)

    # Set driver
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.opt_settings['MAXIT'] = 50
    prob.driver.opt_settings['ACC'] = 1e-6

    # Add design variables
    for des_var, details in des_vars.items():
        prob.model.add_design_var(des_var, lower=details['lower'], upper=details['upper'])

    # Add objective
    model.add_objective('performance.R', scaler=-1.)

    # Add constraints
    model.add_constraint('consistency_constraints.gc_D', equals=0.0)
    model.add_constraint('consistency_constraints.gc_WE', equals=0.0)
    model.add_constraint('consistency_constraints.gc_WT', equals=0.0)
    model.add_constraint('consistency_constraints.gc_L', equals=0.0)
    model.add_constraint('consistency_constraints.gc_Theta', equals=0.0)
    model.add_constraint('consistency_constraints.gc_ESF', equals=0.0)
    model.add_constraint('consistency_constraints.gc_WT_L', equals=0.0)
    model.add_constraint('constraints.con_dpdx', upper=0.0)

    # Add recorder
    recorder = SqliteRecorder(os.path.join(cr_files_folder, 'ssbj_cr_{}_system_loop{:02d}.sql'.format(cr_files_keyword, loop_number)))
    prob.driver.add_recorder(recorder)
    prob.driver.recording_options['includes'] = []
    prob.driver.recording_options['record_objectives'] = True
    prob.driver.recording_options['record_constraints'] = True
    prob.driver.recording_options['record_desvars'] = True
    prob.driver.recording_options['record_metadata'] = True

    # Set up
    prob.setup(mode='rev')

    # View model
    view_model(prob, outfile=os.path.join(cr_files_folder, 'bliss2000_sys_ssbj.html'), show_browser=False)

    # Run problem (either once (run_model) or full optimization (run_driver))
    prob.run_driver()

    # Report result in the log
    print('- - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('\nOutcome of system optimization (BLISS loop: {})'.format(loop_number))
    print('\n\nDesign variables')
    print('z_sh_low= ', des_vars['z_sh']['lower'])
    print('z_sh_val= ', prob['z_sh'])
    print('z_sh_upp= ', des_vars['z_sh']['upper'])
    print('')
    print('z_c_low= ', des_vars['z_c']['lower'])
    print('z_c_val= ', prob['z_c'])
    print('z_c_upp= ', des_vars['z_c']['upper'])
    print('')
    print('z_w_low= ', des_vars['z_w']['lower'])
    print('z_w_val= ', prob['z_w'])
    print('z_w_upp= ', des_vars['z_w']['upper'])
    print('')
    print('\nObjectives')
    print('R_opt=', prob['performance.R'] * scalers['R'])

    print('\nConstraints')
    print('gc_D=', prob['consistency_constraints.gc_D'])
    print('gc_WE=', prob['consistency_constraints.gc_WE'])
    print('gc_WT=', prob['consistency_constraints.gc_WT'])
    print('gc_L=', prob['consistency_constraints.gc_L'])
    print('gc_Theta=', prob['consistency_constraints.gc_Theta'])
    print('gc_ESF=', prob['consistency_constraints.gc_ESF'])
    print('gc_WT_L=', prob['consistency_constraints.gc_WT_L'])
    print('c_dpdx=', prob['constraints.con_dpdx'])
    print('- - - - - - - - - - - - - - - - - - - - - - - - - -')

    return prob, prob.driver.fail


def get_new_bounds(des_vars, loop_number, z_opt, f_k_red, f_int_inc, f_int_inc_abs, f_int_range, optimization_failed):
    """Method that determines new bounds for the design variables for the next BLISS loop. Bounds are initially reduced,
    but will be increased if bounds are hit or if the system-level optimization failed.

    :param des_vars: object containing all design variable details
    :type des_vars: list
    :param loop_number: number of the BLISS iteration loop
    :type loop_number: int
    :param z_opt: optimal design vectors
    :type z_opt: dict
    :param f_k_red: K-factor reduction
    :type f_k_red: float
    :param f_int_inc: percentage of interval increase if bound is hit
    :type f_int_inc: float
    :param f_int_inc_abs: absolute interval increase: minimum increase if percentual increase is too low
    :type f_int_inc_abs: float
    :param f_int_range: minimum width of the design variable interval
    :type f_int_range: float
    :param optimization_failed: indication whether optimization was successful
    :type optimization_failed: bool
    :return: enriched design variables object with new bounds
    :rtype: dict
    """

    # Pick up values
    z = des_vars[loop_number]
    z_new = copy.deepcopy(z)

    # Loop over all design variables and adjust bounds accordingly
    for var_name, des_var in z.items():
        for idx in range(len(z_opt[var_name])):
            val_opt = z_opt[var_name][idx]
            val_nom = des_var['nominal'][idx]
            val_low = des_var['lower'][idx]
            val_upp = des_var['upper'][idx]
            val_min = des_var['min'][idx]
            val_max = des_var['max'][idx]
            val_interval = val_upp - val_low

            if not optimization_failed:
                # If reduction_type is K-factor-based -> reduce accordingly
                adjust = abs((val_upp + val_low) / 2 - val_opt) / ((val_upp + val_low) / 2 - val_low)
                reduce_val = adjust + (1 - adjust) * f_k_red
                val_low_new = val_opt - ((val_interval) / (reduce_val)) / 2
                val_upp_new = val_opt + ((val_interval) / (reduce_val)) / 2

            # If bound has been hit (twice ==> increase)
            if loop_number > 0 or optimization_failed:
                lower_bound_hit = False
                upper_bound_hit = False
                if (val_opt - 1e-2 <= val_low and des_var['nominal'][idx] - 1e-2 <= des_vars[loop_number-1][var_name]
                ['lower'][idx])  or optimization_failed:  # lower bound hit twice or optimization failed
                    lower_bound_hit = True
                    dist_lb = abs(val_opt-val_low)
                if (val_opt + 1e-2 >= val_upp and des_var['nominal'][idx] + 1e-2 >= des_vars[loop_number-1][var_name]
                ['upper'][idx]) or optimization_failed:  # upper bound hit twice or optimization failed
                    upper_bound_hit = True
                    dist_ub = abs(val_opt-val_upp)
                if lower_bound_hit and upper_bound_hit:
                    if dist_lb < dist_ub:
                        change_bound = 'lb'
                    elif dist_ub < dist_lb:
                        change_bound = 'ub'
                    else:
                        change_bound = 'lub'
                elif lower_bound_hit or upper_bound_hit:
                    if upper_bound_hit:
                        change_bound = 'ub'
                    else:
                        change_bound = 'lb'
                else:
                    change_bound = None
                incr = abs(val_interval * f_int_inc / 2)
                if change_bound in ['lb', 'lub']:
                    if incr >= f_int_inc_abs:
                        val_low_new = val_low - val_interval * f_int_inc / 2
                    else:
                        val_low_new = val_low - f_int_inc_abs
                elif change_bound in ['ub', 'lub']:
                    if incr >= f_int_inc_abs:
                        val_upp_new = val_upp + val_interval * f_int_inc / 2
                    else:
                        val_upp_new = val_upp + f_int_inc_abs

            # Check if bounds are not reversed -> otherwise set equal with minimal range
            if val_low_new > val_upp_new:
                val_low_new = val_opt - .5*f_int_range
                val_upp_new = val_opt + .5*f_int_range

            # If interval range is smaller than the minimum range -> adjust accordingly
            if abs(val_upp_new - val_low_new) < f_int_range:
                # First consider upper bound
                dist_ub = abs(val_opt-val_max)
                if dist_ub < .5*f_int_range:
                    val_upp_new = val_max
                    rest_range_ub = .5*f_int_range-dist_ub
                else:
                    val_upp_new = val_opt + .5*f_int_range
                    rest_range_ub = 0.
                # Then adjust lower bound accordingly
                dist_lb = abs(val_opt-val_min)
                if dist_lb < .5*f_int_range:
                    val_low_new = val_min
                    rest_range_lb = .5*f_int_range-dist_lb
                else:
                    val_low_new = val_opt - .5*f_int_range-rest_range_ub
                    rest_range_lb = 0.
                # Add lower bound rest range to the upper bound
                val_upp_new += rest_range_lb

            # If interval is outside maximum bounds -> set equal to appropriate extremum
            if val_low_new < val_min:
                val_low_new = val_min
            if val_upp_new > val_max:
                val_upp_new = val_max

            # Save new bounds and nominal values in z-vector
            if optimization_failed:
                z_new[var_name]['nominal'][idx] = val_nom
            else:
                z_new[var_name]['nominal'][idx] = val_opt
            z_new[var_name]['lower'][idx] = val_low_new
            z_new[var_name]['upper'][idx] = val_upp_new

    return z_new


def pickle_object(obj, filename, dst=None):
    """Method to pickle and object.

    :param obj: object to be pickled
    :type obj: any
    :param filename: name of the pickled file
    :type filename: basestring
    :param dst: name of destination folder (optional)
    :type dst: basestring
    """
    if dst is None:
        filepath = filename
    else:
        filepath = os.path.join(dst, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    pickle.dump(obj, open(filepath, 'wb'))


if __name__ == '__main__':
    # Initialize
    print('Initializing overall system...')

    # Initialize SSBJ problem to get the right scalers
    scalers = init_ssbj_mda()

    # Initialize design vectors and bounds
    z_ini = set_initial_values(START_TYPE,
                               cr_file_folder_name=RESTART_FOLDER, cr_file_key_word=RESTART_KEYWORD,
                               n_loop=RESTART_FROM_LOOP)

    # Check if output folder exists and else create it
    if not os.path.exists(cr_files_folder):
        os.mkdir(cr_files_folder)

    # Define the subsystems
    subsystems = {'structures': dict(samples=[None]*MAX_LOOPS, results=[None]*MAX_LOOPS,
                                     surrogate_model=[None]*MAX_LOOPS),
                  'aerodynamics': dict(samples=[None]*MAX_LOOPS, results=[None]*MAX_LOOPS,
                                       surrogate_model=[None]*MAX_LOOPS),
                  'propulsion': dict(samples=[None]*MAX_LOOPS, results=[None]*MAX_LOOPS,
                                     surrogate_model=[None]*MAX_LOOPS)}
    sys_order = ['structures', 'aerodynamics', 'propulsion']
    sys_problems = [None]*MAX_LOOPS
    des_vars = [None]*MAX_LOOPS
    fail_bools = [None]*MAX_LOOPS
    des_vars[0] = z_ini
    print('Initialization done!')
    print('Starting BLISS loops...')

    # Start BLISS-2000 loop
    for l in range(0, MAX_LOOPS):
        print('')
        print('Started BLISS loop {}/{}'.format(l, MAX_LOOPS-1))
        z = des_vars[l]
        print('\nPerforming DOEs for optimized subsystems...')
        for discipline in sys_order:
            subsys_dis = subsystems[discipline]
            # Perform DOE for optimized subsystems
            print('\nPerform subsystem optimizations for {} discipline.'.format(discipline))
            subsys_dis['samples'][l], subsys_dis['results'][l] = get_optimized_subsystem(discipline, z, scalers,
                                                                                         ScipyOptimizeDriver())

            # Create surrogate model
            subsys_dis['surrogate_model'][l] = sm = MetaModelUnStructuredComp(default_surrogate=ResponseSurface())
            sm.add_input('x', val=np.zeros(len(subsys_dis['samples'][l][0])), training_data=subsys_dis['samples'][l])
            sm.add_output('y', val=np.zeros(len(subsys_dis['results'][l][0])), training_data=subsys_dis['results'][l])

        # Perform system optimization using surrogate models
        print('\nPerforming system optimization using surrogate models..')
        sys_problems[l], fail_bools[l] = run_system_optimization(z, subsystems, scalers, l)

        # Check optimization results and prepare next loop
        if fail_bools[l]:
            warnings.warn('System-level optimization failed at loop {}/{}.'.format(l, MAX_LOOPS-1))
        elif l > 0:
            if not fail_bools[l-1] and not fail_bools[l]:  # Check if both optimization were successful
                conv_abs = abs(sys_problems[l-1]['performance.R'] - sys_problems[l]['performance.R'])
                conv_rel = abs((sys_problems[l-1]['performance.R'] - sys_problems[l]['performance.R']) /
                               sys_problems[l-1]['performance.R'])
                if conv_abs < CONV_ABS_TOL or conv_rel < CONV_REL_TOL:
                    print('BLISS loop {}/{} converged (conv_abs = {} and conv_rel = {}).'.format(l, MAX_LOOPS-1,
                                                                                                 conv_abs, conv_rel))
                    sys_problems = sys_problems[:l + 1]
                    des_vars = des_vars[:l + 1]
                    fail_bools = fail_bools[:l + 1]

                    # Save the data
                    pickle_object(des_vars, 'ssbj_des_vars_{}_system_loops.p'.format(cr_files_keyword),
                                  dst=cr_files_folder)
                    pickle_object(fail_bools, 'ssbj_fail_bools_{}_system_loops.p'.format(cr_files_keyword),
                                  dst=cr_files_folder)
                    break
                else:
                    print('BLISS loop {}/{} did not converge (conv_abs = {} and conv_rel = {}).'
                          .format(l, MAX_LOOPS-1, conv_abs, conv_rel))

        # Update variables and bounds for next loop
        if l < MAX_LOOPS-1:
            z_opt = {'z_sh': sys_problems[l]['z_sh'],
                     'z_c': sys_problems[l]['z_c'],
                     'z_w': sys_problems[l]['z_w']}
            des_vars[l + 1] = get_new_bounds(des_vars, l, z_opt, F_K_RED, F_INT_INC, F_INT_INC_ABS, F_INT_RANGE,
                                             fail_bools[l])

        # Save the data as pickled objects (overwrite every time in case of intermediate failure)
        pickle_object(des_vars, 'ssbj_des_vars_{}_system_loops.p'.format(cr_files_keyword), dst=cr_files_folder)
        pickle_object(fail_bools, 'ssbj_fail_bools_{}_system_loops.p'.format(cr_files_keyword), dst=cr_files_folder)

        if l == MAX_LOOPS-1:
            print('BLISS loops never converged. Terminated optimization because maximum BLISS iterations of {} '
                  'has been reached.'.format(MAX_LOOPS))

    R = float(sys_problems[l]['performance.R']*scalers[R])
    assert(R > 3960.)
    assert(R < 3970.)
    print('Reached end of script.')
