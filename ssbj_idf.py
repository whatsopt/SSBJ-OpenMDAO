"""
SSBJ test case - http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Python implementation and OpenMDAO integration developed by
Sylvain Dubreuil and Remi Lafage of ONERA, the French Aerospace Lab.
"""
from sys import argv
import re
import numpy as np
import matplotlib.pylab as plt

from openmdao.api import Problem
from openmdao.api import SqliteRecorder, WebRecorder
from openmdao.api import ScipyOptimizer, pyOptSparseDriver
from ssbj_idf_mda import SSBJ_IDF_MDA
from ssbj_mda import init_ssbj_mda
# pylint: disable=C0103

# Optimization problem
scalers = init_ssbj_mda()

prob = Problem()
prob.model = SSBJ_IDF_MDA(scalers)

# Optimizer options
# prob.driver = ScipyOptimizer()
prob.driver = pyOptSparseDriver()
optimizer = 'SLSQP'
prob.driver.options['optimizer'] = optimizer
#prob.driver.options['tol'] = 1e-3

#Design variables
prob.model.add_design_var('z', lower=np.array([0.2, 0.666,0.875,0.45,0.72,0.5]),
                            upper=np.array([1.8,1.333,1.125,1.45,1.27,1.5]))
prob.model.add_design_var('x_str', lower=np.array([0.4,0.75]), upper=np.array([1.6,1.25]))
prob.model.add_design_var('x_aer', lower=0.75, upper=1.25)
prob.model.add_design_var('x_pro', lower=0.18, upper=1.81)

prob.model.add_design_var('Theta')
prob.model.add_design_var('L')
prob.model.add_design_var('WE')
prob.model.add_design_var('WT')
prob.model.add_design_var('ESF')
prob.model.add_design_var('D')

# Objective function
prob.model.add_objective('obj')

#Constraints
prob.model.add_constraint('con_dt', upper=0.0)
prob.model.add_constraint('con_Theta_up', upper=0.0)
prob.model.add_constraint('con_Theta_low', upper=0.0)
prob.model.add_constraint('con_sigma1', upper=0.0)
prob.model.add_constraint('con_sigma2', upper=0.0)
prob.model.add_constraint('con_sigma3', upper=0.0)
prob.model.add_constraint('con_sigma4', upper=0.0)
prob.model.add_constraint('con_sigma5', upper=0.0)
prob.model.add_constraint('con_dpdx', upper=0.0)
prob.model.add_constraint('con_esf', upper=0.0)
prob.model.add_constraint('con_temp', upper=0.0)

#Coupling constraints
#Threshold for the coupling (constraints define as (x_in-x_out)**2<epsilon)
epsilon=1e-10
prob.model.add_constraint('con_str_aer_wt',upper=epsilon)
prob.model.add_constraint('con_str_aer_theta',upper=epsilon)
prob.model.add_constraint('con_aer_str_l',upper=epsilon)
prob.model.add_constraint('con_aer_pro_d',upper=epsilon)
prob.model.add_constraint('con_pro_aer_esf',upper=epsilon)
prob.model.add_constraint('con_pro_str_we',upper=epsilon)
#Recorder
db_name = 'ssbj_idf.sqlite'
if "--plot" in argv:
    recorder = SqliteRecorder(db_name)
    recorder2 = WebRecorder("2CJSXJGV44", case_name="SSBJ IDF")
    recorder.options['record_desvars'] = True
    recorder.options['record_objectives'] = True
    recorder.options['record_constraints'] = True
    recorder2.options['record_desvars'] = True
    recorder2.options['record_objectives'] = True
    recorder2.options['record_constraints'] = True
    prob.driver.add_recorder(recorder)
#    prob.driver.add_recorder(recorder2)

#Run optimization
prob.setup()
prob.run_driver()
#prob.cleanup()

print 'Z_opt=', prob['z']*scalers['z']
print 'X_str_opt=', prob['x_str']*scalers['x_str']
print 'X_aer_opt=', prob['x_aer']
print 'X_pro_opt=', prob['x_pro']*scalers['x_pro']
print 'R_opt=', -prob['obj']*scalers['R']

if "--plot" in argv:
    from openmdao.recorders.case_reader import CaseReader
    plt.figure()

    cr = CaseReader(db_name)
    print('Number of driver cases recorded =', cr.driver_cases.num_cases )
    case_keys = cr.driver_cases.list_cases()
    r = []
    for case_key in case_keys:    
        r.append(cr.driver_cases.get_case(case_key).objectives['Obj.obj']*scalers['R'])

    plt.plot(r)
    plt.show()

# Check R =~ 3964Nm
R = float(-prob['obj']*scalers['R'])
assert(R > 3963.)
assert(R < 3965.)
# from openmdao.devtools.problem_viewer.problem_viewer import view_model
# view_model(prob)


