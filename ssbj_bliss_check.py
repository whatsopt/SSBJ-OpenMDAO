"""
SSBJ test case - http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Python implementation and OpenMDAO integration developed by
Sylvain Dubreuil and Remi Lafage of ONERA, the French Aerospace Lab.
BLISS-2000 strategy optimization and postprocessing scripts
developed by Imco van Gent of TU Delft, Faculty of Aerospace Engineering.
"""
from __future__ import print_function
import pickle
import numpy as np
import os

from openmdao.api import Problem
from openmdao.api import CaseReader

from ssbj_mda import SSBJ_MDA, init_ssbj_mda

if __name__ == "__main__":
    scalers = init_ssbj_mda()

    # Pick up values from recorder
    cr_file_folder_name = 'files'
    cr_file_key_word = 'bliss_newrun4'
    n_loop = 13
    des_vars_list = pickle.load(open(os.path.join(cr_file_folder_name,
                                                  'ssbj_des_vars_{}_system_loops.p'.format(cr_file_key_word)), 'rb'))
    cr_sys = CaseReader(
        os.path.join(cr_file_folder_name, 'ssbj_cr_{}_system_loop{:02d}.sql'.format(cr_file_key_word, n_loop)))
    n_loops = len(des_vars_list)
    # Get last case
    case = cr_sys.driver_cases.get_case(-1)
    des_vars_sh = case.outputs['z_sh']

    prob = Problem()
    prob.model = SSBJ_MDA(scalers)
    prob.setup()
    prob['z'] = des_vars_sh
    prob['z'][0] = 1.2
    prob['x_str'] = np.array([1.6, 0.75])
    prob['x_aer'] = np.array([0.75])
    prob['x_pro'] = np.array([0.3125])
    prob.run_model()

    print('Z_opt=', prob['z'])
    print('X_str_opt=', prob['x_str'])
    print('X_aer_opt=', prob['x_aer'])
    print('X_pro_opt=', prob['x_pro'])
    print('R_opt=', prob['R'])
    print('con_sigmas:')
    print('   con_sigma1 (< 0.0)=', prob['con_sigma1'])
    print('   con_sigma2 (< 0.0)=', prob['con_sigma2'])
    print('   con_sigma3 (< 0.0)=', prob['con_sigma3'])
    print('   con_sigma4 (< 0.0)=', prob['con_sigma4'])
    print('   con_sigma5 (< 0.0)=', prob['con_sigma5'])
    print('con_theta_up (< 0.0)=', prob['con_theta_up'])
    print('con_theta_low (< 0.0)=', prob['con_theta_low'])
    print('con_dpdx (<0.0)=', prob['con_dpdx'])
    print('con1_esf (<0.0)', prob['con1_esf'])
    print('con2_esf (<0.0)=', prob['con2_esf'])
    print('con_temp (<0.0)=', prob['con_temp'])
    print('con_dt (<0.0)=', prob['con_dt'])
    print('Couplings')
    print('L=', prob['L'])
    print('D=', prob['D'])
    print('WE=', prob['WE'])
    print('WT=', prob['WT'])
    print('Theta=', prob['Theta'])
    print('WF=', prob['WF'])
    print('ESF=', prob['ESF'])
    print('fin=', prob['fin'])
    print('SFC=', prob['SFC'])

