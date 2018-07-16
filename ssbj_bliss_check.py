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
    cr_file_key_word = 'bliss_run'
    n_loop = 23
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
    print('   con_sigma1:', prob['con_sigma1'])
    print('   con_sigma2:', prob['con_sigma2'])
    print('   con_sigma3:', prob['con_sigma3'])
    print('   con_sigma4:', prob['con_sigma4'])
    print('   con_sigma5:', prob['con_sigma5'])
    print('con_dpdx=', prob['con_dpdx'])
    print('con1_esf=', prob['con1_esf'])
    print('con2_esf=', prob['con2_esf'])
    print('con_temp=', prob['con_temp'])
    print('con_dt=', prob['con_dt'])
