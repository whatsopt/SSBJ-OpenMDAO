"""
SSBJ test case - http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Python implementation and OpenMDAO integration developed by
Sylvain Dubreuil and Remi Lafage of ONERA, the French Aerospace Lab.
BLISS-2000 strategy optimization and postprocessing scripts
developed by Imco van Gent of TU Delft, Faculty of Aerospace Engineering.
"""
import pickle
import os

from openmdao.recorders.case_reader import CaseReader

import plotly
import plotly.graph_objs as go

cr_file_folder_name = 'files'
cr_file_key_word = 'bliss_run'

# Open des_vars
des_vars_list = pickle.load(open(os.path.join(cr_file_folder_name,
                                              'ssbj_des_vars_{}_system_loops.p'.format(cr_file_key_word)), 'rb'))

# Get the number of BLISS loops
n_loops = len(des_vars_list) - 1 if cr_file_key_word == 'bliss_removews3' else len(des_vars_list)

# Plot top-level optimization results
objectives = []
constraints = []
des_vars_sh = []
des_vars_sh_low = []
des_vars_sh_upp = []
des_vars_c = []
des_vars_c_low = []
des_vars_c_upp = []
des_vars_w = []
des_vars_w_low = []
des_vars_w_upp = []
iters = []

for n_loop in range(n_loops):

    cr_sys = CaseReader(os.path.join(cr_file_folder_name,
                                     'ssbj_cr_{}_system_loop{:02d}.sql'.format(cr_file_key_word, n_loop)))

    print('Number of driver cases recorded = {}'.format(cr_sys.driver_cases.num_cases))
    # case_keys = cr_sys.driver_cases.list_cases()
    # Get last case
    case = cr_sys.driver_cases.get_case(-1)
    des_vars_sh.append(list(case.outputs['z_sh']))
    des_vars_sh_low.append(list(des_vars_list[n_loop]['z_sh']['lower']))
    des_vars_sh_upp.append(list(des_vars_list[n_loop]['z_sh']['upper']))
    des_vars_c.append(list(case.outputs['z_c']))
    des_vars_c_low.append(list(des_vars_list[n_loop]['z_c']['lower']))
    des_vars_c_upp.append(list(des_vars_list[n_loop]['z_c']['upper']))
    des_vars_w.append(list(case.outputs['z_w']))
    des_vars_w_low.append(list(des_vars_list[n_loop]['z_w']['lower']))
    des_vars_w_upp.append(list(des_vars_list[n_loop]['z_w']['upper']))
    iters.append(n_loop)
    objectives.append(float(case.outputs['performance.R']))
    constraints.append([float(case.outputs['consistency_constraints.gc_D']),
                        float(case.outputs['consistency_constraints.gc_WE']),
                        float(case.outputs['consistency_constraints.gc_WT']),
                        float(case.outputs['consistency_constraints.gc_L']),
                        float(case.outputs['consistency_constraints.gc_Theta']),
                        float(case.outputs['consistency_constraints.gc_ESF'])])

# Plot objective
trace_obj = [go.Scatter(x=iters, y=objectives, mode='markers', name='objective R')]
layout_obj = go.Layout(title='Objective of top-level system optimization', showlegend=True,
                       xaxis=dict(title='iteration'), yaxis=dict(title='scaled value [-]'))
fig_obj = go.Figure(data=trace_obj, layout=layout_obj)
plotly.offline.plot(fig_obj, filename=os.path.join('files', 'ssbj_bliss2000_top_objective.html'))

# Plot constraint
traces_con = []
legend_entries = ['gc_D', 'gc_WE', 'gc_WT', 'gc_L', 'gc_Theta', 'gc_ESF', 'gc_WT_L']
for i in range(0, len(constraints[0])):
    trace = go.Scatter(x=iters, y=[val[i] for val in constraints], mode='markers', name=legend_entries[i])
    traces_con.append(trace)
layout_constraints = go.Layout(title='Constraints of top-level system optimization', showlegend=True,
                               xaxis=dict(title='iteration'), yaxis=dict(title='scaled value [-]'))
fig_des_z = go.Figure(data=traces_con, layout=layout_constraints)
plotly.offline.plot(fig_des_z, filename=os.path.join('files', 'ssbj_bliss2000_top_constraints.html'))

# Plot design variables
traces_des_z = []
legend_entries = ['Thickness-to-chord ratio (t/c)', 'Cruise height (h)', 'Mach number (M)', 'Aspect ratio (AR)',
                  'Sweep (Lambda)', 'Wing area (Sref)']
for i in range(0, len(des_vars_sh[0])):
    val_y = [val[i] for val in des_vars_sh]
    val_y_low = [val[i] for val in des_vars_sh_low]
    val_y_upp = [val[i] for val in des_vars_sh_upp]
    error_y = [abs(val_y_upp[j]-val_y[j]) for j in range(len(val_y))]
    error_y_minus = [abs(val_y_low[j] - val_y[j]) for j in range(len(val_y))]
    trace = go.Scatter(x=iters, y=val_y, mode='markers', name=legend_entries[i],
                       error_y=dict(type='data', symmetric=False, array=error_y, arrayminus=error_y_minus),
                       marker=dict(size=12))
    traces_des_z.append(trace)
layout_des_z = go.Layout(title='Design variables of top-level system optimization', showlegend=True,
                         xaxis=dict(title='iteration'), yaxis=dict(title='scaled value [-]'))
fig_des_z = go.Figure(data=traces_des_z, layout=layout_des_z)
plotly.offline.plot(fig_des_z, filename=os.path.join('files', 'ssbj_bliss2000_top_des_vars_z_sh.html'))

traces_des_z = []
legend_entries = ['D', 'WE', 'WT', 'Theta', 'ESF', 'L']
for i in range(0, len(des_vars_c[0])):
    val_y = [val[i] for val in des_vars_c]
    val_y_low = [val[i] for val in des_vars_c_low]
    val_y_upp = [val[i] for val in des_vars_c_upp]
    error_y = [abs(val_y_upp[j] - val_y[j]) for j in range(len(val_y))]
    error_y_minus = [abs(val_y_low[j] - val_y[j]) for j in range(len(val_y))]
    trace = go.Scatter(x=iters, y=[val[i] for val in des_vars_c], mode='markers', name=legend_entries[i],
                       error_y=dict(type='data', symmetric=False, array=error_y, arrayminus=error_y_minus),
                       marker=dict(size=12))
    traces_des_z.append(trace)
layout_des_z = go.Layout(title='Design variables of top-level system optimization', showlegend=True,
                         xaxis=dict(title='iteration'), yaxis=dict(title='scaled value [-]'))
fig_des_z = go.Figure(data=traces_des_z, layout=layout_des_z)
plotly.offline.plot(fig_des_z, filename=os.path.join('files', 'ssbj_bliss2000_top_des_vars_z_c.html'))

traces_des_z = []
legend_entries = ['w_D', 'w_WE', 'w_WT', 'w_Theta', 'w_ESF', 'w_L']
for i in range(0, len(des_vars_w[0])):
    val_y = [val[i] for val in des_vars_w]
    val_y_low = [val[i] for val in des_vars_w_low]
    val_y_upp = [val[i] for val in des_vars_w_upp]
    error_y = [abs(val_y_upp[j] - val_y[j]) for j in range(len(val_y))]
    error_y_minus = [abs(val_y_low[j] - val_y[j]) for j in range(len(val_y))]
    trace = go.Scatter(x=iters, y=[val[i] for val in des_vars_w], mode='markers', name=legend_entries[i],
                       error_y=dict(type='data', symmetric=False, array=error_y, arrayminus=error_y_minus),
                       marker=dict(size=12))
    traces_des_z.append(trace)
layout_des_z = go.Layout(title='Design variables of top-level system optimization', showlegend=True,
                         xaxis=dict(title='iteration'), yaxis=dict(title='scaled value [-]'))
fig_des_z = go.Figure(data=traces_des_z, layout=layout_des_z)
plotly.offline.plot(fig_des_z, filename=os.path.join('files', 'ssbj_bliss2000_top_des_vars_z_w.html'))
