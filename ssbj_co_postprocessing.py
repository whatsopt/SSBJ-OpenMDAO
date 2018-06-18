"""
SSBJ test case - http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Python implementation and OpenMDAO integration developed by
Sylvain Dubreuil and Remi Lafage of ONERA, the French Aerospace Lab.
Collaborative Optimization (CO) strategy optimization and postprocessing scripts
developed by Imco van Gent of TU Delft, Faculty of Aerospace Engineering
"""
import os

from openmdao.recorders.case_reader import CaseReader

import plotly
import plotly.graph_objs as go

cr_file_folder_name = 'files'
cr_file_key_word = 'results'

# Plot top-level optimization results
cr_sys = CaseReader(os.path.join(cr_file_folder_name, 'ssbj_cr_{}_co_system.sql'.format(cr_file_key_word)))

print('Number of driver cases recorded = {}'.format(cr_sys.driver_cases.num_cases))
case_keys = cr_sys.driver_cases.list_cases()
objectives = []
constraints = []
des_vars_z = []
des_vars_c = []
iters = []
for i, case_key in enumerate(case_keys):
    case = cr_sys.driver_cases.get_case(case_key)
    des_vars_c_i = [float(case.outputs['D_hat']), float(case.outputs['WE_hat']), float(case.outputs['WT_hat']),
                    float(case.outputs['Theta_hat']), float(case.outputs['WF_hat']), float(case.outputs['ESF_hat']),
                    float(case.outputs['fin_hat']), float(case.outputs['SFC_hat'])]
    iters.append(i)
    objectives.append(float(case.outputs['performance.R']))
    constraints.append(float(case.outputs['J.J']))
    des_vars_z.append(list(case.outputs['z']))
    des_vars_c.append(des_vars_c_i)

# Plot objective
trace_obj = [go.Scatter(x=iters, y=objectives, mode='markers', name='objective R')]
layout_obj = go.Layout(title='Objective of top-level system optimization', showlegend=True,
                       xaxis=dict(title='iteration'), yaxis=dict(title='scaled value [-]'))
fig_obj = go.Figure(data=trace_obj, layout=layout_obj)
plotly.offline.plot(fig_obj, filename=os.path.join('files', 'ssbj_co_top_objective.html'))

# Plot constraint
trace_con = [go.Scatter(x=iters, y=constraints, mode='markers', name='constraint J')]
layout_con = go.Layout(title='Constraint(s) of top-level system optimization', showlegend=True,
                       xaxis=dict(title='iteration'), yaxis=dict(title='scaled value [-]'))
fig_con = go.Figure(data=trace_con, layout=layout_con)
plotly.offline.plot(fig_con, filename=os.path.join('files', 'ssbj_co_top_constraints.html'))

# Plot design variables
traces_des_z = []
legend_entries = ['Thickness-to-chord ratio (t/c)', 'Cruise height (h)', 'Mach number (M)', 'Aspect ratio (AR)',
                  'Sweep (Lambda)', 'Wing area (Sref)']
for i in range(0, len(des_vars_z[0])):
    trace = go.Scatter(x=iters, y=[val[i] for val in des_vars_z], mode='markers', name=legend_entries[i])
    traces_des_z.append(trace)
layout_des_z = go.Layout(title='Design variables of top-level system optimization', showlegend=True,
                       xaxis=dict(title='iteration'), yaxis=dict(title='scaled value [-]'))
fig_des_z = go.Figure(data=traces_des_z, layout=layout_des_z)
plotly.offline.plot(fig_des_z, filename=os.path.join('files', 'ssbj_co_top_des_vars_z.html'))

# Plot design variables
traces_des_c = []
legend_entries = ['D', 'WE', 'WT', 'Theta', 'WF', 'ESF', 'fin', 'SFC']
for i in range(0, len(des_vars_c[0])):
    trace = go.Scatter(x=iters, y=[val[i] for val in des_vars_c], mode='markers', name=legend_entries[i])
    traces_des_c.append(trace)
layout_des_c = go.Layout(title='Design variables (coupling targets) of top-level system optimization', showlegend=True,
                       xaxis=dict(title='iteration'), yaxis=dict(title='scaled value [-]'))
fig_des_c = go.Figure(data=traces_des_c, layout=layout_des_c)
plotly.offline.plot(fig_des_c, filename=os.path.join('files', 'ssbj_co_top_des_vars_c.html'))


# Plot sub-level optimization results (this might take a while)
cr_struc = CaseReader(os.path.join(cr_file_folder_name, 'ssbj_cr_{}_subsystems.sql'.format(cr_file_key_word)))

case_keys = cr_struc.driver_cases.list_cases()
objectives_str = []
constraints_str = []
des_vars_str = []
objectives_aer = []
constraints_aer = []
des_vars_aer = []
objectives_prop = []
constraints_prop = []
des_vars_prop = []
for i, case_key in enumerate(case_keys):
    #print('Case:', case_key)
    case = cr_struc.driver_cases.get_case(case_key)
    if 'subopt_struc' in case_key:
        des_vars_str.append([float(case.outputs['x_str'][0]),
                        float(case.outputs['x_str'][1])])
        objectives_str.append(float(case.outputs['J.J']))
        constraints_str.append([float(case.outputs['con_sigma1']),
                            float(case.outputs['con_sigma2']),
                            float(case.outputs['con_sigma3']),
                            float(case.outputs['con_sigma4']),
                            float(case.outputs['con_sigma5']),
                            float(case.outputs['con_theta'])])
    elif 'subopt_aero' in case_key:
        des_vars_aer.append([float(case.outputs['x_aer'][0])])
        objectives_aer.append(float(case.outputs['J.J']))
        constraints_aer.append([float(case.outputs['constraints.con_dpdx'])])
    elif 'subopt_prop' in case_key:
        des_vars_prop.append([float(case.outputs['x_pro'][0])])
        objectives_prop.append(float(case.outputs['J.J']))
        constraints_prop.append([float(case.outputs['constraints.con_esf']),
                                 float(case.outputs['constraints.con_temp']),
                                 float(case.outputs['constraints.con_dt'])])
iters_str = range(0, len(des_vars_str))
iters_aer = range(0, len(des_vars_aer))
iters_prop = range(0, len(des_vars_prop))

# Plot objective
trace_obj = go.Scatter(x=iters_str,
                       y=objectives_str,
                       mode='markers',
                       name='objective J_str')
data = [trace_obj]
plotly.offline.plot(data, filename=os.path.join('files', 'ssbj_co_sub_str_objectives.html'))

# Plot constraint
data = []
legend_entries = ['con_sigma1', 'con_sigma1', 'con_sigma1', 'con_sigma1', 'con_sigma1', 'con_theta']
for i in range(0, len(constraints_str[0])):
    trace = go.Scatter(x=iters_str,
                       y=[val[i] for val in constraints_str],
                       mode='markers',
                       name='{}'.format(legend_entries[i]))
    data.append(trace)
plotly.offline.plot(data, filename=os.path.join('files', 'ssbj_co_sub_str_constraints.html'))

# Plot design variables
data = []
legend_entries = ['x_str[0] (taper ratio / lambda)', 'x_str[1] (section caisson)']
for i in range(0, len(des_vars_str[0])):
    trace = go.Scatter(x=iters_str,
                       y=[val[i] for val in des_vars_str],
                       mode='markers',
                       name='{}'.format(legend_entries[i]))
    data.append(trace)
plotly.offline.plot(data, filename=os.path.join('files', 'ssbj_co_sub_str_des_vars.html'))



# Plot objective
trace_obj = go.Scatter(x=iters_aer,
                       y=objectives_aer,
                       mode='markers',
                       name='objective J_aer')
data = [trace_obj]
plotly.offline.plot(data, filename=os.path.join('files', 'ssbj_co_sub_aer_objectives.html'))

# Plot constraint
data = []
legend_entries = ['con_dpdx']
for i in range(0, len(constraints_aer[0])):
    trace = go.Scatter(x=iters_aer,
                       y=[val[i] for val in constraints_aer],
                       mode='markers',
                       name='{}'.format(legend_entries[i]))
    data.append(trace)
plotly.offline.plot(data, filename=os.path.join('files', 'ssbj_co_sub_aer_constraints.html'))

# Plot design variables
data = []
legend_entries = ['x_aer (Cf)']
for i in range(0, len(des_vars_aer[0])):
    trace = go.Scatter(x=iters_aer,
                       y=[val[i] for val in des_vars_aer],
                       mode='markers',
                       name='{}'.format(legend_entries[i]))
    data.append(trace)
plotly.offline.plot(data, filename=os.path.join('files', 'ssbj_co_sub_aer_des_vars.html'))

# Plot objective
trace_obj = go.Scatter(x=iters_prop,
                       y=objectives_prop,
                       mode='markers',
                       name='objective J_prop')
data = [trace_obj]
plotly.offline.plot(data, filename=os.path.join('files', 'ssbj_co_sub_prop_objectives.html'))

# Plot constraint
data = []
legend_entries = ['con_esf', 'con_temp', 'con_dt']
for i in range(0, len(constraints_prop[0])):
    trace = go.Scatter(x=iters_prop,
                       y=[val[i] for val in constraints_prop],
                       mode='markers',
                       name='{}'.format(legend_entries[i]))
    data.append(trace)
plotly.offline.plot(data, filename=os.path.join('files', 'ssbj_co_sub_prop_constraints.html'))

# Plot design variables
data = []
legend_entries = ['x_pro (T, throttle)']
for i in range(0, len(des_vars_prop[0])):
    trace = go.Scatter(x=iters_prop,
                       y=[val[i] for val in des_vars_prop],
                       mode='markers',
                       name='{}'.format(legend_entries[i]))
    data.append(trace)
plotly.offline.plot(data, filename=os.path.join('files', 'ssbj_co_sub_prop_des_vars.html'))
