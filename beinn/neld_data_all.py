import os,sys 




nams=[]


path_heads_show=[]
paraws={}
params={mm:[] for mm in ['dnn_modes','nam_gen','path_heads_show','nam']}

paraw={mm:[] for mm in ['dict_param','gdas']}

'''
from neld_data_1 import gdas,param_flow,nam_gen,dict_param,data_dir,Nmarkov,N,Nperiod,nma,g_names,dnn_modes,nam


action='train'
nam=f'{nam_gen}_{action}'
namt=f'{nam_gen}_test'
dict_param['path_heads_show']=[f'dnn_PINN_NELD___{nam}',f'dnn_PINN_NELD___{namt}'] 

path_dict={'run':'neld_pinn_0.run_00'}
params['dnn_modes'].extend(dnn_modes)
params['path_heads_show'].extend(dict_param['path_heads_show'])
params['nam'].append(nam)

paraw={mm:[] for mm in ['dict_param','gdas','path_dict','nam_gen']}
paraw['dict_param']=dict_param
paraw['gdas']=gdas
paraw['path_dict']=path_dict
paraw['nam_gen']=nam_gen

paraws[nam_gen]=paraw



from neld_data_1_shear import gdas,param_flow,nam_gen,dict_param,data_dir,Nmarkov,N,Nperiod,nma,g_names,dnn_modes,nam
 


action='train'
nam=f'{nam_gen}_{action}'
namt=f'{nam_gen}_test'
dict_param['path_heads_show']=[f'dnn_PINN_NELD___{nam}',f'dnn_PINN_NELD___{namt}'] 

path_dict={'run':'neld_pinn_0.run_00'}
params['dnn_modes'].extend(dnn_modes)
params['path_heads_show'].extend(dict_param['path_heads_show'])
params['nam'].append(nam)


paraw={mm:[] for mm in ['dict_param','gdas','path_dict','nam_gen']}
paraw['dict_param']=dict_param
paraw['gdas']=gdas
paraw['path_dict']=path_dict
paraw['nam_gen']=nam_gen
paraws[nam_gen]=paraw

'''



from neld_data_harmonic_2_shear import gdas,param_flow,nam_gen,dict_param,data_dir,Nmarkov,N,Nperiod,nma,g_names,dnn_modes,nam
  

path_dict={'run':'neld_pinn_0.run_00'}
action='train'
nam=f'{nam_gen}_{action}'
namt=f'{nam_gen}_test'
dict_param['path_heads_show']=[f'dnn_PINN_NELD___{nam}',f'dnn_PINN_NELD___{namt}'] 

params['dnn_modes'].extend(dnn_modes)
params['path_heads_show'].extend(dict_param['path_heads_show'])
params['nam'].append(nam)

paraw={mm:[] for mm in ['dict_param','gdas','path_dict','nam_gen']}
paraw['dict_param']=dict_param
paraw['gdas']=gdas
paraw['path_dict']=path_dict
paraw['nam_gen']=nam_gen

paraws[nam_gen]=paraw


 











