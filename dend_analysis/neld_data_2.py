import os,sys
# path = os.path.expanduser("~/Desktop")
# os.makedirs(path, exist_ok=True)
# base_dir=os.path.join(path,'CURVATURE-BASED-MACHINE-LEARNING-FOR-AUTOMATED-SEGMENTATION-OF-DENDRITIC-smodS')

# file_path_org=os.getcwd()  #os.path.join(base_dir,'dend_analysis')# 
# file_path_org=os.path.join(os.path.dirname(file_path_org) ,'files')
# base_dir=os.path.dirname(file_path_org)    # Change this directory to the path dataset 
 





force='zero'
force='harmonic'
force='fLJ'
flow='pef' 
flow = 'eld' 
flow = 'shear'   
nPart = 5                         # Number of particles
epsilon = 2.0                     # rate of the deformation of the background flow
rcut = 30                         # radius cut
beta,gamma=1,1
N = 50                             # number of steps in a period
Nperiod =  10000                   # number of periods
Nperiod =  5                  # number of periods
N = 200                             # number of steps in a period

Nmarkov=500
N_grid = 10
param_flow=dict(
        force=force,
        flow=flow ,  
        nPart = nPart,                       # Number of particles
        epsilon = epsilon ,                    # rate of the deformation of the background flow
        rcut = rcut,                         # radius cut
        beta=beta,
        gamma=gamma,  
        Nperiod =  Nperiod,                  # number of periods
        N = N,                             # number of steps in a 
        # rhs_dim=dict(zip(['i','j'],[rhs_dim_i,rhs_dim_j])),
)



path_0=os.getcwd() 


 
# file_path_org=os.getcwd()  
# file_path_org=os.path.join(os.path.dirname(file_path_org) ,'files')
# file_path_data=os.path.dirname(file_path_org)  
# gdas=get_data(file_path_data) 



 # 

import numpy as np
import sys,os
DTYPE='float32'  
from dend_fun_0.main_0 import  app_run_param,algorithm,algorithm_param,get_data, get_data_all ,get_dict_param
from neld_pinn_0.run_0 import get_configs  
# from neld_pinn_0 import run

file_path_org=os.getcwd()  #os.path.join(base_dir,'dend_analysis')# 
file_path_data=os.path.join(os.path.dirname(file_path_org) ,'files','data_initial')
os.makedirs(file_path_data,exist_ok=True)
file_path_org=os.path.join(os.path.dirname(file_path_org) ,'files')
# file_path_data=os.path.dirname(file_path_org)    # Change this directory to the path dataset  
gdas=get_data(file_path_data)    # Dataset Path
dend_names_chld={} 
action= 'resize'  



# nam_gen=f'neld_{flow}_{nPart}_{force}'

nam_gen=f'{flow}_{force}_{nPart}_{N}_{Nperiod}_{Nmarkov}'
action='train'
nam=f'{nam_gen}_{action}'
n_obj=51
obj_list=list(np.arange(50,80))
obj_list=list(np.arange(n_obj))#+
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names]  
 
obj_org_path= os.path.join(file_path_data,nam_gen, action )
dend_path_inits=[nam for _ in range(len(dend_names))]
dend_names_chld[nam]= dict(  
                # dend_namess=dend_namess ,
                dend_names=dend_names , 
                obj_org_path=obj_org_path,
                dend_last=dend_last ,
                dend_first=dend_first ,
                dend_path_inits=dend_path_inits, 
                param_flow=param_flow,
                )  

 
# nam_gen=f'neld_{flow}_{nPart}_{force}'
action='test'
nam=f'{nam_gen}_{action}'

obj_list=np.arange(5)
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names]  

obj_org_path= os.path.join(file_path_data,nam_gen, action )
dend_path_inits=[nam for _ in range(len(dend_names))]
dend_names_chld[nam]= dict(  
                # dend_namess=dend_namess ,
                dend_names=dend_names , 
                obj_org_path=obj_org_path,
                dend_last=dend_last ,
                dend_first=dend_first ,
                dend_path_inits=dend_path_inits, 
                param_flow=param_flow,
                
                )  

rhs_param={mm:{} for mm in ['dim','tf_mean']}
rhs_param['dim']['i'],rhs_param['dim']['j']=[0],[Nperiod]

rhs_param['g_names']=g_names=['posi','momen','press_0','press']
configs=get_configs(Nmarkov=Nmarkov,nTest=n_obj//6,nSample_interval=1,pm=param_flow,rhs_param=rhs_param)

gdas=get_data_all(names_dic=dend_names_chld,
                  dend_data=gdas.dend_data,
                  file_path_data=file_path_data,
                  cpath=[nam_gen, action],)  




# for ii,g_name in enumerate(['pos','mom','press_0','press']):
# nma=f'mean_{rhs_dim_i}{rhs_dim_j}' if tf_mean else f'{rhs_dim_i}{rhs_dim_j}'

dnn_modess=['DNN-0','DNN-1','DNN-2' , 'DNN-3',]
dnn_modes = dnn_modess+[f'{mm}-mean' for mm in dnn_modess]
 

rhs_dim_i,rhs_dim_j=0,Nperiod
dnn_modes=[]
for tf_mean in [True]:    
     for ii,g_name in enumerate(g_names):
        nma=f'mean_{rhs_dim_i}{rhs_dim_j}'   if tf_mean else f'{rhs_dim_i}{rhs_dim_j}'
        mo=f'{g_name}--{nma}'
        dnn_modes.append(mo)



weight=1
path_dir_nam=f'dice'
path_dir_nam=f'loss'
path_dir=f'{path_dir_nam}_we_{weight}' 
data_dir=path_dir

dict_param=get_dict_param(nam=nam,
                    n_step = 0,   
                    configs=configs,
                    dnn_modes=dnn_modes,
                    )


action='train'

nam=f'{nam_gen}_{action}'

dict_param['path_heads_show']=[f'dnn_PINN_NELD___{nam}',f'dnn_GINN_NELD___{nam}'] 





