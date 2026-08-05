import os,sys 




force='zero'
force='harmonic'

flow = 'shear'  
flow='pef' 
nPart = 1  
flow = 'eld'                         # Number of particles
epsilon = 2.0                     # rate of the deformation of the background flow
rcut = 30                         # radius cut
beta,gamma=1,1
N = 50                             # number of steps in a period
Nperiod =  10000                   # number of periods
Nperiod =  2                  # number of periods
N = 200                             # number of steps in a period

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
        N = N,                             # number of steps in a period
)



path_0=os.getcwd() 


 
# file_path_org=os.getcwd()  
# file_path_org=os.path.join(os.path.dirname(file_path_org) ,'files')
# file_path_data=os.path.dirname(file_path_org)  
# gdas=get_data(file_path_data) 



 # 
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


import numpy as np

Nmarkov=500

# nam_gen=f'neld_{flow}_{nPart}_{force}'

nam_gen=f'neld_{flow}_{force}_{nPart}_{N}_{Nperiod}_{Nmarkov}'
action='train'
nam=f'{nam_gen}_{action}'
obj_list=np.arange(100)
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

configs=get_configs(Nmarkov=Nmarkov,nTest=5,nSample_interval=1,pm=param_flow)

gdas=get_data_all(names_dic=dend_names_chld,
                  dend_data=gdas.dend_data,
                  file_path_data=file_path_data,
                  cpath=[nam_gen, action],)  












weight=1
path_dir_nam=f'dice'
path_dir_nam=f'loss'
path_dir=f'{path_dir_nam}_we_{weight}' 
data_dir=path_dir

dict_param=get_dict_param(nam=nam,
                    n_step = 0,   
                    configs=configs,
                    )








