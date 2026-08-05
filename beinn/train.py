 
#  # 
import sys,os 
DTYPE='float32'   
import time 
import random#
import numpy as np

  
from neld_fun_0.main_0 import  app_run_param,algorithm,algorithm_param,get_data, get_data_all ,get_dict_param
  


path_dict={'run':'neld_pinn_0.run_00'}

from neld_data_2_shear import gdas,param_flow,nam_gen,dict_param,data_dir,Nmarkov,N,configs,nma,g_names

from neld_data_harmonic_2_shear import gdas,param_flow,nam_gen,dict_param,data_dir,Nmarkov,N,Nperiod,nma,g_names,dnn_modes





flow,nPart,force=param_flow['flow'],param_flow['nPart'],param_flow['force']

  
action='test'
action='train'

nam=f'{nam_gen}_{action}'

 


 # Pick name for the model  

model_type=f'dnn_GINN_NELD___{nam}' 
model_type=f'dnn_PINN_NELD___{nam}' 

 

g_names=['pos','mom','press_0','press'] 
g_name=g_names[3]
g_name=g_names[2]
dnn_mode=f'{g_name}--{nma}'



path_heads_show=[  
            f'dnn_PINN_NELD',  
] 


path_heads_show=path_heads_show if model_type in path_heads_show else path_heads_show+[model_type]
path_heads=[  ] 
path_heads = list(set(path_heads+path_heads_show))
if model_type not in path_heads:
    path_heads.append(model_type) 
 
  
path_display = ['dest_hmod_path', ]

  
dict_param['model_type_data']=model_type 
param = algorithm_param(**dict_param)



neld_data = gdas.part(nam)  
mapp = app_run_param(param)
param=mapp.emerge_param()
param['Smooth']['tf'] = False   #  False   #False   # False   # True/False: choose whether to smooth the data
param['annotations']['tf'] =  False   # True   # True/False: choose whether to generate annotations for training, accuracy, and recall 
param['get_training']['tf'] = True  # True/False: enable training data generation
param['clean_path_dir']['tf'] = False          # True/False: delete computed data
mj = [x / 1000 for x in range(1, 1000, 200)]  
param['get_training']['param']['weight']=[[a, b] for a, b in zip(mj, mj[::-1])]
param['model_pred']['param']['param_dic']['tf_restart']['get_pinn_features']= False   #True   #

entry_names=[None,]
ls =[1,2,3,4,] 
  
 
param['get_training']['param']['ls']=ls
param['get_training']['param']['itime']=100000 
alg = algorithm(param)
alg.train(
    neld_data = neld_data,  
    true_name = 'true_0', 
    dnn_mode = dnn_mode,
    model_type = model_type,  
    path_display = path_display, 
    entry_names=entry_names,
    path_dir=data_dir,
    path_heads_show=path_heads, 
    configs=configs,
    path_dict=path_dict,
)
