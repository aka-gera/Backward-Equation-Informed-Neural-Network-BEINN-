 
import sys,os
DTYPE='float32'  
from beinn.neld_fun_0.main_0 import  app_run_param,algorithm,algorithm_param,get_data, get_data_all ,get_dict_param
 


from beinn.neld_pinn_0.run_00 import get_neld_data,get_neld_data_train
path_dict={'run':'beinn.neld_pinn_0.run_00'}

from neld_data_1_shear import gdas,param_flow,nam_gen,dict_param,data_dir,Nmarkov,N,Nperiod,nma,g_names,dnn_modes
from neld_data_harmonic_2_shear import gdas,param_flow,nam_gen,dict_param,data_dir,Nmarkov,N,Nperiod,nma,g_names,dnn_modes


flow,nPart,force=param_flow['flow'],param_flow['nPart'],param_flow['force']

 

# nam_gen=f'neld_{flow}_{nPart}_{force}'
action='train'

nam=f'{nam_gen}_{action}'
 
 
model_type=f'dnn_GINN_NELD___{nam}' 
model_type=f'dnn_PINN_NELD___{nam}'


action='train'
action='test'

nam=f'{nam_gen}_{action}'
# Pick feature type     
dnn_mode = 'DNN-3'                     

 
weight=1
path_dir=f'oloss_we_{weight}'  
path_dir=f'loss_we_{weight}' 

data_dir=path_dir
# Pick feature type     
dnn_mode = "DNN-4"        
dnn_mode = 'DNN-2'        
          
                 
dnn_mode = 'DNN-1'
dnn_mode = 'DNN-3' 
dnn_mode = 'DNN-0'
dnn_mode = 'DNN-0-mean'     


g_names=['pos','mom','press_0','press'] 
g_name=g_names[3]
g_name=g_names[2]
dnn_mode=f'{g_name}--{nma}'

# dict_param['path_heads_show']=[model_type,] 
dict_param['model_type_data']=model_type 
param = algorithm_param( **dict_param,)


neld_data = gdas.part(nam)  
mapp = app_run_param(param)
param=mapp.emerge_param()  
param['dnn_modes']['param']=dnn_modes
print('[[[[[[]]]]]]',param['dnn_modes']['param'])
param['model-pred']['tf'] = True    #False   # True/False: choose whether to predict hmods/smods
param['skl_hmod_pred']['tf']  =False   #True  #
param['dash_pages']['tf'] =True # False   #True          #  True/False: generate Dash pages
param['roc']['tf']=False   #True # 


# param['model-pred']['tf'] =False   # True    # True/False: choose whether to predict hmods/smods
# param['skl_hmod_pred']['tf']  =False   #True  #
# param['dash_pages']['tf'] =True # False   #True          #  True/False: generate Dash pages
 

alg = algorithm(param)
self=alg.test(   
    neld_data = neld_data,  
    true_name = 'true_0', 
    dnn_mode = dnn_mode,
    model_type = model_type, 
    path_dir=data_dir,
    data_dir=data_dir,
    path_dict=path_dict,
    **dict_param 
)


'''
get_neld_data(**param_flow,self=self,
                  dt_r_0=3,
                  dt_r1=1,                  
                fmt='%.4f',
                Nmarkov=Nmarkov,
                # Nsample=50,
                # tf_train=True,
    )
'''