 

import os, sys ,dash 
file_path_org=os.getcwd()
sys.path.append(file_path_org ) 
file_path_org=os.path.join(os.path.dirname(file_path_org) ,'files')
from  neld_fun_0.app_param_test import app_param
from dash import callback  

neld_names = ['d004']
neld_namess = ['d004_', 'd004', 'd004']
neld_path_inits =['shear_harmonic_2_200_5_1000_test']
data_studied = 'test'  
model_sufix = 'press_0--mean_05' 
path_train= None
path_file= None
path_file_sub=None
path_file_dir='/Users/akag/Desktop/Backward-Equation-Informed-Neural-Network-BEINN--main/files/app/pages/test/dnn_PINN_NELD___shear_harmonic_2_200_5_1000_train/press_0--mean_05/files/data_initial/shear_harmonic_2_200_5_1000/test/path_files.pkl'
pinn_dir_data='None'
neld_data=None
index=4
model_type='dnn_PINN_NELD___shear_harmonic_2_200_5_1000_train'
obj_org_path_dict=None
model_sufix_dic=None
path_display=None
path_display_dic=None
mapp = app_param(
    file_path_org=file_path_org,
    model_sufix=model_sufix,
    path_train=path_train, 
    path_file=path_file,
    path_file_sub=path_file_sub,
    path_file_dir=path_file_dir,
    pinn_dir_data=pinn_dir_data,
    index=index, 
    data_studied=data_studied,  
    neld_data=neld_data,
    model_type=model_type,
    obj_org_path_dict=obj_org_path_dict,
    model_sufix_dic=model_sufix_dic,
    path_display=path_display,
    path_display_dic=path_display_dic,
)

title_neld_name = f'd004'
dash.register_page(__name__, title=title_neld_name, name=title_neld_name, order=4) 

def layout():
    return mapp.app_layout

@callback(
    mapp.Output,
    mapp.Input, 
    prevent_initial_call=mapp.prevent_initial_call
)
def update_output(*args):  
    return mapp.Get_output(*args)
