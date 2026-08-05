 
import sys
import os
 
import pickle
# import dash
from dash import dcc, html, dash_table, Input, Output, State, callback 
import dash_bootstrap_components as dbc
import numpy as np  
from dend_fun_0.get_path import get_files,get_app_param ,safe_id  













class get_layout:
    def __init__(self): 
        self.app_layout = html.Div(
            style={
                # 'color': 'black',
                # 'backgroundColor': 'grey',
                # 'height': '100vh'  # Full viewport height
                'font-size': 20
            },
            children=self.Get_children()
        )
        

        self.Output=[
            Output(self.output_graph_1['id'], 'children'), 
            Output(self.output_text_1['id'], 'children'), 
        ]
        self.Input=[
            # Input(self.dropdown_true_keys['id'],      'value'),
            Input(self.dropdown_path_head['id'],      'value'), 
            Input(self.dropdown_model_suf['id'],      'value'),
            Input(self.dropdown_path['id'],      'value'),
            Input(self.dropdown_mode['id'],      'value'),
            # Input(self.dropdown_dend['id'],      'value'),
            # Input(self.dropdown_cluster['id'],   'value'), 
            # Input(self.dropdown_intensity['id'], 'value'),
            Input(self.width_slider['id'],           'value'),
            Input(self.height_slider['id'],          'value'), 
            Input(self.dropdown_template['id'], 'value'),
            Input(self.hist_slider['id'],           'value'),
            Input(self.dropdown_index['id'],   'value'), 
        ], 
 





    def Get_children(self): 
        svv=[
            html.Label('Histogram Bin Count:', style=self.dropdown_options_style),
            dcc.Slider(
                id=self.hist_slider['id'],
                min=self.hist_slider['min'],
                max=self.hist_slider['max'],
                step=self.hist_slider['step'],
                value=self.hist_slider['value'],
                marks=self.hist_slider['marks'],
            ),  
            html.Br(),
            html.Label('Graph Width:', style=self.dropdown_options_style),
            dcc.Slider(
                id=self.width_slider['id'],
                min=self.width_slider['min'],
                max=self.width_slider['max'],
                step=self.width_slider['step'],
                value=self.width_slider['value'],
                marks=self.width_slider['marks'],
            ),
            html.Br(),
            html.Label('Graph Height:', style=self.dropdown_options_style),
            dcc.Slider(
                id=self.height_slider['id'],
                min=self.height_slider['min'],
                max=self.height_slider['max'],
                step=self.height_slider['step'],
                value=self.height_slider['value'],
                marks=self.height_slider['marks'],
            ),
            html.Br(),
    ] 

        dfc = []
        dropdown_sources = [
            self.dropdown_index,
            self.dropdown_path_head,
            self.dropdown_model_suf,
            self.dropdown_path,
            self.dropdown_mode,
            # self.dropdown_intensity,
            # self.dropdown_cluster,
            # self.dropdown_dend,
            self.dropdown_template,
        ] 
        for key in dropdown_sources: 
            kwargs = key.copy()
            if 'option' in kwargs and 'options' not in kwargs:
                kwargs['options'] = kwargs.pop('option')
                
            dfc.extend([dcc.Dropdown(**kwargs, style=self.box_style), html.Br()])

        dffc = dfc + svv


        return [ 
            # Additional Dropdown for Graph 2
            html.Br(),
            dbc.Row([
                dbc.Col(
                    html.Div(dffc),
                    style={'flex': '0 0 20%'}  # Set the width of the column using flex property
                ),
                # Column 2: Graph Output
                dbc.Col(
                    html.Div([
                        html.Br(),
                        # Graph 1 Output
                        html.Div(id=self.output_graph_1['id'], 
                                 style=self.output_graph_1['style']),
                        html.Br(),
                        html.Div(id=self.output_text_1['id'], 
                                 style=self.output_graph_1['style']
                                ),
                        html.Br(),
                    ]),
                    style={'flex': '0 0 70%'}  # Set the width of the column using flex property
                ),
            ], style={
                'display': 'flex',
                'justify-content': 'center',
                'align-items': 'center',
                'margin-top': '20px',
                'margin-bottom': '20px'
            }),
            # Text Output
            html.Br(),
            html.Br(), 
        ]


 

from dend_fun_0.app_get_pinn_neld import class_data


import random
from pathlib import Path 

dtype = float
class app_param(get_files,get_app_param,get_layout,class_data): 
    def __init__(self, file_path_org, 
                model_sufix,
                path_train,
                index=0, 
                dend_names=None,
                dend_namess=None, 
                data_studied=None,
                dend_path_inits=None,
                path_file=None,
                path_file_sub=None,
                prevent_initial_call=False,  
                pinn_dir_data=None,
                dend_data=None,
                path_display_dic=None, 
                path_display= ['dest_hmod_path', 'dest_smod_path',],
                model_type=None,
                obj_org_path_dict=None,
                model_sufix_dic=None, 
                path_file_dir=None,
                param_dic=None,

                 ):
        pass 
        path_dir=os.path.join(file_path_org, 'data')
        model_sufix_all=np.loadtxt(os.path.join(path_dir, 'model_sufix_all.txt'),dtype=str,ndmin=1) 
        pinn_dir_data_all=np.loadtxt(os.path.join(path_dir, 'pinn_dir_data_all.txt'),dtype=str,ndmin=1)
        path_heads=np.loadtxt(os.path.join(path_dir, 'path_heads.txt'),dtype=str,ndmin=1)
        true_keys=np.loadtxt(os.path.join(path_dir, 'true_keys.txt'),dtype=str,ndmin=1)
        self.path_heads_show=None
        self.path_heads_show = self.path_heads_show if self.path_heads_show is not None else path_heads
        self.file_path_org=file_path_org
        print('[[[[[[[[[[[[[[[[[[[99999999999999999]]]]]]]]]]]]]]]]]]]',Path(path_file_dir))
        if path_file_dir is not None:
            path_file_dir=Path(path_file_dir)
            with open(path_file_dir, "rb") as f: 
                loaded_dict = pickle.load(f) 
            path_file_dir=loaded_dict['path_file_dir']
            path_train=loaded_dict['path_train']
            self.path_file_init=path_file=loaded_dict['path_file']
            self.path_file_sub_init=loaded_dict['path_file_sub']
            pinn_dir_data=loaded_dict['pinn_dir_data']
            dend_data=loaded_dict['dend_data']
            obj_org_path_dict=loaded_dict['obj_org_path_dict']
            model_sufix_dic=loaded_dict['model_sufix_dic']
            self.path_display=loaded_dict['path_display']
            path_display_dic=loaded_dict['path_display_dic']
            self.path_heads_show=model_sufix_dic.get('path_heads_show',None)
            self.train_neld_param=loaded_dict['train_neld_param']
            # Default structure
            self.param_dic = {
                hh: {
                    kk: True
                    for kk in [
                        'get_pinn_features',
                        'get_wrap',
                        'get_scale',
                        'get_smooth',
                        'get_hmod_pred',
                        'get_dend_name',
                    ]
                }
                for hh in ['tf_restart', 'get_info', 'data']
            }

            # Load saved param_dic if available
            self.param_dic = loaded_dict.get('param_dic', self.param_dic)

            # Ensure data/get_dend_name has the correct structure
            if self.param_dic['data']['get_dend_name'] ==True:
                self.param_dic['data']['get_dend_name'] =  {
                                                            'dict_dend_path': 'current',
                                                            'drop_dic_name': None,
                                                        } 



        if dend_data is not None: 
            dend_names=dend_names if not None else dend_data['dend_names']
            dend_namess=dend_namess if not None else dend_data['dend_namess']
            dend_path_inits=dend_path_inits if not None else dend_data['dend_path_inits'] 
        get_files.__init__(self,
                            dend_data=dend_data,
                            file_path_org=file_path_org,
                            dend_names=dend_names,
                            dend_namess=dend_namess, 
                            dend_path_inits=dend_path_inits,
                            data_studied=data_studied,   
                            model_sufix=model_sufix,
                            path_file=path_file,
                            path_file_sub=path_file_sub,
                            pinn_dir_data=pinn_dir_data,
                            pinn_dir_data_all=pinn_dir_data_all,
                            model_sufix_all=model_sufix_all,
                            true_keys=true_keys,
                            model_type=model_type,
                            path_heads=path_heads, 
                            obj_org_path_dict=obj_org_path_dict,
                            model_sufix_dic=model_sufix_dic,
                            path_display_dic=path_display_dic, 
                            param_dic=self.param_dic,
                            # path_file_dir=path_file_dir,
                 )
        self.path_train=path_train
        self.index=index
        get_app_param.__init__(self,
                               dropdown_path_head_option=self.dropdown_path_head_option,
                               dropdown_model_suf_option=self.dropdown_model_suf_option,
                               dropdown_path_option=self.dropdown_path_option,
                               dropdown_true_keys_option=self.dropdown_true_keys_option,
                               ) 
        class_data.__init__(self)
        self.get_model_opt_name(model_sufix=model_sufix,model_type=model_type ) 
        self.get_dend_name(data_studied=data_studied,index=0, 
                            **self.param_dic['data']['get_dend_name'],) 
        cxc=self.param_dic['data']['get_dend_name']['dict_dend_path'] 
        file_path=self.file_path=self.dict_dend['path'][cxc]['file_path'] 
        path_dir=self.model_sufix_dic['path_dir'] 


         
        dend_name=self.dend_name or dend_names[index]
        dend_namess=self.dend_namess    
        self.prevent_initial_call=prevent_initial_call
        num = random.randint(100000000, 9999999999) 
        id_name_end=f'{model_type}_{dend_name}_{index}_{file_path}_{model_sufix}_{num}'
        id_name_end=safe_id(id_name_end)    
            
        smod_path = self.path_file[path_train['data_hmod_path']]   
        # smod_path = self.path_file[path_train['dest_smod_path']] 
        iou_path=os.path.join(smod_path , self.txt_smod_iou) 
        if os.path.exists(iou_path):
            iou_count = np.loadtxt(iou_path, dtype=float)
        else:
            iou_count=np.array([[-1,-1,0,0]])
        # print('iou_count',len(iou_count),id_name_end,dend_name) 
        iou_count=iou_count if iou_count.ndim==2 else np.array([[-1,-1,0,0]])
        
        self.model_test()
        self.more_param(id_name_end=id_name_end,
                        model_type=model_type,
                        model_sufix=model_sufix,
                        path_dir=path_dir,
                        dend_name=dend_name,)
        self.get_dropdown_cluster(id_name_end,iou_count[:,:2].astype(int))
        self.get_dropdown_index(id_name_end=id_name_end,
                                dend_names=dend_data['dend_names'],
                                index=index,)
        print(dend_names)
        self.get_data(
                    dend_data=self.dend_data,
                    model_sufix=model_sufix,
                    data_studied=self.data_studied,
                    index=index,
                    )  

        get_layout.__init__(self)
