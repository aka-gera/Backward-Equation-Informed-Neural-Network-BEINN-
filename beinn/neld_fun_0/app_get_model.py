

import sys
import os
 
import pickle
# import dash
from dash import dcc, html, dash_table, Input, Output, State, callback 
import dash_bootstrap_components as dbc
import numpy as np
import beinn.neld_fun_0.curvature as cu  
from beinn.neld_fun_0.help_funn import get_color
import beinn.neld_fun_0.help_fun as hf
# import geometry as geo
# import help_plotly as hpp
import beinn.neld_fun_0.help_plotly as hp
from beinn.neld_fun_0.help_plotly import aka_plot 
# import density as den
import plotly.graph_objects as go 
import beinn.neld_fun_0.help_funn as hff
from beinn.neld_fun_0.get_path import get_files,get_app_param ,safe_id 
from beinn.neld_fun_0.help_graph import get_iou_graph,get_cm_iou,compute_kl 
from beinn.neld_fun_0.help_save_iou import iou_train
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, auc
import pickle




ccoll = [
        'red', 'yellow', 'blue', 'green', 'black', 'purple',
        'orange', 'pink', 'brown', 'cyan', 'magenta', 'lime',
        'teal', 'navy', 'maroon', 'olive', 'gold', 'silver'
    ] 
     
ccol=[]
for _ in range( 1000):
    ccol=ccol+ccoll


def add_unit(label, unit="µm",):
    return f"{label} ({unit})"

def add_unit(label: str, unit="µm",) -> str:
    label_lower = label.lower()
    if any(word in label_lower for word in ["length", "width", "diameter"]):
        return f"{label} ({unit})"
    elif "area" in label_lower:
        return f"{label} ({unit}<sup>2</sup>)"
    elif "volume" in label_lower:
        return f"{label} ({unit}<sup>3</sup>)"
    else:
        return label   

def get_metric(
    akp,mode, metric, metric_name, width=500, height=500, 
    nbinsx=30, xtitle='Length', ytitle='Count'
):
    metric_map = {
        metric_name[0]: 0,
        metric_name[1]: 1,
        metric_name[2]: 2,
    }
    
    if mode not in metric_map:
        raise ValueError(f"Invalid mode: {mode}")

    index = metric_map[mode]
    data = metric[:, index]
    
    title = f'{mode} Histogram'

    scatter_comp, layout_comp = hp.Plotly_histogram_return(
        data, nbinsx=nbinsx, title=title, xtitle=xtitle, ytitle=ytitle, width=width, height=height
    )
    
    return akp.Plotly_Figure(data=scatter_comp, layout=layout_comp)








class class_metric_figure:
    def __init__(self):
        pass

    def get_figure(self,true_keys, path_head,model_suf,path,vertices_pl,intensity, spid, width, height,ffcouleur ):   

        id_path=f'{true_keys}_save_save' if path_head=='true' else f'{path_head}_{model_suf}_{path}'
        id_pathss=f'{path_head}_{model_suf}_{path}_{true_keys}'
        smod_path=self.path_file[id_path] 
        # hmod_path=self.path_mapping[self.name_smod][path]  
        if (spid == 0) : 
            self.figure_3d=cu.plotly_mesh(vertices=vertices_pl,
                                                    faces=self.faces ,
                                                    intensity=intensity,
                                                    width=width, 
                                                    height=height,
                                                    colorscale='purd')   
            self.scatter=[ 
                    hf.plotly_scatter(points=vertices_pl ,
                                      color='red',
                                      size=1.07,
                                      opacity=.8,
                                      name='Approx'),   
                    ]
            self.layout = go.Layout(width=width, 
                            height=height,
                            title=f'model', 
                            ) 
 
        elif spid==1: 
            self.smod_index =smod_index = np.loadtxt(os.path.join(smod_path,self.txt_hmod_index), dtype=int) 
            self.faces_index =faces_index = np.loadtxt(os.path.join(smod_path,self.txt_hmod_faces),dtype=int)
            self.figure_3d=cu.plotly_mesh(vertices=vertices_pl[smod_index],
                                                    faces=faces_index,
                                                    intensity=intensity[smod_index],
                                                    width=width, 
                                                    height=height,
                                                    colorscale='purd') 

            self.scatter=[ 
                    hf.plotly_scatter(points=vertices_pl[smod_index],
                                      color='red',
                                      size=.7,
                                      opacity=.6,
                                      name='Approx'),   
                    ]  
            self.layout = go.Layout(width=width, 
                            height=height,  
                            )

        else:
            color='red'  
            clustss = spid - 2
            clustsss = spid - 2
            if id_pathss in self.iou_tr: 
                self.clusts=clustss=self.iou_tr[id_pathss].count_appr_tmp[clustss]
                
            # scatter=[] 
            self.smod_index = smod_index = np.loadtxt(os.path.join(smod_path, f'{self.name_smod}_{self.name_index}_{clustss}.txt'),dtype=int)
            self.smod_faces =smod_faces  = np.loadtxt(os.path.join(smod_path, f'{self.name_smod}_{self.name_faces}_{clustss}.txt'),dtype=int)
            # print(os.path.join(smod_path, f'{self.name_smod}_{self.name_index}_{clustss}.txt') )

            self.figure_3d=cu.plotly_mesh(vertices=vertices_pl[smod_index],
                                                    faces=smod_faces,
                                                    intensity=intensity[smod_index],
                                                    width=width, height=height,
                                                    colorscale='purd') 
            if id_pathss in self.iou_tr:
                self.scatter= self.iou_tr[id_pathss].get_graph(vertices_0=vertices_pl,index=clustsss)
                rate,rate_un=self.iou_count[id_path][clustsss,-2:]
                # clustss=int(self.iou_count[path][clustss,0])
                self.clusts=clustss=self.iou_tr[id_pathss].count_appr_tmp[clustsss]
                
                
                self.layout = go.Layout(width=width, 
                                height=height,
                                title=f'model {self.neld_name} smod <br>UOI         : {rate:.2f}<br>UOI union: {rate_un:.2f}',  
                                )   
            else:
                self.scatter=[ 
                        hf.plotly_scatter(points=vertices_pl[smod_index],
                                        color='red',
                                        size=.7,
                                        opacity=.6,
                                        name='Approx'),   
                        ]  
                self.layout = go.Layout(width=width, 
                                height=height,  
                                )

        self.scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            zaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            bgcolor=ffcouleur 
        )



    def get_metric(self,mode,metric,width=800,height=800,nbinsx=10000,opacity=1.,color='blue',name=None): 
        self.scatter_metric,self.layout_metric=hp.Plotly_histogram_return(data=metric,
                                                                        nbinsx=nbinsx,
                                                                        title=mode,
                                                                        xtitle=self.metric_mapping['xtitle'],
                                                                        ytitle='Count',
                                                                        width=width, 
                                                                        height=height,
                                                                        opacity=opacity,
                                                                        color=color,
                                                                        name=name,)












class class_data:
    def __init__(self):
        pass

    def get_data(self,model_sufix,data_studied,index,neld_data=None): 
        path_train=self.path_train
        if neld_data is not None:
            self.neld_path_inits= neld_data['neld_path_inits']
            self.neld_names= neld_data['neld_names']
            self.neld_namess= neld_data['neld_namess']
        self.get_model_opt_name(model_sufix=model_sufix,model_type=self.model_type )
        self.get_neld_name(data_studied=data_studied,
                           index=index,
                            neld_names= self.neld_names,
                            neld_namess=self.neld_namess,  
                            **self.param_dic['data']['get_neld_name'],) 
        cxc=self.param_dic['data']['get_neld_name']['dict_neld_path'] 
        file_path_feat=self.dict_neld['path'][cxc]['file_path_feat']
        file_path=self.file_path=self.dict_neld['path'][cxc]['file_path'] 
        smod_path= self.path_file[path_train['dest_hmod_path']] 
        # smod_path= self.path_file[path_train['dest_smod_path']]  
        neld_name=self.neld_name 
        neld_namess=self.neld_namess 
        self.plot_data_iou=None 
        self.model_shap_dic={}
        self.plot_data_iou_dic={}
        self.plot_data_center_curv={} 
        self.plot_data_cylinder_heatmap={} 
        self.annot_intensity={ke:{} for ke in self.inten_file_train} 
        # self.logit_intensity_head_neck={ke:{} for ke in self.intensity_head_neck_logit} 
        # self.logit_intensity={ke:{} for ke in self.intensity_smods_logit} 
        self.logit_intensity={ke:{} for ke in self.intensity_logit} 
        self.iou_count={}
        self.scatter_loss_dic={}
        self.scatter_iou_dic={}
        self.scatter_loss_smod_dic={}
        self.scatter_iou_smod_dic={}
        self.scatter_auc_smod_dic={}
        self.scatter_dice_smod_dic={}
        self.metric_total_dic={}
       #   self.metric_total_dic['union']['single']={}
        self.metric_total_dic={}
        self.metric_total_dic['single']={}
        self.metric_total_dic['union']={}
        self.metric_total_dic['single_dice']={}
        self.metric_total_dic['union_dice']={}
        self.iou_tr={}
 

        # self.metric_total_dic = {
        #     key: {}
        #     for iii in self.neld_path_original_mm['keys'].values() 
        #     for key in (f'single_{iii}', f'union_{iii}')
        # } 
        # print('[[[[[[[[[=============================]]]]]]]]]',self.path_heads,self.path_heads_show)
        for path_head in self.model_sufix_dic['path_heads_show']:
            for model_suf in self.model_sufix_dic['model_sufix_show']:
                for path    in self.model_sufix_dic['path_dirs_show']: 
                    id_path=f'{path_head}_{model_suf}_{path}' 
                    # if not path in self.path_display_dic['path']:
                    #     continue 
                    if id_path not in self.path_file:
                        continue
                    fgff=os.path.join(self.path_file[id_path], self.txt_smod_iou)
                    if os.path.exists(fgff):
                        self.iou_count[id_path]=  np.loadtxt(fgff, dtype=float)    

                    path_grap_center_curv=os.path.join(self.path_file[id_path] ,'plot_data_center_curv.pkl')
                    if os.path.exists(path_grap_center_curv):
                        with open(os.path.join(path_grap_center_curv), "rb") as file:
                            self.plot_data_center_curv[id_path] = pickle.load(file) 

                    path_grap_center_curv=os.path.join(self.path_file[id_path] ,'cylinder_heatmap.pkl')
                    if os.path.exists(path_grap_center_curv):
                        with open(os.path.join(path_grap_center_curv), "rb") as file:
                            self.plot_data_cylinder_heatmap[id_path] = pickle.load(file) 
                            
                    for intensity_type in self.inten_file_train :
                        path_grap_center_curv=os.path.join(self.path_file[id_path] ,f'{intensity_type}.txt')
                        if os.path.exists(path_grap_center_curv):
                            self.annot_intensity[intensity_type][id_path] = path_grap_center_curv


                    ''' DONT DELETE
                    for intensity_type in self.intensity_smods_logit:
                        path_grap_center_curv=self.path_file_sub[intensity_type][id_path] 
                        if os.path.exists(path_grap_center_curv):
                            self.logit_intensity[intensity_type][id_path] = path_grap_center_curv





                    self.scatter_loss_data=[] 
                    tyy =self.inten_file_model_head_neck_loss[0]
                    loss_path=self.path_file_sub[tyy][id_path]
                    # print('ooop------===================================',loss_path)

                    if os.path.exists(loss_path):
                        self.loss_data = np.loadtxt(loss_path, dtype=float)  
                        self.loss_data= np.vstack((np.arange(len(self.loss_data)),self.loss_data)).T
                        # print('pp----------',self.loss_data.shape)
                        self.scatter_loss_data.append(
                                hf.plotly_scatter(points=self.loss_data ,
                                                color='red',
                                                size=4.07,
                                                opacity=.8,
                                                name='Approx')
                                                ) 
                    self.scatter_loss_dic[id_path]=  self.scatter_loss_data
 

                    tyy =self.inten_file_model_head_neck_iou[0] 
                    self.scatter_iou_data=[]   
                    for ii,(tyyy,nam,couleur) in enumerate(zip(self.inten_file_model_train_iou[:-1],['head','neck','hmod'],['red','green','blue'])):
                        # iou_path=self.model_dir_path['head_neck']['iou'][ii]
                        iou_path=self.path_file_sub[tyy][id_path][tyyy]
                        if os.path.exists(iou_path):
                            _data=np.loadtxt(iou_path,dtype=float)
                            print('pp----------',_data.shape) 
                            for ii in range(_data.shape[1]):
                                _dataa= np.hstack((np.arange(_data.shape[0]).reshape(-1,1),_data[:,ii:ii+1]))
                                self.scatter_iou_data.append(hf.plotly_scatter(points=_dataa ,
                                                        color=couleur,
                                                        size=4.07,
                                                        opacity=.8,
                                                        name=f'{nam}_{ii}'))

                    self.scatter_iou_dic[id_path]=self.scatter_iou_data

'''



                    scatter_loss_data=[] 
                    tyy =self.inten_file_model_smod_loss[0]
                    loss_path=self.path_file_sub[tyy].get(id_path,None)
                    # print('ooop------',loss_path,self.path_file_sub[tyy].keys())
                    # print('ooop------',id_path,id_path in list(self.path_file_sub[tyy].keys()))
                    # if loss_path is None:
                        # continueloss_path is not None and 
                    if loss_path is not None and os.path.exists(loss_path):
                        self.loss_data = np.loadtxt(loss_path, dtype=float,ndmin=1)  
                        # print('pp----------',self.loss_data )
                        self.loss_data= np.vstack((np.arange(len(self.loss_data)),self.loss_data)).T
                        scatter_loss_data.append(
                                hf.plotly_scatter(points=self.loss_data ,
                                                color='red',
                                                size=4.07,
                                                opacity=.8,
                                                name='Approx',)
                                                ) 

                    self.scatter_loss_smod_dic[id_path]= scatter_loss_data

                                     
                    tyy =self.inten_file_model_smod_iou[0] 
                    scatter_iou_data=[]   
                    cnt=True
                    for ii,(tyyy,nam,couleur) in enumerate(zip(self.inten_file_model_train_smod_iou,['hmod','smod'],['red','blue'])):
                        # iou_path=self.model_dir_path['head_neck']['iou'][ii]
                        # if  loss_path is None:
                        #     continue
                        if id_path in list(self.path_file_sub[tyy].keys()):
                        #     continue
                            iou_path=self.path_file_sub[tyy][id_path][tyyy]
                            if os.path.exists(iou_path) and os.path.getsize(iou_path) > 0: 
                                # print('[[[[[[[[[[[------========-------]]]]]]]]]]]',os.path.getsize(iou_path))
                                try: 
                                    _data=np.loadtxt(iou_path,dtype=float) 
                                except ValueError: 
                                    continue
                                if len(_data)<=0 or len(_data.shape)==1:
                                    continue
                                m1,m2=.2,.8
                                for ii,(symb,mdd,siz) in enumerate(zip(['star','star','star','star','square','square'],
                                                                ['train','train','train','train','test','test'],
                                                                [m1,m1,m1,m1,m2,m2])):
                                    if ii<_data.shape[1]:
                                        _dataa= np.hstack((np.arange(len(_data)).reshape(-1,1),_data[:,ii:ii+1])) 
                                        scatter_iou_data.append(hf.plotly_scatter(points=_dataa ,
                                                                color=couleur,
                                                                size=10.,
                                                                opacity=siz,
                                                                name=f'{nam}_{mdd}_{ii}',
                                                                symbol=symb))

                        elif os.path.exists(self.df_metric_algorithms_dir) and cnt: 
                            df_union=pd.read_csv(self.df_metric_algorithms_dir)#.sort_values(by=f'Accuracy', ascending=True) 
                            ylabels=df_union.columns[1:]
                            xlabels=df_union.iloc[:,0]
                            df_union = df_union.set_index("Unnamed: 0")
                            df_union=df_union.T.sort_values(by=f'Accuracy', ascending=True)

                            # df_union=df_union[df_union.columns[1:]]
                            cm=np.array(df_union.select_dtypes(include=[np.number]))
                            cm= np.round(cm*1e3)/1e3
 
                            scatter_iou_data.append(
                                                    go.Heatmap(
                                                                z=cm,
                                                                x=xlabels,  
                                                                y=ylabels,  
                                                                colorscale='Blues',
                                                                text=cm,  # Show numbers in cells
                                                                texttemplate="%{text}",  # Format as numbers
                                                                # hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                                                                showscale=False ,
                                                                textfont=dict(size=18),
                                                            ) 
                                            ) 
                            cnt=False
 

                    self.scatter_iou_smod_dic[id_path]=scatter_iou_data



                                     
                    tyy =self.inten_file_model_smod_auc[0] 
                    scatter_data=[]   
                    cnt=True
                    for ii,(tyyy,nam,couleur) in enumerate(zip(self.inten_file_model_train_smod_auc,['hmod','smod'],['red','blue'])): 

                        # if id_path not in list(self.path_file_sub[tyy].keys()):
                        #     continue
                        # iou_path=self.path_file_sub[tyy][id_path][tyyy]
                        # if os.path.exists(iou_path) and os.path.getsize(iou_path) > 0:  
                        #     try: 
                        #         _data=np.loadtxt(iou_path,dtype=float) 
                        #     except ValueError: 
                        #         continue
                        #     if len(_data)<=0 :
                        #         continue   
                        #     scatter_data.append(hf.plotly_scatter(points=_dataa ,
                        #                             color=couleur,
                        #                             size=10.,
                        #                             opacity=siz,
                        #                             name=f'{nam}'
                        #                             ))


                        # iou_path=self.model_dir_path['head_neck']['iou'][ii]
                        # if  loss_path is None:
                        #     continue
                        if id_path in list(self.path_file_sub[tyy].keys()):
                        #     continue
                            iou_path=self.path_file_sub[tyy][id_path][tyyy]
                            if os.path.exists(iou_path) and os.path.getsize(iou_path) > 0: 
                                # print('[[[[[[[[[[[------========-------]]]]]]]]]]]',os.path.getsize(iou_path))
                                try: 
                                    _data=np.loadtxt(iou_path,dtype=float) 
                                except ValueError: 
                                    continue
                                if len(_data)<=0 or len(_data.shape)==1:
                                    continue
                                m1,m2=.2,.8
                                for ii,(symb,mdd,siz) in enumerate(zip(['star','star','star','star','square','square'],
                                                                ['train','train','train','train','test','test'],
                                                                [m1,m1,m1,m1,m2,m2])):
                                    if ii<_data.shape[1]:
                                        _dataa= np.hstack((np.arange(len(_data)).reshape(-1,1),_data[:,ii:ii+1])) 
                                        scatter_data.append(hf.plotly_scatter(points=_dataa ,
                                                                color=couleur,
                                                                size=10.,
                                                                opacity=siz,
                                                                name=f'{nam}_{mdd}_{ii}',
                                                                symbol=symb))



                    self.scatter_auc_smod_dic[id_path]=scatter_data

 
 


                                     
                    tyy =self.inten_file_model_smod_dice[0] 
                    scatter_data=[]   
                    cnt=True
                    for ii,(tyyy,nam,couleur) in enumerate(zip(self.inten_file_model_train_smod_dice,['hmod','smod'],['red','blue'])): 

                        # if id_path not in list(self.path_file_sub[tyy].keys()):
                        #     continue
                        # iou_path=self.path_file_sub[tyy][id_path][tyyy]
                        # if os.path.exists(iou_path) and os.path.getsize(iou_path) > 0:  
                        #     try: 
                        #         _data=np.loadtxt(iou_path,dtype=float) 
                        #     except ValueError: 
                        #         continue
                        #     if len(_data)<=0 :
                        #         continue   
                        #     scatter_data.append(hf.plotly_scatter(points=_dataa ,
                        #                             color=couleur,
                        #                             size=10.,
                        #                             opacity=siz,
                        #                             name=f'{nam}'
                        #                             ))

                        if id_path not in list(self.path_file_sub[tyy].keys()):
                            continue
                        iou_path=self.path_file_sub[tyy][id_path][tyyy]
                        if os.path.exists(iou_path) and os.path.getsize(iou_path) > 0: 
                            # print('[[[[[[[[[[[------========-------]]]]]]]]]]]',os.path.getsize(iou_path))
                            try: 
                                _data=np.loadtxt(iou_path,dtype=float) 
                            except ValueError: 
                                continue
                            if len(_data)<=0 or len(_data.shape)==1:
                                continue
                            m1,m2=.2,.8
                            for ii,(symb,mdd,siz) in enumerate(zip(['star','star','star','star','square','square'],
                                                            ['train','train','train','train','test','test'],
                                                            [m1,m1,m1,m1,m2,m2])):
                                if ii<_data.shape[1]:
                                    _dataa= np.hstack((np.arange(len(_data)).reshape(-1,1),_data[:,ii:ii+1])) 
                                    scatter_data.append(hf.plotly_scatter(points=_dataa ,
                                                            color=couleur,
                                                            size=10.,
                                                            opacity=siz,
                                                            name=f'{nam}_{mdd}_{ii}',
                                                            symbol=symb))

                    self.scatter_dice_smod_dic[id_path]=scatter_data

 

 



                    clor=['red','blue' ]
                    vds=[1,-1]
                    valss=['hmod','smod']
                    figg=go.Figure()
                    # tyy =self.inten_file_model_shap[0]  
                    # iou_path=self.path_file_sub[tyy][id_path] 
                    self.model_shap=[]   
                    head_neck_path = 'dest_hmod_path'
                    pathh=path_train[head_neck_path]
                    smod_path_save=     self.path_file[f'result_{pathh}']
                    iou_path=os.path.join(smod_path_save,'shap.csv') 
                     
                    if os.path.exists(iou_path):
                        # print('---------------->>>>>>>>><<<<<<<<<<<<<<<,,',iou_path)

                        df = pd.read_csv(iou_path)   
                        for nam,cl,vd,va in zip(df.columns[1:],clor,vds,valss):
                            figg.add_trace(
                                go.Bar(
                                    x=vd*df[nam][::-1],
                                    y=df['Feature'][::-1],
                                    orientation='h',
                                    marker_color=cl,
                                    name=f'{va}'
                                )
                            )  
                        figg.update_layout(
                            title='Diverging SHAP Summary',
                            barmode='relative',
                            xaxis_title='SHAP value',
                            yaxis_title='Feature',
                            xaxis=dict(zeroline=True),
                            bargap=.2,
                        )

                    self.model_shap_dic[id_path]=figg






        self.metric_total_dic['roc_curve']={}
        self.metric_total_dic['roc_curve']['curve']={'smod':[],'hmod':[]}
        self.metric_total_dic['roc_curve']['score']={'smod':[],'hmod':[]}
        icoci=0
        for iii,(true_path,true_key) in enumerate( self.neld_path_original_mm['keys'].items() ):
            for path_head in self.model_sufix_dic['path_heads_show']:
                for model_suf in self.model_sufix_dic['model_sufix_show']:
                    for path in self.model_sufix_dic['path_dirs_show']: 
                        id_path=f'{path_head}_{model_suf}_{path}'
                        
                        if id_path not in self.path_file:
                            continue
                        # if not path in self.path_display_dic['path']:
                        #     continue
                        # if not id_path in ppatt:
                        #     continue
                        nhh=self.model_sufix_dic['model_sufix_dic'][model_suf],self.model_sufix_dic['path_dirs_dic'][path]
                        path_head_sh=self.model_sufix_dic['path_heads_dic'][path_head]
                        id_path_nam=f'{path_head_sh}_{nhh[0]}_{nhh[1]}'
                        id_pathss=f'{path_head}_{model_suf}_{path}_{true_key}'
                        path_grap_iou=os.path.join(self.path_file[id_path] ,self.pkl_mp)
                        if os.path.exists(path_grap_iou):
                            with open(path_grap_iou, "rb") as file:
                                mp = pickle.load(file) 
                            self.iou_tr[id_pathss]=iou_train(
                                                path_true=self.path_file[true_path],
                                                path_appr=self.path_file[id_path],
                                                mp=mp,
                                                ) 
                            
                        path_grap_iou=os.path.join(self.path_file[id_path] ,f'plot_iou_graph_{true_key}.pkl')
                        if os.path.exists(path_grap_iou): 
                                with open(os.path.join(path_grap_iou), "rb") as file: 
                                    self.plot_data_iou_dic[id_pathss]=pickle.load(file) 
                        '''
                        # true_path=self.neld_path_original_mm['keys']['true_0']
                        path_grap_iou=os.path.join(self.path_file[id_path] ,'intensity_smods_logit.txt')
                        path_grap_true=os.path.join(self.path_file[true_path] ,'intensity_smods_logit.txt') 
                        print(']]]]]]]][[[[[[[[[[[[[]]]]]]]]]]]]]',path_grap_iou)
                        if os.path.exists(path_grap_iou) and os.path.exists(path_grap_true): 
                            print(']]]]]]]][[[[[[[[[[[[[]]]]]]]]]]]]]',path_grap_iou)
                            self.metric_total_dic['roc_curve']['true']+=np.loadtxt(path_grap_true,dtype=float)
                            self.metric_total_dic['roc_curve']['score']+=np.loadtxt(path_grap_iou,dtype=float)
'''

                        if model_suf !='save':
                            smod_path_save=     self.path_file[f'result_{id_path}']
                            for nm,nmm in zip(['hmod','smod'],['hmod','smod']):
                                metric_path=os.path.join( smod_path_save,f'roc_{nm}_{true_key}.txt') 
                                if os.path.exists(metric_path):
                                    roc=np.loadtxt(metric_path,dtype=float)
                                    scr=auc(roc[:,0], roc[:,1])
                                    # print('======[[[[[[]]]]]]',roc)
                                    self.metric_total_dic['roc_curve']['score'][nm]=scr
                                    self.metric_total_dic['roc_curve']['curve'][nm].append(go.Scatter( 
                                                                            x=roc[:,0], 
                                                                            y=roc[:,1], 
                                                                            mode="lines", 
                                                                            name=f"{id_path_nam} (AUC {nmm} = {scr:.3f})", 
                                                                            line=dict(width=3),
                                                                            marker=dict(color=ccol[icoci]),
                                                                            ) ,
                                                                            )
                            icoci+=1


                            
                            smod_path_save=     self.path_file[f'result_{id_path}']
                            metric_path=os.path.join( smod_path_save,f'iou_{true_key}.csv') 
                            if os.path.exists(metric_path):
                                df = pd.read_csv(metric_path)
                                # print('----=====----','im her')
                                # nhf=df['id_true']>0
                                # sze_checks_0 ,sze_check ,sze_check_un=sze_checks_0[nhf] ,sze_check[nhf] ,sze_check_un[nhf]
                                sze_checks_0 ,sze_check ,sze_check_un=df['id_true'],df['iou_single'],df['iou_union']
                                metric_name_path=df.columns
                                iou_dict={}
                                get_cm_iou(sze_checks_0 ,sze_check ,sze_check_un ,iou_dict=iou_dict, iou_per=70,labels = ['False', 'True'],nbinsx=300,) 
                                accuracy , precision, recall, f1_score=iou_dict['single']['metrics'].values() 
                                # self.metric_total_dic[id_path]={}
                                self.metric_total_dic['single'][id_path_nam]=dict(accuracy=accuracy,
                                                                    precision=precision,
                                                                    recall=recall,
                                                                    f1_score=f1_score) 
                                accuracy, precision, recall, f1_score=iou_dict['union']['metrics'].values()
                                self.metric_total_dic['union'][id_path_nam]=dict(accuracy=accuracy,
                                                                    precision=precision,
                                                                    recall=recall,
                                                                    f1_score=f1_score,
                                                                    )
                                
                                if 'dice_single' in list(df.columns):
                                    sze_checks_0 ,sze_check ,sze_check_un=df['id_true'],df['dice_single'],df['dice_union']
                                    metric_name_path=df.columns
                                    iou_dict={}
                                    get_cm_iou(sze_checks_0 ,sze_check ,sze_check_un ,iou_dict=iou_dict, iou_per=70,labels = ['False', 'True'],nbinsx=300,) 
                                    accuracy , precision, recall, f1_score=iou_dict['single']['metrics'].values() 
                                    # self.metric_total_dic[id_path]={}
                                    self.metric_total_dic['single_dice'][id_path_nam]=dict(accuracy=accuracy,
                                                                        precision=precision,
                                                                        recall=recall,
                                                                        f1_score=f1_score) 
                                    accuracy, precision, recall, f1_score=iou_dict['union']['metrics'].values()
                                    self.metric_total_dic['union_dice'][id_path_nam]=dict(accuracy=accuracy,
                                                                        precision=precision,
                                                                        recall=recall,
                                                                        f1_score=f1_score,
                                                                        )
                                


                                for uuyy in ['single','union','single_dice','union_dice',]:
                                    if (uuyy in self.metric_total_dic) and (id_path_nam in self.metric_total_dic[uuyy]):
                                        for nm in ['hmod','smod']: 
                                            self.metric_total_dic[uuyy][id_path_nam][f'AUC_{nm}']=self.metric_total_dic['roc_curve']['score'][nm]






            if len(self.metric_total_dic['single'])>0:
                df_union=pd.DataFrame(self.metric_total_dic['union']).T.sort_values(by='accuracy', ascending=False)
                df_union.to_csv(os.path.join(self.path_file[f'result_appr'],f'metric_union_{true_key}.csv'))
                df_single=pd.DataFrame(self.metric_total_dic['single']).T.sort_values(by='accuracy', ascending=False)
                df_single.to_csv(os.path.join(self.path_file[f'result_appr'],f'metric_single_{true_key}.csv'))
            if len(self.metric_total_dic['single_dice'])>0:
                df_union=pd.DataFrame(self.metric_total_dic['union_dice']).T.sort_values(by='accuracy', ascending=False)
                df_union.to_csv(os.path.join(self.path_file[f'result_appr'],f'metric_union_dice_{true_key}.csv'))
                df_single=pd.DataFrame(self.metric_total_dic['single_dice']).T.sort_values(by='accuracy', ascending=False)
                df_single.to_csv(os.path.join(self.path_file[f'result_appr'],f'metric_single_dice_{true_key}.csv'))





        self.file_path =file_path
        self.neld_name=neld_name
        self.itera=index  
        self.vertices_0=self.vertices_00=self.faces=np.array([[0,0,0]])
        '''

        self.vertices_0=vertices_0       = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_0), dtype=float) 
        
        self.vertices_00=vertices_00      = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_old), dtype=float)
        self.faces=faces = np.loadtxt(os.path.join(self.file_path, self.txt_faces), dtype=int)


        print(f'Starting analysis of {neld_name}')
        print(f"Number of vertices: {len(vertices_0)}")
        print(f"Number of vertices: {len(vertices_00)}")
        print(f"Number of faces: {len(faces)}")
        print(f"Number of vertices: {os.path.join(self.file_path, self.txt_vertices_0)}")
        print(f"Number of vertices: {os.path.join(self.file_path, self.txt_vertices_old)}")
        
        self.inten={} 
        for port in self.pre_portions:
            self.inten[f'{port}_body']=np.zeros_like(vertices_00[:,0]) 
            mmm=os.path.join(smod_path,f'intensity_{port}_segm.txt') 
            if os.path.exists(mmm):  
                self.inten[f'{port}_body']=mmm

'''



    # def update_output(mode,neldd, clusts,   metric, intensity_type, width, height, radius_level, radius_level_max,uoi_per,templ):
    # def Get_output(self, path_head,model_suf,path,mode,neldd, clusts,    intensity_type, width, height ,templ,nbin,index=None,get_return=True,hide_button_tf=True): 
    def Get_output(self, path_head,model_suf,path,mode,  width, height ,templ,nbin,index=None,get_return=True,hide_button_tf=True): 
        neldd, clusts,    intensity_type=None,None,None
        path_train=self.path_train
        true_keys='true_0'
        # model_suf=self.model_sufix if path_head == 'pinn' else 'save'  
        id_path=f'{true_keys}_save_save' if path_head=='true' else f'{path_head}_{model_suf}_{path}'
        id_pathss=f'{path_head}_{model_suf}_{path}_{true_keys}'
        self.get_data(
                    neld_data=self.neld_data,
                    model_sufix=model_suf,
                    data_studied=self.data_studied,
                    index=index,
                    ) 
        print('---------',true_keys,path,mode,neldd,path_head,'---inte',intensity_type,'===',model_suf,self.model_sufix)
        # print('---------',path_train['dest_smod_path'],self.path_file_sub[intensity_type][id_path] )
        # self.smod_path=smod_path=self.path_file_sub[self.inten_file_sub[0]][id_path] 
        neld_name=self.neld_name 
        bcouleur=self.bcouleur
        fsize=self.fsize 
        ppp= path.split('_')
        
        path_init=f'{ppp[0]}'
        ppp=ppp[1:]
        for pp in ppp[:-1]:
            path_init=f'{path_init}_{pp}'
        print('0000000',path_init,ppp)
        if templ=="plotly_dark":
            fcouleur='white'
            ffcouleur='black'#'black'
        else:
            fcouleur='black'
            ffcouleur='white'
        akp=aka_plot(tcouleur=templ,
                    bcouleur=bcouleur,
                    fcouleur=fcouleur,
                    fsize=fsize)
 
        vertices_pl= self.vertices_0 if neldd=='smooth' else self.vertices_00 

        intensity=None;intensity_path=''
        if intensity_type in self.inten_file_sub:  
            intensity_path =self.path_file_sub[intensity_type][id_path]
            # intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # intensity=np.loadtxt(intensity_path, dtype=float)
            # print('smod--------',intensity_path)
            if not os.path.exists(intensity_path):
                print('[[[[[Fail intensity_path]]]]]--------',intensity_path)

        elif intensity_type in self.inten_file:
            intensity_path =self.path_file_sub[intensity_type][id_path] 
            intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            if not os.path.exists(intensity_path):
                print('[[[[[Fail intensity_path]]]]]--------',intensity_path)
        elif intensity_type in self.inten_pca:
            intensity_path =self.path_file_sub[intensity_type][id_path]
            intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # intensity=np.loadtxt(intensity_path, dtype=float)
            # print('smod--------',intensity_path) 
        elif intensity_type in self.inten_file_train:
            intensity_path =self.path_file_sub[intensity_type][id_path]#self.annot_intensity[intensity_type][id_path]#
            if not os.path.exists(intensity_path):
                print('[[[[[Fail intensity_path]]]]]--------',intensity_path) 
            # intensity=np.loadtxt(intensity_path, dtype=float)
        elif intensity_type in self.intensity_logit:
            intensity_path =self.path_file_sub[intensity_type][id_path]#self.annot_intensity[intensity_type][id_path]#
            if not os.path.exists(intensity_path):
                print('[[[[[Fail intensity_path]]]]]--------',intensity_path)
        elif intensity_type in self.intensity_head_neck_logit:
            intensity_path =self.path_file_sub[intensity_type][id_path]#self.annot_intensity[intensity_type][id_path]#
            if not os.path.exists(intensity_path):
                print('[[[[[Fail intensity_path]]]]]--------',intensity_path)

        elif intensity_type in self.inten_file_model_head_neck:
            intensity_path =self.path_file_sub[intensity_type][id_path]
            intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # print('smod--------',intensity_path) 
            # intensity=np.loadtxt(intensity_path, dtype=float)


        elif intensity_type in self.base_features_dict.keys():
            intensity_path =self.path_file_sub[intensity_type][id_path]
            # intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # intensity=np.loadtxt(intensity_path, dtype=float)
 
        intensity= None 
        if os.path.exists(intensity_path):
            intensity=np.loadtxt(intensity_path, dtype=float)
            print('smod--------',intensity_path)
        else:
            print('Intensity DOESNT EXIST --------',intensity_path)
                # intensity_path =self.path_file_sub[intensity_type]['true_save_save'] 
            # elif intensity_type in ['smod_body','head_neck_body']: 
            #     intensity_path=self.inten[intensity_type] 
            # else:
            #     intensity_path = self.neld_mapping[neldd]["intensity"][intensity_type] 

        if intensity_type in ["gauss_curv_init","mean_curv_init"]:
            vertices_pl=self.vertices_00
        elif intensity_type in ["gauss_curv_smooth","mean_curv_smooth"]:
            vertices_pl=self.vertices_0


        figure=go.Figure()
        # self.get_figure(true_keys,path_head,model_suf, path,vertices_pl,intensity, clusts, width, height,ffcouleur=ffcouleur  )
 
    
        self.layout = go.Layout(width=width, 
                        height=height,
                        title=f'NELD', 
                        ) 

        self.scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            zaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            bgcolor=ffcouleur 
        )

        if mode=='algorithm':
            figure=  self.figure_3d 


        elif mode=='comparison':
            figure=akp.Plotly_Figure(data=self.scatter, layout=self.layout)
            figure.update_layout(scene=self.scene)  
        elif mode=='skeleton':
            if self.plot_data_center_curv is not None:  
                scatterr=self.plot_data_center_curv[id_path][clusts][0:1]
                skl_path=os.path.join(self.file_path_feat, self.txt_skl_vertices)
                if os.path.exists(skl_path): 
                    scatterr.append(hf.plotly_scatter(points=np.loadtxt(skl_path,dtype=float), color='yellow', size=5.3, name='skeleton smooth.',opacity=0.5))
                skl_path=os.path.join(self.file_path_feat, self.txt_skl_vertices_org)
                if os.path.exists(skl_path): 
                    scatterr.append(hf.plotly_scatter(points=np.loadtxt(skl_path,dtype=float), color='green', size=5.3, name='skeleton init.',opacity=0.5))
                for val in self.plot_data_center_curv[id_path][clusts][1:]:
                    scatterr.append(val)
                figure=akp.Plotly_Figure(data= scatterr, layout=self.layout)
                figure.update_layout(scene=self.scene)   
        elif mode in ['heatmap_cylinder','heatmap_cylinder_surface']:
            if self.plot_data_cylinder_heatmap is not None: 
                *pathc, last = path.split('_') 
                if mode =='heatmap_cylinder':
                    data=[self.plot_data_cylinder_heatmap[id_path].density_heatmap,self.plot_data_cylinder_heatmap[id_path].density_heatmap_points,self.plot_data_cylinder_heatmap[id_path].density_heatmap_points_org]
                    figure=akp.Plotly_Figure(data= data, layout=self.layout) 
                    figure.update_layout(
                        xaxis=dict(showgrid=False),  
                        yaxis=dict(showgrid=False)  
                    ) 
                elif mode =='heatmap_cylinder_surface':
                    data=[self.plot_data_cylinder_heatmap[id_path].density_heatmap_surface]
                    data.extend(self.plot_data_cylinder_heatmap[id_path].density_org_points)
                    figure=akp.Plotly_Figure(data= data, layout=self.layout)
                figure.update_layout(scene=self.scene)
        elif mode=='IOU':
            # if self.plot_data_iou  is not None: 
            #     figure=akp.Plotly_Figure(data= self.plot_data_iou, layout=self.layout)
            #     figure.update_layout(scene=self.scene)  
            if  len(self.plot_data_iou_dic)>0:  

                figure=akp.Plotly_Figure(data= self.plot_data_iou_dic[id_pathss] , layout=self.layout)
                figure.update_layout(scene=self.scene) 




        elif mode =='INV_MEASURE':
            ii=nbin
            import pickle
            from beinn.neld_pinn_0.help_fun import aka_fun
            from beinn.neld_pinn_0.Graph import mygraph

            path_gen=self.obj_org_path
            path_init_param=os.path.join(path_gen,f"param.pkl")
            with open(path_init_param, "rb") as f:
                pm = pickle.load(f)
            path_1=self.neld_path_org_new
            time=np.loadtxt(os.path.join(path_1,f"time.txt"),dtype=float)
            qq=np.loadtxt(os.path.join(path_1,f"qq_{ii}.txt"),dtype=float)

            Ntime= pm.N
            Ndim = pm.dim
            Nperiod = pm.Nperiod-1
            xint = 0
            yint = xint+pm.nPart
            Q1,Q2 =[ aka_fun().vec_to_mat_(qq,x,Ntime,Nperiod) for x in [xint,yint]]




            # tcouleur = 'plotly_dark'
            # bcouleur = 'navy'
            # fcouleur = 'white'
            # fsize = 20


            bheight = 600
            bwidth = 600 

            mygraph_ = mygraph(tcouleur=templ,
                            bcouleur=bcouleur,
                            fcouleur=fcouleur,
                            fsize=fsize)

            figure=mygraph_.plot_history_matrixxy(Q1,Q2,time.reshape(-1,1),bheight,bwidth)





        elif mode=='logit':
            print('[[[[[[[[[[[--------------------------logit]]]]]]]]]]]')
            dic={
                v:{
                    n:{
                        None
                    } for n in ['initial','final']
                } for v in ['inference','true']
            }
            self.hist_slider['min']=-1
            self.hist_slider['max']=10
            self.hist_slider['step']=1
            self.hist_slider['value']=10
            self.hist_slider['marks']={i: f'{i}' for i in range(-1, 10, 2)}
            rhs_dim=self.train_neld_param['rhs_dim']
            path_1=self.neld_path_org_new
            path_init_param=os.path.join(self.obj_org_path,f"param.pkl")
            import pickle
            with open(path_init_param, "rb") as f:
                pm = pickle.load(f) 
            time=np.loadtxt(os.path.join(path_1,f"time.txt"),dtype=float)
            nnn=time.shape[0]
            scatterr=[] 

            ii=nbin
            nn=nnn
            ntim,mtim=1+nn*ii-1,nn*(ii+1)
            print('[[[[nn]]]]',ii,nn,[ntim,mtim]) 

            ku = 'inference'
            tyy = self.intensity_logit_dict[ku][0]
            cpath=self.path_file_sub[tyy][id_path]
            if os.path.exists(cpath): 
                snn=np.loadtxt(cpath, dtype=float)
                for ii in range(rhs_dim['i'],rhs_dim['j']+1):
                    ntimi,mtimi=1+pm.nPart*pm.dimm*ii-1,pm.nPart*pm.dimm*(ii+1)
                    skl_path = np.sum(snn[:,ntimi:mtimi],axis=1)
                    print('[[[]]]',[skl_path.shape   ])
                    points = np.column_stack((time,skl_path))  
                    scatterr.append(
                        hf.plotly_scatter(
                            points=points,
                            color=ccoll[ii] ,
                            size=14.3,
                            symbol="circle" ,
                            name=f'Inference-period-{ii+1}',
                            opacity=0.2
                        ) 
                    )

            # -------- true --------
            ku = 'true'
            tyy = self.intensity_logit_dict[ku][0]
            cpath=self.path_file_sub[tyy][id_path]
            if os.path.exists(cpath):
                snn=np.loadtxt(cpath, dtype=float)
                for ii in range(rhs_dim['i'],rhs_dim['j']+1):
                    ntimi,mtimi=pm.nPart*pm.dimm*ii,pm.nPart*pm.dimm*(ii+1)
                    skl_path = np.sum(snn[:,ntimi:mtimi],axis=1)

                    points = np.column_stack((time,skl_path))  
                    scatterr.append(
                        hf.plotly_scatter(
                            points=points,
                            color=ccoll[ii] ,
                            size=8.3,
                            symbol="cross"  ,
                            name=f'True-period-{ii+1}',
                            opacity=0.4
                        ) 
                    )

            figure=akp.Plotly_Figure(data= scatterr , layout=self.layout)
            figure.update_layout(scene=self.scene) 

            figure.update_yaxes(title_text="BE" )
            figure.update_xaxes(title_text="Time", ) 
  











            '''

        elif mode=='logit':
            print('[[[[[[[[[[[--------------------------logit]]]]]]]]]]]')
            dic={
                v:{
                    n:{
                        None
                    } for n in ['initial','final']
                } for v in ['inference','true']
            }
            self.hist_slider['min']=-1
            self.hist_slider['max']=10
            self.hist_slider['step']=1
            self.hist_slider['value']=10
            self.hist_slider['marks']={i: f'{i}' for i in range(-1, 10, 2)}

            path_1=self.neld_path_org_new
            path_init_param=os.path.join(self.obj_org_path,f"param.pkl")
            import pickle
            with open(path_init_param, "rb") as f:
                pm = pickle.load(f) 
            time=np.loadtxt(os.path.join(path_1,f"time.txt"),dtype=float)
            nnn=time.shape[0]
            scatterr=[] 

            ii=nbin
            nn=nnn
            ntim,mtim=1+nn*ii-1,nn*(ii+1)
            print('[[[[nn]]]]',ii,nn,[ntim,mtim])
            subplot_titles = ("Initial", "Final")
            figure = akp.Plotly_Figure_Sub(
                subplot_titles,
                rows=1,
                cols=2,
                shared_xaxes=False,
                shared_yaxes=False
            )
            row=1
            for col, kk in enumerate(['initial', 'final'], start=1):

                # -------- inference --------
                ku = 'inference'
                tyy = self.intensity_logit_dict[ku][col-1]
                cpath=self.path_file_sub[tyy][id_path]
                if os.path.exists(cpath): 
                    snn=np.loadtxt(cpath, dtype=float)
                    for ii in range(pm.Nperiod-1):
                        ntimi,mtimi=1+pm.nPart*pm.dimm*ii-1,pm.nPart*pm.dimm*(ii+1)
                        skl_path = np.sum(snn[:,ntimi:mtimi],axis=1)
                        print('[[[]]]',[skl_path.shape  if kk == 'initial' else skl_path[ntim:mtim].shape])
                        points = np.column_stack((skl_path, time)) if kk == 'initial' else np.column_stack((skl_path[ntim:mtim], time))
                        figure.add_trace(
                            hf.plotly_scatter(
                                points=points,
                                color='red' if kk == 'initial' else 'blue',
                                size=5.3,
                                symbol="circle" if ku == "true" else "cross",
                                name=f'Inference-{kk}--period-{ii+1}',
                                opacity=0.5
                            ),
                            row=row, col=col
                        )

                # -------- true --------
                ku = 'true'
                tyy = self.intensity_logit_dict[ku][col-1]
                cpath=self.path_file_sub[tyy][id_path]
                if os.path.exists(cpath):
                    snn=np.loadtxt(cpath, dtype=float)
                    for ii in range(pm.Nperiod-1):
                        ntimi,mtimi=1+pm.nPart*pm.dimm*ii-1,pm.nPart*pm.dimm*(ii+1)
                        skl_path = np.sum(snn[:,ntimi:mtimi],axis=1)

                        points = np.column_stack((skl_path, time)) if kk == 'initial' else np.column_stack((skl_path[ntim:mtim], time))
                        figure.add_trace(
                            hf.plotly_scatter(
                                points=points,
                                color='green' if kk == 'initial' else 'purple',
                                size=5.3,
                                symbol="circle" if ku == "true" else "cross",
                                name=f'True-{kk}--period-{ii+1}',
                                opacity=0.5
                            ),
                            row=row, col=col
                        )

            figure.update_xaxes(title_text="BE", row=1, col=2)
            figure.update_yaxes(title_text="Time", row=1, col=1)
            figure.update_yaxes(title_text="Time", row=1, col=2)

            figure.update_layout(
                legend=dict(groupclick="toggleitem")
            )
            figure.update_layout(self.layout)
            figure.update_layout(scene=self.scene)






'''









        elif mode=='accuracy':
            title='Accuracy'
            width=width
            height=height
            subplot_titles=('IOU Single','IOU Union')
            rows=2
            listt=['single','union',]
            if 'single_dice' in self.metric_total_dic:
                subplot_titles=('IOU Single','IOU Union','DICE Single','DICE Union')
                rows=4
                listt=['single','union','single_dice','union_dice']

            figure=akp.Plotly_Figure_Sub( subplot_titles,rows=rows, cols=1, 
                                            shared_xaxes=False,
                                            shared_yaxes=False)
            if len(self.metric_total_dic )>0: 
                for ii,typ in enumerate(listt):
                    # df_union=pd.read_csv(os.path.join(self.path_file[f'result_appr'],f'metric_{typ}.csv')).sort_values(by=f'single_{true_keys}', ascending=True)
                    df_union=pd.read_csv(os.path.join(self.path_file[f'result_appr'],f'metric_{typ}_{true_keys}.csv')).sort_values(by=f'accuracy', ascending=True)
                    xlabels=df_union.columns[1:]
                    ylabels=df_union.iloc[:,0]

                    df_union=df_union[df_union.columns[1:]]
                    cm=np.array(df_union.select_dtypes(include=[np.number]))
                    cm= np.round(cm*1e3)/1e3

                    scatter=go.Heatmap(
                                        z=cm,
                                        x=xlabels,  
                                        y=ylabels,  
                                        colorscale='Blues',
                                        text=cm,  # Show numbers in cells
                                        texttemplate="%{text}",  # Format as numbers
                                        # hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                                        showscale=False ,
                                        textfont=dict(size=18),
                                    )  
                    figure.add_trace( scatter, row=ii+1, col=1) 
            figure.update_layout(height=2.5*height, width=width)

            figure.update_layout(
                xaxis=dict(
                    side="top",          
                    ticks="outside",     # Optional: ticks outside the plot
                    showticklabels=True, # Ensure tick labels are visible 
                    tickmode="array",
                    tickvals=['accuracy','precision','recall','f1_score','AUC_smod','AUC_hmod'],
                    ticktext=['Accuracy','Precision','Recall','F1 Score','AUC smod','AUC hmod'],
                )
            )

            figure.update_layout(
                xaxis2=dict(
                    side="top",          
                    ticks="outside",     # Optional: ticks outside the plot
                    showticklabels=True, # Ensure tick labels are visible 
                    tickmode="array",
                    tickvals=['accuracy','precision','recall','f1_score','AUC_smod','AUC_hmod'],
                    ticktext=['Accuracy','Precision','Recall','F1 Score','AUC smod','AUC hmod'],
                )
            )
            figure.update_layout(
                xaxis3=dict(
                    side="top",          
                    ticks="outside",     # Optional: ticks outside the plot
                    showticklabels=True, # Ensure tick labels are visible 
                    tickmode="array",
                    tickvals=['accuracy','precision','recall','f1_score','AUC_smod','AUC_hmod'],
                    ticktext=['Accuracy','Precision','Recall','F1 Score','AUC smod','AUC hmod'],
                )
            )
            figure.update_layout(
                xaxis4=dict(
                    side="top",          
                    ticks="outside",    
                    showticklabels=True, 
                    tickmode="array",
                    tickvals=['accuracy','precision','recall','f1_score','AUC_smod','AUC_hmod'],
                    ticktext=['Accuracy','Precision','Recall','F1 Score','AUC smod','AUC hmod'],
                )
            )
            for annotation in figure['layout']['annotations']:
                annotation['y'] += 0.045








        elif mode in self.inten_file_model_head_neck:  
            self.layout = go.Layout(width=width, 
                            height=height,  )
            # ['model_hn_loss','model_hn_iou']
            if (mode == 'model_hn_loss' ) and id_path in self.scatter_loss_dic and len( self.scatter_loss_dic[id_path])>0: 
                figure=akp.Plotly_Figure(data=self.scatter_loss_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene) 
            elif (mode == 'model_hn_iou' ) and id_path in self.scatter_iou_dic and len(self.scatter_iou_dic[id_path])>0: 
                figure=akp.Plotly_Figure(data= self.scatter_iou_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene) 

            if (mode == 'model_sp_loss' ) and id_path in self.scatter_loss_smod_dic and len( self.scatter_loss_smod_dic[id_path])>0: 
                figure=akp.Plotly_Figure(data=self.scatter_loss_smod_dic[id_path], layout=self.layout)
                figure.update_layout(yaxis_type="log")
                figure.update_layout(scene=self.scene) 
            elif (mode == 'model_sp_iou' ) and id_path in self.scatter_iou_smod_dic and len(self.scatter_iou_smod_dic[id_path])>0: 
                print('----=====----','9088==============---========================================',len(self.scatter_iou_smod_dic[id_path]))
                figure=akp.Plotly_Figure(data= self.scatter_iou_smod_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene)
            elif (mode == 'model_sp_auc' ) and id_path in self.scatter_auc_smod_dic and len(self.scatter_auc_smod_dic[id_path])>0: 
                print('----=====----','model_sp_auc==============---========================================',len(self.scatter_auc_smod_dic[id_path]))
                figure=akp.Plotly_Figure(data= self.scatter_auc_smod_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene)
            elif (mode == 'model_sp_dice' ) and id_path in self.scatter_dice_smod_dic and len(self.scatter_dice_smod_dic[id_path])>0: 
                print('----=====----','model_sp_auc==============---========================================',len(self.scatter_dice_smod_dic[id_path]))
                figure=akp.Plotly_Figure(data= self.scatter_dice_smod_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene)
                          
            if path_head.startswith(('cML','CML','ML')): 
                figure.update_layout(
                    xaxis=dict(
                        side="bottom",  # keep the bottom axis if you want
                        showticklabels=False  # hide bottom labels if you only want top
                    ),
                    xaxis2=dict(
                        side="top",
                        overlaying="x",      
                        ticks="outside",
                        showticklabels=True,
                        tickmode="array",
                        tickvals=[0, 1, 2, 3],  
                        ticktext=['Accuracy','Precision','Recall','F1 Score'],
                    )
                )

                # Make sure your trace uses the top axis
                for trace in figure.data:
                    trace.update(xaxis="x2")

                                

        elif mode == 'model_shap':
            # figure=self.model_shap_dic[id_path]
            # print('figure',figure)
            figure=akp.Plotly_Figure(data= self.model_shap_dic[id_path]['data'], layout=self.model_shap_dic[id_path]['layout'])
            figure.update_layout(scene=self.scene) 


        elif mode =='roc_curve':  
            subplot_titles=('ROC smod','ROC hmod')
            figure = akp.Plotly_Figure_Sub(
                subplot_titles,
                rows=2, cols=1,
                shared_xaxes=False,
                shared_yaxes=False
            )

            if len(self.metric_total_dic['roc_curve']) > 0:
                for ii, typ in enumerate(['smod','hmod']):
                    traces = self.metric_total_dic['roc_curve']['curve'][typ]

                    figure.add_traces(
                        traces,
                        rows=[ii+1] * len(traces),
                        cols=[1] * len(traces)
                    )

                    figure.add_trace(
                        go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode="lines",
                            name="",
                            line=dict(color="gray", dash="dash")
                        ),
                        row=ii+1, col=1
                    )

                figure.update_layout(scene=self.scene)

                figure.update_layout(
                    title="ROC Curves",
                    xaxis=dict(title="False Positive Rate"),
                    xaxis2=dict(title="False Positive Rate"),
                    yaxis=dict(title="True Positive Rate"),
                    yaxis2=dict(title="True Positive Rate"),
                    barmode="overlay"
                )



        elif mode in ['heatmap_iou','heatmap_iou_union','histogram_iou','roc_curve']: 
            smod_path_save=     self.path_file[f'result_{id_path}']
            metric_path=os.path.join( smod_path_save,f'iou_{true_keys}.csv') 
            if os.path.exists(metric_path):
                df = pd.read_csv(metric_path) 
                sze_checks_0 ,sze_check ,sze_check_un=df['id'],df['iou_single'],df['iou_union']
                # nhf=df['id_true']>0
                # sze_checks_0 ,sze_check ,sze_check_un=sze_checks_0[nhf] ,sze_check[nhf] ,sze_check_un[nhf]
                metric_name_path=df.columns
                iou_dict={}
                get_cm_iou(sze_checks_0 ,sze_check ,sze_check_un ,iou_dict=iou_dict, iou_per=70,labels = ['False', 'True'],nbinsx=nbin,)
                if mode =='heatmap_iou':
                    accuracy, precision, recall, f1_score=iou_dict['single']['metrics'].values()
                    metrics_text = (f'Accuracy: {accuracy:.3f}   '
                                    f'Precision: {precision:.3f}<br>'
                                    f'Recall    : {recall:.3f}   '
                                    f'F1 Score: {f1_score:.3f}')
                    figure=akp.Plotly_Figure(data=iou_dict['single']['heatmap_cm'], layout=self.layout)
                    figure.update_layout(scene=self.scene)

                    figure.update_layout(
                        title={
                            'text': metrics_text,
                            'x': 0.5,
                            'y': 0.92,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        }
                    )
                    figure.update_layout(
                        xaxis=dict(
                            # title=dict(text="Iterations", font=dict(size=16)),
                            tickmode="array",
                            tickvals=['accuracy','precision','recall','f1_score'],
                            ticktext=['Accuracy','Precision','Recall','F1 Score'],
                            # tickfont=dict(size=14, family="Arial", color="black")
                        ),
                        # yaxis=dict(
                        #     title=dict(text="IoU", font=dict(size=16)),
                        #     tickmode="linear",
                        #     dtick=0.1,  # step size
                        #     tickfont=dict(size=14, family="Arial", color="black")
                        # )
                    )
                elif mode =='heatmap_iou_union':
                    accuracy, precision, recall, f1_score=iou_dict['union']['metrics'].values()
                    metrics_text = (f'Accuracy: {accuracy:.3f}   '
                                    f'Precision: {precision:.3f}<br>'
                                    f'Recall    : {recall:.3f}   '
                                    f'F1 Score: {f1_score:.3f}')
                    figure=akp.Plotly_Figure(data=iou_dict['union']['heatmap_cm'], layout=self.layout)
                    figure.update_layout(scene=self.scene)

                    figure.update_layout(
                        title={
                            'text': metrics_text,
                            'x': 0.5,
                            'y': 0.92,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        } 
                    )
                    fig.update_layout(
                        xaxis=dict(
                            # title=dict(text="Iterations", font=dict(size=16)),
                            tickmode="array",
                            tickvals=['accuracy','precision','recall','f1_score'],
                            ticktext=['Accuracy','Precision','Recall','F1 Score'],
                            # tickfont=dict(size=14, family="Arial", color="black")
                        ), 
                    )
                elif mode =='histogram_iou':

                    figure=akp.Plotly_Figure(data= iou_dict['single']['histogram'] , layout=self.layout)
                    figure.add_annotation(
                        text=f'Total Count: {len(sze_checks_0)}',
                        xref='paper',
                        yref='paper',
                        x=0.98,
                        y=0.95,
                        showarrow=False,
                        font=dict(size=22)
                    )
                    figure.update_layout(scene=self.scene)  

                    figure.add_trace(iou_dict['union']['histogram'] )
                    figure.add_annotation(
                        text=f'Total Count Union.: {len(sze_checks_0)}',
                        xref='paper',
                        yref='paper',    
                        x=0.98,
                        y=0.88,
                        showarrow=False,
                        font=dict(size=22)
                    )

                    figure.update_layout(
                        barmode='overlay',  
                        title=mode,
                        xaxis_title='Values',
                        yaxis_title='Iou'
                    )



                    # figure.add_annotation(
                    #     text=f'Total Count Union.: {len(sze_checks_0)}',
                    #     xref='paper',
                    #     yref='paper',    
                    #     x=0.98,
                    #     y=0.88,
                    #     showarrow=False,
                    #     font=dict(size=22)
                    # )

                    # figure.update_layout(
                    #     barmode='overlay',  
                    #     title=mode,
                    #     xaxis_title='Values',
                    #     yaxis_title='Iou'
                    # )



        elif mode in self.metric_mapping_combine['name']:  
            neld_names=self.neld_names 
            smod_path_save=     self.path_file[f'result_{id_path}']
            metric_path=os.path.join( smod_path_save,'metrics.csv') 
            if os.path.exists(metric_path):  
                df = pd.read_csv(metric_path)   
                
                headd,neckk,lengthh=df[self.metrics_combine[mode]['key'][0]],df[self.metrics_combine[mode]['key'][1]],df[self.metrics_combine[mode]['key'][2]]
                fig=hp.plotly_metric(headd,neckk,lengthh,height=height,width=width,title_size=22,colorscale='Blues',
                                         xtitle=add_unit(self.metrics_combine[mode]['label'][0]),
                                         ytitle=add_unit(self.metrics_combine[mode]['label'][1]),
                                         ztitle=add_unit(self.metrics_combine[mode]['label'][2]),
                                         marginal='box')
                figure=akp.Plotly_Figure(data= fig.data, layout=fig.layout) 
                # if mode=='vol_area_length_smod':
                m, b = np.polyfit(headd, neckk, 1) 
                y_pred = m * headd + b 
                y_mean = np.mean(neckk)
                ss_res = np.sum((neckk - y_pred)**2)
                ss_tot = np.sum((neckk - y_mean)**2)
                r2 = 1 - (ss_res / ss_tot)
                n=len(neckk) 
                reg_x = np.linspace(headd.min(), headd.max()*1.2, 100)
                reg_y = m * reg_x + b 
                # reg_y = np.clip(reg_y, neckk.min(), neckk.max()*1.2) 
                figure.add_trace(go.Scatter(x=reg_x, y=reg_y, mode="lines", name="Linear fit"))
                equation_text = f"y = {m:.3f}x + {b:.3f}<br>R<sup>2</sup> = {r2:.3f}<br>n = {n}" 
                # equation_text = ( 
                #     f"y   = {m:.3f}x + {b:.3f}\n"
                #     f"R<sup>2</sup> = {r2:.3f}\n"
                #     f"n   = {n}" 
                # ) 
                figure.add_annotation(
                    x=headd.min() + 0.01*(headd.max() - headd.min()),  
                    y=neckk.min() + 0.9*(neckk.max() - neckk.min()),   
                    text=equation_text,
                    showarrow=False,
                    font=dict(  size=20, color="white"),
                    bgcolor="black",
                    bordercolor="black", 
                    xanchor="left",    
                    align="left"   ,
                )
  
            #     figure.update_layout(
            #     # title=title,
            #     xaxis2_title='Counts',#self.metrics_combine[mode]['label'][0],
            #     yaxis2_title='Counts',self.metrics_combine[mode]['label'][1], 
            # )                
                figure.update_layout(
                                xaxis1=dict(
                                    title=dict(
                                        text=add_unit(self.metrics_combine[mode]['label'][0]),
                                        font=dict(size=20)
                                    ),
                                    # range=[-3, 3],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),
                                yaxis1=dict(
                                    title=dict(
                                        text=add_unit(self.metrics_combine[mode]['label'][1]),
                                        font=dict(size=20),
                                    ),
                                    # range=[-5, 5],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),  

                                
                                xaxis2=dict(
                                    title=dict(
                                        text="Counts",
                                        font=dict(size=20)
                                    ),
                                    # range=[-3, 3],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),
                                yaxis2=dict(
                                    title=dict(
                                        text="Counts",
                                        font=dict(size=20),
                                    ),
                                    # range=[-5, 5],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),  



                                
                                xaxis3=dict(
                                    title=dict(
                                        text="x3",
                                        font=dict(size=20)
                                    ),
                                    range=[-3, 3],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),
                                yaxis3=dict(
                                    title=dict(
                                        text="y3",
                                        font=dict(size=20),
                                    ),
                                    range=[-5, 5],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),  
            )

 
        elif mode in self.metric_mapping['name']:  
            neld_names=self.neld_names 
            smod_path_save=     self.path_file[f'result_{id_path}']
            metric_path=os.path.join( smod_path_save,'metrics.csv') 
            # figure=akp.Plotly_Figure(data=[],layout=None)
            # figure.update_layout(scene=self.scene)  

            scolor=['blue','red','yellow','purple','green']
            data_path=self.path_file['result_true']

            df_save={}
            name=name_approx='Approx'
            df_save[name]={}
            xmax=0
            xmin=-np.inf
            df_save[name]['ann']=['blue',.6,0.98,0.95]
            df_save[name]['path']=metric_path
            if os.path.exists(df_save[name]['path']):  
                df_save[name]['df']=df = pd.read_csv(df_save[name]['path'])   
                # df=df[df[df.columns[0]].astype(str).str.startswith(tuple(neld_names))] 
                df_save[name]['df']=  df
                if mode in df.columns:  
                    xmax=max(xmax,max(df[mode]))
                    xmin=min(xmin,min(df[mode]))
            namesan=[]
            names=[]
            for ii in range(1,5):
                data_name=f'smod_head_analysis.trial_{ii}.dat' 
                df_save[name]['ann']=[scolor[ii],.5,0.98,0.95-(ii*0.07)/1.3] 
                if os.path.exists( os.path.join(data_path,data_name)):
                    name=f'Annot_{ii}'
                    names.append(name)
                    namesan.append(name)
                    df_save[name]={}
                    df_save[name]['path']=hff.get_conversion_file(data_path=data_path,data_name=data_name)
                    df = pd.read_csv(df_save[name]['path'])  
                    # df=df[df[df.columns[0]].astype(str).str.startswith(tuple(neld_names))] 
                    df_save[name]['df']=  df
                    if mode in df.columns:  
                        xmax=max(xmax,max(df[mode]))
                        xmin=min(xmin,min(df[mode]))
            names.append(name_approx)
            title='KL Divergence D_KL(P||Q)'
            xaxis_title,yaxis_title='P','Q'


            if len(namesan)>0 :
                subplot_titles=('Histogram','KL Divergence D_KL(P||Q)')
                figure=akp.Plotly_Figure_Sub( subplot_titles,rows=2, cols=1, )
            else:
                figure= akp.Plotly_Figure(data=[],layout=None)

            for name in names:
                df = df_save[name]['df']
                color,opacity,x_ann,y_ann=df_save[name]['ann']  
                if os.path.exists(df_save[name]['path']):  
                    if mode in df.columns:  
                        scatter_metric,layout_metric=hp.Plotly_histogram_return(data=np.abs(df[mode]),
                                                                                nbinsx=nbin,
                                                                                title=mode,
                                                                                xtitle='Length',
                                                                                ytitle='Count',
                                                                                xrange=[xmin,xmax],
                                                                                yrange=None,
                                                                                width=width, 
                                                                                height=height,
                                                                                opacity=opacity,
                                                                                color=color,
                                                                                name=name,)
                        if len(namesan)>0 :
                            figure.add_trace( scatter_metric, row=1, col=1)
                        else:
                            figure.add_trace( scatter_metric )
                        cname=f'Count {name}  ' if name=='Approx' else f'Count {name}' 
                        figure.add_annotation(
                            text=f'{cname}: {len(df[mode])}',
                            xref='paper',
                            yref='paper',    
                            x=x_ann,
                            y=y_ann,
                            showarrow=False,
                            font=dict(size=22)
                        )  

                    figure.update_layout(layout_metric)
                        
                    if len(namesan)>0 :
                        scatter,layout=compute_kl(df_save,names,mode,width,height,bins =nbin)
                        figure.add_trace( scatter, row=2, col=1)
                        figure.update_layout(layout )
                        figure.update_layout(
                        # title=title,
                        xaxis2_title=xaxis_title,
                        yaxis2_title=yaxis_title,
                        # xaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels),
                        # yaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels), 
                        )
                        figure.update_layout(height=1.3*height, width=width)
                    else:
                        figure.update_layout(height=height, width=width)

 

        else:
            figure=go.Figure() 
        self.figure=figure
        if mode not in ['INV_MEASURE',]:
            hp.hide_button(figure,hide_button_tf) 
        if get_return:
            return dcc.Graph(figure=figure) ,f'model Name: {neld_name}'
        else:
            self.figure=figure
            self.vertices_pl,self.intensity, self.clusts=vertices_pl,intensity, clusts
            self.akp=akp


 




 

