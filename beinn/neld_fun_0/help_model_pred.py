

import sys
import os
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import time 
import tensorflow as tf  
tf.config.run_functions_eagerly(True) 
from tensorflow.keras.models import load_model
DTYPE='float32' 
import pickle 
import pandas as pd
  
DTYPE='float32' 
DTYPE = tf.float32
  
from beinn.neld_fun_0.get_path import assign_if_none,get_name,get_param,get_files,remove_directory,safe_id
from beinn.neld_fun_0.side_bar import  get_text_dash_test,get_text_dash_dnn,get_text_dash_app,get_text_dash_main_2
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"  
from tqdm import tqdm 
  
# import torch
import beinn.neld_fun_0.help_neld_train_test as hntt
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

#get_files,get_name,

class model_pred( hntt.train_test_tf):
    def __init__(self, file_path_org,
                        data_studied, 
                        model_sufix,
                        neld_data, 
                        neld_names=None,
                        neld_namess=None,   
                        neld_path_inits=None,
                        name_smod_id=None,
                        name_head_id=None,
                        name_neck_id=None,
                        name_hmod_id=None,
                        path_train=None,
                        txt_true_file=None,
                        txt_save=False,
                        txt_save_pred=False,
                        size_threshold=100,
                        gauss_threshold=10, 
                        name_path_fin='save',
                        cts=6,
                        stoppage=4,
                        zoom_threshold=1000,
                        radius_threshold=0.05,
                        name_path_fin_save_index=20,
                        smod_filter=True,
                        numNeighbours=5,
                        zoom_threshold_min=1,
                        zoom_threshold_max=4, 
                        line_num_points_hmod=200,
                        line_num_points_inter_hmod=300, 
                        spline_smooth_hmod=1, 
                        DTYPE='float32', 
                        disp_infos=False,
                        pre_portion=None,
                        pinn_dir_data=None, 
                        pinn_dir_data_all=None,
                        model_sufix_all=None,
                        path_heads =None,
                        true_keys=None,
                        list_features=None,
                        base_features_list=None,
                        metrics={},
                        model_type=None,
                        data_mode=None,
                        path_dir=None, 
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        obj_org_path_dict=None,
                        path_display=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        model_sufix_dic=None,
                        path_display_dic=None,
                        kmean_n_run=100,
                        kmean_max_iter=600,
                        param_dic=None,

        ): 


        self.common_args = dict(
            file_path_org=file_path_org,
            neld_data=neld_data,
            neld_names=neld_names,
            neld_namess=neld_namess,
            data_studied=data_studied,
            name_smod_id=name_smod_id,
            name_head_id=name_head_id,
            name_neck_id=name_neck_id,
            name_hmod_id=name_hmod_id,
            txt_true_file=txt_true_file,
            txt_save=txt_save,
            txt_save_pred=txt_save_pred,
            size_threshold=size_threshold,
            gauss_threshold=gauss_threshold,
            name_path_fin=name_path_fin,
            cts=cts,
            stoppage=stoppage,
            zoom_threshold=zoom_threshold,
            radius_threshold=radius_threshold,
            name_path_fin_save_index=name_path_fin_save_index,
            smod_filter=smod_filter,
            numNeighbours=numNeighbours,
            zoom_threshold_min=zoom_threshold_min,
            zoom_threshold_max=zoom_threshold_max,
            line_num_points_hmod=line_num_points_hmod,
            line_num_points_inter_hmod=line_num_points_inter_hmod,
            spline_smooth_hmod=spline_smooth_hmod,
            DTYPE=DTYPE,
            model_sufix=model_sufix,
            neld_path_inits=neld_path_inits,
            disp_infos=disp_infos,
            path_train=path_train,
            pre_portion=pre_portion,
            pinn_dir_data=pinn_dir_data,
            pinn_dir_data_all=pinn_dir_data_all,
            model_sufix_all=model_sufix_all,
            path_heads=path_heads,
            true_keys=true_keys,
            list_features=list_features,
            base_features_list=base_features_list,
            metrics=metrics,
            model_type=model_type,
            data_mode=data_mode, 
            thre_target_number_of_triangles=thre_target_number_of_triangles,
            voxel_resolution=voxel_resolution,
            obj_org_path_dict=obj_org_path_dict,
            model_sufix_dic=model_sufix_dic,
            path_display_dic=path_display_dic,
            kmean_n_run=kmean_n_run,
            kmean_max_iter=kmean_max_iter,
            param_dic=param_dic,
        )


        get_name.__init__(self)  
        get_files.__init__(self,**self.common_args) 
        # train_test_tf.__init__(self,**self.common_args) 
        hntt.train_test_tf.__init__(self,**self.common_args)
        
        
        
        self.path_display_dic=path_display_dic
        self.dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh
        if data_mode is not None:
            path_train=data_mode['path_train']
            pre_portion=data_mode['pre_portion']
            # pinn_dir_data=data_mode['pinn_dir_data']
            # list_features=data_mode['list_features']
            # base_features_list=data_mode['base_features_list']
        self.path_display=path_display if path_display is not None else ['dest_smod_path','dest_hmod_path']
        path_dir=os.path.join(file_path_org, 'data') 
        np.savetxt(os.path.join(path_dir, 'model_sufix_all.txt'), np.array(list(model_sufix_all)), fmt='%s') 
        np.savetxt(os.path.join(path_dir, 'pinn_dir_data_all.txt'), np.array(list(pinn_dir_data_all)), fmt='%s')
        np.savetxt(os.path.join(path_dir, 'path_heads.txt'), np.array(path_heads), fmt='%s')
        np.savetxt(os.path.join(path_dir, 'true_keys.txt'), np.array(true_keys), fmt='%s') 
 


    def clean_path_dir(self,   
                        path_head_clean,
                    data_studied=None, 
                    disp_infos=None, 
                    neld_names=None,
                    neld_namess=None,
                    path_train=None,
                    ):   
        print(' path_head_clean ---- started')
        print('==========================================================')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        neld_namess = neld_namess or self.neld_namess
        neld_names = neld_names if neld_names is not None else self.neld_names 
        for  index,neld_name in enumerate(neld_names):
            self.get_neld_name(data_studied=data_studied,index=index )  
            self.clean_neld_name(path_head_clean=path_head_clean)


   
  
 
    def get_head_neck_segss(self,    
                            metrics=None,
                            seg_neld='full',
                            path_train=None ,
                            get_refine_=False,
                            data_studied=None, 
                            disp_infos=None, 
                            neld_names=None,
                            neld_namess=None,
                            smod_hmod_txt=True,
                            size_threshold=None,
                            line_num_points_hmod=None,
                            line_num_points_inter_hmod=None,
                            spline_smooth_hmod=None ,  
                            zoom_thre=15,
                            skip_first_n=1,
                            skip_end_n=52,
                            subdivision_thre=3, 
                            subsample_thre=.02,
                            f=0.99,
                            N=10,
                            num_chunks=100, 
                            line_num_points=300,
                            line_num_points_inter=700,
                            spline_smooth=1.,
                            num_points=50,
                            ctl_run_thre=1,
                            end_thre=40,
                            head_neck_path= 'dest_smod_path',
                            pre_portion=None,
                            dict_mesh_to_skeleton_finder_mesh=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic 
        from beinn.neld_fun_2.help_pinn_data_fun import pinn_data 
        time_start = time.time()
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        path_train=path_train or self.path_train
        pre_portion=pre_portion or self.pre_portion 
        size_threshold=size_threshold or self.size_threshold
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        neld_namess = neld_namess or self.neld_namess
        neld_names = neld_names if neld_names is not None else self.neld_names 
        line_num_points_hmod = line_num_points_hmod or self.line_num_points_hmod
        line_num_points_inter_hmod = line_num_points_inter_hmod or self.line_num_points_inter_hmod
        spline_smooth_hmod = spline_smooth_hmod or self.spline_smooth_hmod
        self.metrics=metrics if metrics is not None else self.metrics
        print('Head Neck Prediction started')
        print('--------------------------------') 
        print(f"learning length : {zoom_thre}") 
        print(f"skip_first_n    : {skip_first_n}") 
        print(f"skip_end_n      : {skip_end_n}") 
        print(f"subdivision_thre: {subdivision_thre}")
        print(f"destination     : {path_train['dest_smod_path']}")
        print(f"data path       : {path_train['data_smod_path']}") 
        print('--------------------------------')  


        metrics_dats=[]
        metrics_name=[]
        metrics_dat={}
        metrics_sp=[]
        gnam=get_name()
        metricss=gnam.metrics 

        for  index,neld_name in enumerate(neld_names):
            self.get_neld_name(data_studied=data_studied,index=index,
                                ** param_dic['data']['get_neld_name'] ) 
            cxc=param_dic['data']['get_neld_name']['dict_neld_path'] 
            file_path_feat=self.dict_neld['path'][cxc]['file_path_feat']
            file_path=self.dict_neld['path'][cxc]['file_path'] 
            neld_name=self.neld_name               
            gnam=get_name()
            metricss=gnam.metrics 
            metrics_name=[]
            pid=pinn_data(file_path=file_path,
                          file_path_feat=file_path_feat,
                            hmod_path=     self.path_file[path_train['dest_hmod_path']] ,
                            smod_path=     self.path_file[path_train['dest_smod_path']] , 
                            hmod_path_pre= self.path_file[path_train['data_hmod_path']] ,
                            smod_path_pre= self.path_file[path_train['data_smod_path']] , 
                            neld_path_original_m=self.neld_path_original_m, 
                            neld_first_name=self.neld_namess[index][1],  
                            path_file=self.path_file,
                            line_num_points_hmod=line_num_points_hmod,
                            line_num_points_inter_hmod=line_num_points_inter_hmod,
                            spline_smooth_hmod=spline_smooth_hmod, 
                                    ) 
            pid.get_head_neck_segss(  
                                pre_portion=pre_portion,
                                metrics=metricss,
                                seg_neld=seg_neld,
                                smod_hmod_txt=smod_hmod_txt, 
                                path_train=path_train,
                                get_refine_=get_refine_,
                                skip_first_n=skip_first_n,
                                skip_end_n=skip_end_n,
                                zoom_thre=zoom_thre,
                                subdivision_thre=subdivision_thre,
                                subsample_thre=subsample_thre,
                                f=f,
                                N=N,
                                num_chunks=num_chunks,
                                num_points=num_points, 
                                line_num_points= line_num_points,
                                line_num_points_inter= line_num_points_inter,
                                spline_smooth= spline_smooth,
                                ctl_run_thre=ctl_run_thre,  
                                size_threshold=size_threshold,
                                end_thre=end_thre,
                                head_neck_path=head_neck_path,
                                dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                        ) 
            for nam in self.metrics_keys:
                if len(metricss[nam].keys())>0:
                    metrics_name.append(nam)
                    if nam not in metrics_dat:
                        metrics_dat[nam]=[]
                    metrics_dat[nam].extend((list(metricss[nam].values())))
                    if nam==metrics_name[0]:
                        metrics_sp.extend(metricss[nam].keys())
            # print( (metrics_name),) 
            # print([len(metrics_dat[nam]) for nam in metrics_name],)
            metricss={}
        if len(metrics_dat)>0:
            metrics_dats=[metrics_dat[nam] for nam in metrics_dat.keys()]
            metrics_dats=np.array(metrics_dats).T 
            print('IM HEAD NECK',metrics_dats.shape,head_neck_path,path_train[head_neck_path])
            metrics_name=np.array(list(metrics_dat.keys()))
            metrics_sp=np.array(metrics_sp)
            path=path_train[head_neck_path] 
            smod_path_save=     self.path_file[f'result_{path}'] 
            with open(os.path.join( smod_path_save,'metrics.csv'), 'w') as f: 
                f.write(',' + ','.join(metrics_name) + '\n') 
                for i, row_name in enumerate(metrics_sp):
                    row_data = ','.join(map(str, metrics_dats[i]))
                    f.write(f'{row_name},{row_data}\n')


        metrics_dats=[]




        mytime0 = time.time() - time_start 
        hours, rem = divmod(mytime0, 3600)
        minutes, seconds = divmod(rem, 60)
        if disp_infos:
            print(f'Head Neck Prediction completed on {neld_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
        print('Head Neck Prediction completed')

 



    def get_rocb(self,   
                    path_train=None ,
                    data_studied=None, 
                    disp_infos=None, 
                    neld_names=None,
                    neld_namess=None,
                    model_type=None, 
                    zoom_thre=10,
                    iou_thre=0.002,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic  
        from sklearn.metrics import roc_curve, auc
        print('ROC started')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos or self.disp_infos    
        data_studied = data_studied or self.data_studied   
        neld_namess = neld_namess or self.neld_namess
        neld_names = neld_names or self.neld_names 
        model_type=model_type or self.model_type 
        # key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'
        # hjh=[self.path_file_sub[tyy][key]   for tyy in self.intensity_smods_logit]
        for head_neck_path in self.path_display:
            iou_dict={}
            y_true,score=[],[]
            for  index,neld_name in enumerate(neld_names):
                self.get_neld_name(data_studied=data_studied,index=index ,
                                ** param_dic['data']['get_neld_name'])
                '''
                cxc=param_dic['data']['get_neld_name']['dict_neld_path']
                file_path_feat=self.dict_neld['path'][cxc]['file_path_feat']
                file_path=self.dict_neld['path'][cxc]['file_path'] 
                neld_path_true_final=self.dict_neld['path'][cxc]['neld_path_true_final']
                # smod_path=self.path_file[self.path_train[head_neck_path]] 
                for true_path,true_key in self.neld_path_original_mm['keys'].items() : 
                    # if true_path not in iou_dict: 
                    iou_dict[true_path]={}
                    path_true=self.path_file[true_path]
                    path_appr=os.path.dirname(self.path_file[self.path_train[head_neck_path]])
                    path_grap_score_sh=os.path.join(path_appr ,'intensity_smods_logit_sh.txt')
                    path_grap_score_sp=os.path.join(path_appr ,'intensity_smods_logit_sp.txt')
                    # path_grap_score_sh=os.path.join(path_appr ,'intensity_smods_logit_sh.txt')
                    # path_grap_score_sp=os.path.join(path_appr ,'intensity_smods_logit_sp.txt')
                    # path_grap_score_sh=os.path.join(hjh[0])
                    # path_grap_score_sp=os.path.join(hjh[1])
                    path_grap_true=os.path.join(neld_path_true_final ,'intensity_1hot_hmod_smod.txt') 
                    # print(']]]]]]]][[[[[[[[[[[[[]]]]]]]]]]]]]',path_grap_true)
                    if os.path.exists(path_grap_score_sh) and os.path.exists(path_grap_true): 
                        nng=np.loadtxt(os.path.join(file_path,self.txt_vertices_old),dtype=float)
                        # print(']]]]]]]][[[[[[[[[[[[[]]]]]]]]]]]]]',path_grap_score_sh,np.loadtxt(path_grap_true,dtype=float).shape,
                        #       np.loadtxt(path_grap_score_sh,dtype=float).shape,
                        #       nng.shape)
                        y_true.append(np.loadtxt(path_grap_true,dtype=float))
                        score.append(np.array([np.loadtxt(path_grap_score_sh,dtype=float),np.loadtxt(path_grap_score_sp,dtype=float)]).T) 

'''

                path_init_param=os.path.join(self.obj_org_path,f"param.pkl")
                import pickle
                with open(path_init_param, "rb") as f:
                    pm = pickle.load(f) 
                rhs_dim=self.data_mode['pinn']['rhs_dim']

                key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'
                ku,kuu='inference','true' 
                for mn,tyy,tyyy in zip(['initial','final'],self.intensity_logit_dict[ku],self.intensity_logit_dict[kuu]):
                    patth,patthy=self.path_file_sub[tyy][key],self.path_file_sub[tyyy][key]
                    if os.path.exists(patth) and os.path.exists(patthy): 
 
                        snn,snny=np.loadtxt(patth, dtype=float),np.loadtxt(patthy, dtype=float)
                        for ii in range(rhs_dim['i'],rhs_dim['j']):
                            ntimi,mtimi=1+pm.nPart*pm.dimm*ii-1,pm.nPart*pm.dimm*(ii+1)
                            yy,sc = np.sum(snn[:,ntimi:mtimi],axis=1),np.sum(snny[:,ntimi:mtimi],axis=1)
                            print('[[[]]]',[yy.shape   ])
                            fpr, tpr, _ = roc_curve(yy, y_score=sc) 
                            # roc_auc = auc(fpr, tpr)

                            path=path_train[head_neck_path] 
                            smod_path_save=     self.path_file[f'result_{path}'] 
                            np.savetxt(os.path.join(smod_path_save,f'roc_{mn}_{ii+1}.txt'),np.array([fpr,tpr]).T)


                        
    '''
                ku='true'
                smods_logit=  self.intensity_logit_dict[ku]
                for kk,tyy in zip(['initial','final'],smods_logit):
                    patth=self.path_file_sub[tyy][key]
                    if os.path.exists(patth):
                        y_true.append(np.loadtxt(patth,dtype=float))


                        # self.metric_total_dic['roc_curve']['true']+=np.loadtxt(path_grap_true,dtype=float)
                        # self.metric_total_dic['roc_curve']['score']+=np.loadtxt(path_grap_iou,dtype=float)
            # print(y_true)print('=============',yy.shape)
            if score and y_true:
                yy_true=np.vstack(y_true)
                y_score=np.vstack(score)
                for yy,sc,nm in zip(yy_true.T,y_score.T,['initial','final']):
                    
                    fpr, tpr, _ = roc_curve(yy, y_score=sc) 
                    # roc_auc = auc(fpr, tpr)

                    path=path_train[head_neck_path] 
                    smod_path_save=     self.path_file[f'result_{path}'] 
                    np.savetxt(os.path.join(smod_path_save,f'roc_{nm}.txt'),np.array([fpr,tpr]).T)


                    iou_tr=iou_train(
                                    iou_thre=iou_thre,
                                    iou_dict=iou_dict[true_path],
                                    neld_first_name=self.neld_first_name,
                                    file_path=self.file_path,
                                    path_result=self.path_file[f'result_{path_train[head_neck_path]}'],)
                    iou_tr.get_mapping(save=True)
                    iou_tr.get_iou_save(save=True) 
                    if true_key=='true_0':
                        iou_tr.get_iou_match(iou_thre=0.2)
                    scatter_graph=get_iou_graph(self,smod_path, save_data=False)
                    with open(os.path.join(smod_path,f'plot_iou_graph_{true_key}.pkl'), 'wb') as f:
                        pickle.dump(scatter_graph, f)  
            path=path_train[head_neck_path] 
            smod_path_save=     self.path_file[f'result_{path}']
            col_name=['id','id_true','iou_single','iou_union']
            for true_path,true_key in self.neld_path_original_mm['keys'].items():
                with open(os.path.join(smod_path_save,f'iou_{true_key}.csv'), 'w') as f: 
                    f.write(',' + ','.join(col_name) + '\n') 
                    for i, row_name in enumerate(list(iou_dict[true_path].keys())):
                        row_data = ','.join(map(str, iou_dict[true_path][row_name])) 
                        f.write(f'{row_name} ,{row_data}\n')
            iou_dict={}'''
 



 
    def get_dash_pages(self,  
                        data_studied=None, 
                        file_path_org=None, 
                        disp_infos=None,
                        dash_pages_path=None,
                        path_train=None,
                        model_type=None,
                        obj_org_path_dict=None,
                        model_sufix_dic=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        disp_infos = disp_infos or self.disp_infos 
        data_studied = data_studied or self.data_studied
        file_path_org = file_path_org or self.file_path_org  
        dash_pages_path = dash_pages_path or self.dash_pages_path
        path_train=path_train or self.path_train
        model_type=model_type or self.model_type
        obj_org_path_dict=obj_org_path_dict or self.obj_org_path_dict
        model_sufix_dic=model_sufix_dic or self.model_sufix_dic


        neld_names=self.neld_names
        neld_namess=self.neld_namess
        neld_path_inits=self.neld_path_inits
        model_sufix=self.model_sufix
  
        print('get_dash_pages') 
        print('--------------------------------')
        print(f"Model Type     : {model_sufix}") 
        print(f"Model Portion  : {self.pre_portion}") 
        print(f"Destination    : {path_train['dest_smod_path']}") 
        print('--------------------------------')
        for index,neld_name in enumerate(neld_names):   
            # self.get_dash_pages_name(index,data_studied)
            self.get_neld_name(data_studied=data_studied,
                                index=index,
                            **param_dic['data']['get_neld_name']
                                ) 
            from pathlib import Path 

            dash_path_neld=os.path.join(self.dash_pages_path,self.data_studied,self.model_type, self.model_sufix,self.file_diff)  
            os.makedirs(dash_path_neld, exist_ok=True)
            self.dash_pages_name=os.path.join(dash_path_neld,f'{self.neld_names[index]}.py')

            dash_pages_name=self.dash_pages_name
            neld_name = neld_names[index] 
            neld_path_init=neld_path_inits[index]
            neld_namessi=neld_namess[index]
            neld_path_init=neld_path_inits[index]
            user_inputt=f'{index}' 
            # path_file_dir=os.path.join(self.dash_pages_path,self.data_studied,self.model_type,self.model_sufix ,os.path.dirname(self.file_diff) ,'path_files.pkl')
            path_file_dir=os.path.join(self.dash_pages_path,self.data_studied,self.model_type,self.model_sufix ,  self.file_diff,'path_files.pkl')
            path_file_dir=path_file_dir.replace('\\','/') 
            print('[[[[[[[[[[[[[[[[[[[[[[]]]dash_pages_path]]]]]]]]]]]]]]]]]]]',self.dash_pages_path)
            print('[[[[[[[[[[[[[[[[[[[[[[]]]dash_pages_path]]]]]]]]]]]]]]]]]]]',path_file_dir)
            dictr=dict(
                        path_file_dir=path_file_dir,
                        path_train=path_train,
                            path_file=self.path_file,
                            path_file_sub=self.path_file_sub,
                            pinn_dir_data=self.pinn_dir_data,
                            neld_data=self.neld_data, 
                            obj_org_path_dict=obj_org_path_dict,
                            model_sufix_dic=model_sufix_dic,
                            path_display=self.path_display,
                            path_display_dic=self.path_display_dic, 
                            path_heads=self.path_heads,
                            neld_path_original_mm=self.neld_path_original_mm,
                            param_dic=param_dic,
                            train_neld_param=self.data_mode['pinn'],
                            
                        )
            with open(path_file_dir, 'wb') as f:
                pickle.dump(dictr, f)

 
            get_text_dash_test(user_inputt,
                                   file_path_org.replace("\\", "//"),
                                   neld_path_init,   
                                   neld_name, 
                                   neld_namessi, 
                                   data_studied, 
                                   model_sufix,
                                   dash_pages_name,
                                   disp_infos,
                                    # path_file=self.path_file,
                                    # path_file_sub=self.path_file_sub,
                                    index=index,
                                    model_type=model_type,
                                   path_file_dir=path_file_dir, 
                                    pinn_dir_data=self.pinn_dir_data, 
                                    
                                   ) 
            from pathlib import Path
 
            model_sufix_all=self.model_sufix_dic['model_sufix_show'] 
            forbidden_page=[]
            for nn in self.path_heads:
                for nnn in model_sufix_all:
                    mmn=os.path.join(f'/{self.data_studied}',nn,nnn).replace('\\','/')
                    forbidden_page.append(mmn)
            # forbidden_page=[f'/{os.path.join(self.data_studied,nn)}/' for nn in self.path_heads]
            forbidden_page.extend(['/dsa',os.path.join('/dsa','test').replace('\\','/'),])
            forbidden_page=[hh.replace('_','-').lower()  for hh in forbidden_page] 
            pdir=os.path.dirname(self.dash_pages_path)  
            print('[[[[[[[[pdir]]]]]]]]',pdir)
            get_text_dash_app(  
                            dash_pages_path=os.path.join(pdir,'app.py'),
                            disp_infos=False, 
                            head_navbar=model_sufix_dic['head_navbar'],
                            forbidden_page = tuple(list(set(forbidden_page))),
                            # forbidden_endswith='-data-', 
                            ) 
            dash_pages_path = str(Path(self.dash_pages_path) / "Main_2.py")
            get_text_dash_main_2(
                    page_name="DSA",
                    page_dir_txt="DSA-2",
                    dash_pages_path=dash_pages_path,
                    disp_infos=False, 
                path_heads_show=model_sufix_dic['path_heads_show'],
                categories=model_sufix_dic['categories'],
                path_display=model_sufix_dic['path_display'],
                dnn_modes=model_sufix_dic['model_sufix_show'],
                )

            # page_name=self.model_sufix_dic['path_heads_dic'][self.model_type] #safe_id(self.neld_path_inits[index])
            page_name= self.model_sufix_dic['path_heads_dic_sec'].get(self.model_type,self.model_type)
            dash_pages_path=os.path.join(self.dash_pages_path,self.data_studied, f'{self.model_type}_data.py')  
            page_dir_txt=os.path.join(self.data_studied,self.model_type  ) 
            page_dir_txt=os.path.join(page_dir_txt,f'{self.model_type}-data-' ).replace('_','-').replace('\\','/').lower()  
            page_dir_txt=page_dir_txt
            get_text_dash_dnn(  
                                page_name=f'{page_name}',
                                page_dir_txt=page_dir_txt,
                                dash_pages_path=dash_pages_path,
                                disp_infos=False,
                                path_train=None, 
                                path_file=None, 
                                pinn_dir_data=None,
                                neld_data=None,
                                index=None,
                                page_view='results', 
                                # forbidden='-data-',
                                )
# 

            id_name_end=self.neld_path_inits[index]
            page_name=self.model_sufix
            page_name=model_sufix_dic['model_sufix_dic'][self.model_sufix]
            dash_pages_path=os.path.join(self.dash_pages_path,self.data_studied,self.model_type ,f'{self.model_type}_data_{self.model_sufix}.py')

            dash_pages_path=f'{dash_pages_path}' 
            page_dir_txt=os.path.join(self.data_studied,self.model_type ,self.model_sufix ,f'{self.model_type}-data-{self.model_sufix}'  ).replace('_','-').replace('\\','/').lower() 
            get_text_dash_dnn(  
                                page_name=f'{page_name}',
                                page_dir_txt=page_dir_txt,
                                dash_pages_path=dash_pages_path,
                                disp_infos=False,
                                path_train=None, 
                                path_file=None, 
                                pinn_dir_data=None,
                                neld_data=None,
                                index=None,
                                page_view='results', 
                                # forbidden='-data-',
                                )

            id_name_end=self.neld_path_inits[index]
            vhhv = self.file_diff.replace("\\", "_").replace("/", "_")
            dash_pages_path=os.path.join(self.dash_pages_path,self.data_studied,self.model_type ,self.model_sufix ,   f'{self.model_type}_data_{self.model_sufix}_{vhhv}.py')

            dash_pages_path=f'{dash_pages_path}' 
            page_dir_txt=os.path.join(self.data_studied,self.model_type ,self.model_sufix ,self.file_diff ).replace('_','-').replace('\\','/').lower() 
            # page_dir_txt=os.path.join(self.data_studied,self.model_type ,self.model_sufix ,id_name_end -{self.file_diff}).replace('_','-').replace('\\','/').lower() 
            # os.makedirs(os.path.dirname())
            print('[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]', page_dir_txt,vhhv)
            # page_name=self.file_diff.split('/')[-1]

            page_name = os.path.basename(self.file_diff)
            print('[[[[[[[[[[[[[[[page_name]]]]]]]]]]]]]]]',page_name)
            get_text_dash_dnn(  
                                page_name=page_name,
                                # page_name=f'{id_name_end}',
                                page_dir_txt=page_dir_txt,
                                dash_pages_path=dash_pages_path,
                                disp_infos=False,
                                path_train=None, 
                                path_file=None, 
                                pinn_dir_data=None,
                                neld_data=None,
                                index=None,
                                page_view='visualization', 
                                )



