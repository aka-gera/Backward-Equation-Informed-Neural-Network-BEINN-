











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
import trimesh

import importlib
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"  
from tqdm import tqdm 
 
import neld_fun_0.help_funn as hff   


from neld_fun_0.get_path import assign_if_none,get_name,get_param,get_files
 


class train_test_tf(get_files,get_name): 
    def __init__(self, **kwargs):
        get_name.__init__(self) 
        get_files.__init__(self,**kwargs)

    def get_train_input_ML(self,  
                        path_train, 
                        pre_portion,  
                        DTYPE=None, 
                        file_path_model_data=None,
                        data_studied=None, 
                        line_num_points_hmod=None,
                        line_num_points_inter_hmod=None,
                        spline_smooth_hmod=None,   
                        model_sufix=None,
                        disp_infos=None,
                        txt_save_file=None,
                        neld_names=None,
                        weight_positive=.5,  
                        list_features=None,
                        base_features_list=None,
                        model_type=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        tf_train=True,
                        entry_names=[None],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features = list_features if list_features is not None else self.list_features
        disp_infos = disp_infos or self.disp_infos  
        model_sufix = model_sufix or self.model_sufix  
        file_path_model_data = file_path_model_data or self.file_path_model_data
        data_studied = data_studied or self.data_studied  
        line_num_points_hmod = line_num_points_hmod or self.line_num_points_hmod
        line_num_points_inter_hmod = line_num_points_inter_hmod or self.line_num_points_inter_hmod
        spline_smooth_hmod = spline_smooth_hmod or self.spline_smooth_hmod
        neld_names = neld_names if neld_names is not None else self.neld_names
        DTYPE=DTYPE or self.DTYPE 

        curv, rhs, weight, indices,adj,neld = [], [], [], [],[],[]

        for index, neld_name in enumerate(neld_names): 
            self.get_neld_name(data_studied=data_studied, index=index,model_type=model_type) 
            smod_portion_path=self.path_file[path_train['data_smod_path']] 
            hmod_portion_path=self.path_file[path_train['data_hmod_path']] 
            pid = pinn_data(file_path=self.file_path,
                            file_path_feat=self.file_path_feat,
                            path_file=self.path_file,
                            smod_path = smod_portion_path,
                            hmod_path =  hmod_portion_path , 
                            smod_path_pre=smod_portion_path,
                            hmod_path_pre=  hmod_portion_path ,  
                            neld_path_original_m=self.neld_path_original_m,
                            neld_first_name=self.neld_namess[index][1],
                            model_sufix=model_sufix,
                            path_train=path_train,
                            line_num_points_hmod=line_num_points_hmod,
                            line_num_points_inter_hmod=line_num_points_inter_hmod,
                            spline_smooth_hmod=spline_smooth_hmod,
                            thre_target_number_of_triangles=thre_target_number_of_triangles,
                            voxel_resolution=voxel_resolution,
                            dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                            neld_path_true_final=self.neld_path_true_final,
                                )
            pid.save_pinn_data()    
            pid.get_neld_data()
            feat_paths=[]  
            for path_inten,name_inten in list_features:  
                pathh=os.path.join( self.file_path_feat ,name_inten)
                if os.path.exists(pathh):
                    feat_paths.append(pathh)
                    print('train data path ---->>',self.file_path_feat,pathh) 
                    print(np.loadtxt(pathh))
                else:
                    print('path doesnt exists ===----->>>',pathh)

            curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,base_features_list=base_features_list,))) 
            neld.append(pid.neld)     
            if tf_train:
                rhs.append(pid.get_pinn_rhs(pre_portion=pre_portion) )

        return curv, rhs ,adj,neld





 


 




    def get_train_input_dnn(self,  
                        path_train, 
                        pre_portion,  
                        DTYPE=None, 
                        file_path_model_data=None,
                        data_studied=None, 
                        line_num_points_hmod=None,
                        line_num_points_inter_hmod=None,
                        spline_smooth_hmod=None,   
                        model_sufix=None,
                        disp_infos=None,
                        txt_save_file=None,
                        neld_names=None,
                        weight_positive=.5,  
                        list_features=None,
                        base_features_list=None,
                        model_type=None,
                        model_init=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        tf_train=True,
                        entry_names=[None],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features = list_features if list_features is not None else self.list_features
        model_init = model_init if model_init is not None else self.model_init
        disp_infos = disp_infos or self.disp_infos  
        model_sufix = model_sufix or self.model_sufix  
        file_path_model_data = file_path_model_data or self.file_path_model_data
        data_studied = data_studied or self.data_studied  
        line_num_points_hmod = line_num_points_hmod or self.line_num_points_hmod
        line_num_points_inter_hmod = line_num_points_inter_hmod or self.line_num_points_inter_hmod
        spline_smooth_hmod = spline_smooth_hmod or self.spline_smooth_hmod
        neld_names = neld_names if neld_names is not None else self.neld_names
        DTYPE=DTYPE or self.DTYPE 

        curv, rhs, weight, indices,adj,neld = [], [], [], [],[],[]

        indexx={neld_name:index for index, neld_name  in enumerate(self.neld_names)} 
        for entry_name in entry_names:
            for neld_name in neld_names: 
                index=indexx[neld_name]
                self.get_neld_name(data_studied=data_studied, index=index,model_type=model_type,entry_name=entry_name) 
                smod_portion_path=self.path_file[path_train['data_smod_path']] 
                hmod_portion_path=self.path_file[path_train['data_hmod_path']] 
                pid = pinn_data(file_path=self.file_path,
                                file_path_feat=self.file_path_feat,
                                path_file=self.path_file,
                                smod_path = smod_portion_path,
                                hmod_path =  hmod_portion_path , 
                                smod_path_pre=smod_portion_path,
                                hmod_path_pre=  hmod_portion_path ,  
                                neld_path_original_m=self.neld_path_original_m,
                                neld_first_name=self.neld_namess[index][1],
                                model_sufix=model_sufix,
                                path_train=path_train,
                                line_num_points_hmod=line_num_points_hmod,
                                line_num_points_inter_hmod=line_num_points_inter_hmod,
                                spline_smooth_hmod=spline_smooth_hmod,
                                thre_target_number_of_triangles=thre_target_number_of_triangles,
                                voxel_resolution=voxel_resolution,
                                dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                                kmean_n_run=kmean_n_run,
                                kmean_max_iter=kmean_max_iter,
                                param_dic=param_dic, 
                                        neld_path_true_final=self.neld_path_true_final,
                                    ) 
                file_path_feat = self.file_path_feat if entry_name is None else self.file_path_feat_entry
                file_path = self.file_path if entry_name is None else self.file_path_entry
                pid.save_pinn_data(file_path_feat=file_path_feat,
                                file_path=file_path,
                                )    
                pid.get_neld_data(file_path_feat=file_path_feat,
                                file_path=file_path,)
                feat_paths=[]  
                for path_inten,name_inten in list_features: 
                    if model_init is not None: 
                        mode_dnn_1=self.model_sufix_dic['model_sufix_inverse'][model_init]
                        path_dir=self.model_sufix_dic['path_dir']
                        path_hmod_dir=f'{self.model_type}_{mode_dnn_1}_{path_dir}'
                        pathini=self.path_file[path_hmod_dir]
                    else:
                        pathini=file_path_feat

                    pathini=  self.path_file[path_train['data_hmod_path']]   

                    pathh=os.path.join( pathini ,name_inten)
                    if os.path.exists(pathh):
                        feat_paths.append(pathh)
                        print('train data path ---->>',pathini,pathh) 
                        print(np.loadtxt(pathh))
                    else:
                        print('path doesnt exists ===----->>>',pathh)
                neld.append(pid.neld)     
                if tf_train:
                    # if pre_portion=='head': 
                    #     pathh=os.path.join( self.neld_path_true_final ,'intensity_hmod.neck_head.txt')
                    # elif pre_portion=='neck_head': 
                    #     pathh=os.path.join( self.neld_path_true_final ,'intensity_hmod_neck_head.txt')
                    # else:
                    #     pathh=os.path.join( self.neld_path_true_final ,'intensity_hmod_smod.txt')
                    # rhs.append(tf.cast(
                    #     pid.get_pinn_rhs(pre_portion=pre_portion,
                    #                     file_path_feat=file_path_feat,
                    #                     file_path=file_path,
                    #                     neld_path_true_final=self.neld_path_true_final,
                    #                     ), 
                    #     dtype=DTYPE)
                    #         ) 
                    rhs.append(tf.cast(
                        np.loadtxt(os.path.join( self.neld_path_true_final ,f'intensity_1hot_hmod_{pre_portion}.txt'),dtype=int), 
                        dtype=DTYPE)
                            ) 
                    pathh=os.path.join( self.neld_path_true_final ,f'intensity_hmod_{pre_portion}.txt')
                    mask=np.loadtxt(pathh,dtype=int)
                    unique=np.sort(np.unique(mask))
                    labels = {v: np.argwhere(mask == v) for v in unique}
                    counts = {v: labels[v].shape[0] for v in labels}
                    total = sum(counts.values()) 

                    w_prime = np.array([max(np.log(2 * total / counts[k]), 1) for k in unique ])
        
                    weight.append(w_prime / np.sum(w_prime))






                if len(list(list_features)+list(base_features_list))==0:
                    curv.append(pid.neld.vertices)  
                    continue
                    
                curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,
                                                            base_features_list=base_features_list,
                                                            file_path_feat=file_path_feat,
                                                            file_path=file_path,))) 

        return curv, rhs ,adj,neld,weight
 
 
    def train_model_smod_dnn(self,
                            path_train=None,
                            get_training=True, 
                            pre_portion=None,
                            full_neld=True, 
                            hidden_layers=None, 
                            neurons_per_layer=None, 
                            activation_init=None,
                            activation_hidden=None,
                            activation_last=None, 
                            curv=None,
                            rhs=None,
                            weight=None,
                            indices=None,
                            DTYPE=None, 
                            model_sufix=None,
                            file_path_model_data=None,
                            line_num_points_hmod=None,
                            line_num_points_inter_hmod=None, 
                            spline_smooth_hmod=None,
                            data_studied=None,
                            vv_cts=None,
                            new_model=True, 
                            itime = 10000, 
                            itime_div=1,
                            loss_save_dir=None,
                            iou_save_dir=None,  
                            auc_save_dir=None,   
                            dice_save_dir=None,
                            index_save_dir=None,  
                            model_dir=None, 
                            loss_mode="bce",
                            ls=[2,3,4,5],
                            disp_infos=None, 
                            weight_positive= .5,
                            list_features=None,
                            base_features_list=None,
                            train_smods=False,
                            dest_path='dest_smod_path',
                            model_type=None,
                            num_sub_nodes=None,
                            rl_par= 0.5, 
                            dnn_par= 0.5,
                            thre_target_number_of_triangles=None,
                            voxel_resolution=None,
                            l1_values = [0,   1e-6,   1e-4, 1e-2],
                            l2_values = [0,   1e-6,   1e-4, 1e-2],
                            dict_mesh_to_skeleton_finder_mesh=None,
                        entry_names=[],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        train_neld_param=None,
                        path_dict=None,

                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh 
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features=list_features if list_features is not None else self.list_features
        disp_infos = disp_infos or self.disp_infos 
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        DTYPE = DTYPE or self.DTYPE
        vv_cts = vv_cts or self.vv_cts 
        model_sufix = model_sufix or self.model_sufix 
        self.get_model_opt_name(model_sufix=model_sufix,model_type=model_type)
        line_num_points_hmod=line_num_points_hmod or self.line_num_points_hmod
        line_num_points_inter_hmod=line_num_points_inter_hmod or self.line_num_points_inter_hmod
        file_path_model_data = file_path_model_data or self.file_path_model_data
        data_studied = data_studied or self.data_studied 
        spline_smooth_hmod=self.spline_smooth_hmod
 
 
        hidden_layers=hidden_layers or self.hidden_layers 
        neurons_per_layer=neurons_per_layer or self.neurons_per_layer 
        activation_init=activation_init or self.activation_init
        activation_hidden=activation_hidden or self.activation_hidden
        activation_last=activation_last or self.activation_last
 
        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default']) 
        loss_save_dir = loss_save_dir or model_dirs['loss'] 
        oloss_save_dir =  model_dirs['oloss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou'] 
        auc_save_dir = auc_save_dir or model_dirs['auc']
        dice_save_dir = dice_save_dir or model_dirs['dice']
        index_save_dir = index_save_dir or  model_dirs['index_save'] 
        # model_dir = model_dir or  model_dirs['model']   

        # from neld_pinn_0.run_1 import aka_grad ,lang_loss, neld_coef,get_neld_data_train

        run_module = importlib.import_module(path_dict['run']) 
        lang_loss = run_module.lang_loss
        neld_coef = run_module.neld_coef
        get_neld_data_train = run_module.get_neld_data_train
        aka_grad = run_module.aka_grad

        from neld_pinn_0.NELD_PINN import aka_train_md 
        # from PINN import PINN, aka_train

        train_neld_param=self.data_mode['pinn']
        train_neld_param['tf_train']=True 

        print('[[[[[[[[[]]]]]]]]]',train_neld_param)
        bcs_train_test,sav,par,pm,Nterm=get_neld_data_train(**train_neld_param,self=self)
        # train_neld_param['inter']=np.arange(0,pm.N,1)  

        res_coef_ = neld_coef(par,sav,bcs_train_test['train'],Nterm=Nterm)






        n_classes=par['n_classes']
 
        adj_train =[] 

        from neld_fun_0.help_dnn_one_hot import Get_iou,model_choice,get_auc,model_metric
        with tf.device(device):
            mchoice = model_choice(model_type=model_type,
                                   n_classes=n_classes,
                #  activation_init="tanh",
                #  activation_hidden="tanh",
                #  activation_last="tanh", 
                 )
            model = mchoice.get_model() 
            custom = mchoice.get_custom_objects(model_type) 
                 
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [1e-2, 1e-3, 5e-4])
        optimizer = tf.optimizers.Adam(learning_rate=lr) 
        if new_model:
            print(f"I'm starting a new model")
            print('--------------------------------')
            print(f"Model Type Gen : {model_type}")
            print(f"Model Type     : {model_sufix}")
            print(f"Model Dir.     : {model_dir}")
            print(f"Model Portion  : {pre_portion}")
            print(f"Data Dir.      : {path_train['data_hmod_path']}") 
            # print(f"Base Feat. list: {[mm for mm in base_features_list] if base_features_list is not None else list(self.base_features_dict.keys())}") 
            # print(f"Feat. list     : {[mm for _,mm in list_features]}")  
            print(f"Hidden Layers        : {hidden_layers}")
            print(f"neurons_per_layer    : {neurons_per_layer}")
            print('--------------------------------')
            loss_save = []
            iou_save = {0: [], 1: [], 2: []}
            auc_save = {0: [], 1: [], 2: []}
            dice_save = {0: [], 1: [], 2: []}
            loss_tmp=10**10
            head_tmp=0
        else:
            print("This is a continuation of the previous model") 
            loss_save = np.loadtxt(loss_save_dir, dtype=float).tolist()
            iou_save = np.loadtxt(iou_save_dir, dtype=float).tolist() 
            auc_save = np.loadtxt(auc_save_dir, dtype=float).tolist() 
            dice_save = np.loadtxt(dice_save_dir, dtype=float).tolist() 
            
        print(f"Using device: {device}------------------ ----------------------------------------------")
        aka_train_ = aka_train_md()
        fun = lang_loss() 

        loss_tmp_all=1e10
        modlist=['loss','iou','auc','dice']
        metr={mm:{f'{nn}_tmp':[0,0] for nn in modlist } for mm in modlist}
        metr['loss']['loss_tmp']=1e6 
        tf.random.set_seed(0)
        iou_tmp= {0: 0, 1: 0, 2: 0};auc_tmp= {0: 0, 1: 0, 2: 0};dice_tmp= {0: 0, 1: 0, 2: 0}
        index_save=[]
        pbar = tqdm(range(itime), desc=f"Loss: {loss_tmp:.6f}| IoU: N/A")  
        for i in pbar: 
            loss = aka_train_.train_PINN(optimizer, fun, model,res_coef_)
            # iou=Get_iou(model, curv, lab=indd,
            #             adj=adj, 
            #        neld=neld,)
            # auc=get_auc(model, curv, rhs=rhs)
            '''
            mtr=model_metric(model=model,curv=curv,rhs_index=rhs_index_ind, )
            mtric=mtr.metrics
            iou=mtric['iou']
            auc=mtric['auc']
            dice=mtric['dice']
            for ii in range(rhs[0].shape[1]):
                iou_save[ii].append(mtric['iou_ind'][ii]) 
                np.savetxt(iou_save_dir[ii] , np.array(iou_save[ii]), fmt='%f') 
                auc_save[ii].append(mtric['auc_ind'][ii]) 
                np.savetxt(auc_save_dir[ii] , np.array(auc_save[ii]), fmt='%f') 
                dice_save[ii].append(mtric['dice_ind'][ii]) 
                np.savetxt(dice_save_dir[ii] , np.array(dice_save[ii]), fmt='%f')

                ''' 

            loss_save.append(loss)
            np.savetxt(loss_save_dir, np.array(loss_save), fmt='%f') 

            

            res_coef_test = neld_coef(par,sav,bcs_train_test['test'],Nterm=Nterm)
            lossi=0
            for val in res_coef_test.bcs.values():
                if val['tf']:  
                    lossi+=  aka_grad().loss_fn(model(val['coord']),val['u'])
        
 
            hh=mmjl='loss'
            if lossi < metr[hh][f'{hh}_tmp']: 
                model.save(model_dirs[f'model_{hh}'])  
                metr[hh][f'{hh}_tmp']=lossi
                loss_tmp=lossi
                index_save.append(i)
                np.savetxt(index_save_dir, np.array(index_save), fmt='%d') 



            hh='oloss'
            if loss <loss_tmp_all:
                loss_tmp_all=loss
                # model.save(model_dir)
                model.save(model_dirs[f'model_{hh}']) 


            pbar.set_description(f"Loss save: {loss_tmp:.6f} | Loss: {loss:.6f} ")














            '''
            for hh in modlist:
                if hh !='loss':
                    if mtric[hh][0] > metr[hh][f'{hh}_tmp'][0]:  
                        model.save(model_dirs[f'model_{hh}']) 
                        metr[hh][f'dice_tmp']=dice
                        metr[hh][f'auc_tmp']=auc
                        metr[hh][f'iou_tmp']=iou
                        metr[hh][f'loss_tmp']=loss 
                else:
                    # print('[[[[]]]]',loss.numpy(),metr)
                    if loss.numpy() < metr[hh][f'{hh}_tmp']:  
                        model.save(model_dirs[f'model_{hh}']) 
                        metr[hh][f'dice_tmp']=dice
                        metr[hh][f'auc_tmp']=auc
                        metr[hh][f'iou_tmp']=iou
                        metr[hh][f'loss_tmp']=loss 
    
            # if loss.numpy() < loss_tmp: 
            #     loss_tmp=loss.numpy()
            #     model.save(model_dir['model_loss']) 
            #     metr['loss']=dice[0]
            mmj=np.argmax([metr[mm]['dice_tmp'][0] for mm in modlist])
            mmjl=modlist[mmj]
            # optimizer=metr[mmjl][f'optimizer']
            # old_opt = old_model.optimizer
 
            modelsa = load_model(model_dirs[f'model_{mmjl}'], custom_objects=custom) 
            modelsa.save(model_dir) 
            index_save.append(i)
            np.savetxt(index_save_dir, np.array(index_save), fmt='%d') 
            loss_tmp,iou_tmp,auc_tmp,dice_tmp=metr[mmjl][f'loss_tmp'],metr[mmjl][f'iou_tmp'],metr[mmjl][f'auc_tmp'],metr[mmjl][f'dice_tmp']
            # if loss.numpy() < loss_tmp:
            # # if auc[0] > auc_tmp[0]:
            # # if dice[0] > dice_tmp[0]:
            #     # iou_tmp=iou  
            #     # dice_tmp=dice 
            #     loss_tmp=loss.numpy()
            #     model.save(model_dir) 
            #     index_save.append(i)
            #     np.savetxt(index_save_dir, np.array(index_save), fmt='%d')  
            # print(auc_tmp)
            if pre_portion=='neck_head':
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU sh: { iou_tmp[0]:.4f} IoU nk: {iou_tmp[1]:.4f} IoU hd: {iou_tmp[2]:.4f}")
            elif pre_portion in ('smod','head','neck'):
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU(sh: {iou_tmp[0]:.2f}, sp: {iou_tmp[1]:.2f})|  DICE(sh: {dice_tmp[0]:.2f}, sp: {dice_tmp[1]:.2f}) |  AUC(sh: {auc_tmp[0]:.2f}, sp: {auc_tmp[1]:.2f})")
            else:
            '''
   
 
    def get_hmod_pred_dnn(self, 
                        path_train=None, 
                        pre_portion=None, 
                        n_col=None,
                        hidden_layers=None, 
                        neurons_per_layer=None, 
                        activation_init=None,
                        activation_hidden=None,
                        activation_last=None, 
                        model= None,
                        curv=None,
                        size_threshold=None,
                        neld_names=None, 
                        model_dir=None, 
                        loss_save_dir=None,
                        iou_save_dir=None,
                        auc_save_dir=None,
                        model_sufix=None, 
                        data_studied=None,
                        file_path_org=None,  
                        line_num_points_hmod=None,
                        line_num_points_inter_hmod=None,
                        spline_smooth_hmod=None ,  
                        disp_infos=None, 
                        DTYPE=None,
                        list_features=None,
                        base_features_list=None,
                        hmod_thre=None,
                        train_smods=False,
                        weight=None,
                        weight2=None,
                        weights=None,
                        neck_lim=None,
                        n_clusters=3,  
                        smooth_tf=False,
                        model_type=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        reconstruction_tf=False, 
                        dict_mesh_to_skeleton_finder=None,
                        dict_wrap=None,
                        tf_skl_hmod_distance=False,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        train_neld_param=None,
                        path_dict=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution
        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features=list_features if list_features is not None else self.list_features
        size_threshold=size_threshold or self.size_threshold
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        disp_infos = disp_infos or self.disp_infos  
        file_path_org = file_path_org or self.file_path_org
        model_sufix=model_sufix or self.model_sufix
        self.get_model_opt_name(model_sufix=model_sufix,
                                model_type=model_type,)  
        line_num_points_hmod = line_num_points_hmod or self.line_num_points_hmod
        line_num_points_inter_hmod = line_num_points_inter_hmod or self.line_num_points_inter_hmod
        spline_smooth_hmod = spline_smooth_hmod or self.spline_smooth_hmod
        DTYPE=DTYPE or self.DTYPE 
        data_studied = data_studied or self.data_studied 

        hidden_layers=hidden_layers or self.hidden_layers 
        neurons_per_layer=neurons_per_layer or self.neurons_per_layer 
        activation_init=activation_init or self.activation_init
        activation_hidden=activation_hidden or self.activation_hidden
        activation_last=activation_last or self.activation_last

        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default'])
  
        loss_save_dir = loss_save_dir or model_dirs['loss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou']    
        auc_save_dir = auc_save_dir or  model_dirs['auc'] 
        # model_dir = model_dir or  model_dirs['model'] 
        model_type_split=self.path_dir.lower().split('_') 
        for mo in ['dice','iou','auc','loss','oloss']:
            if mo in model_type_split:
                model_dir =  model_dirs[f'model_{mo}']  
                break 
        # mo='oloss'
        model_dir =  model_dirs[f'model_{mo}']
        # model_dir =  model_dirs[f'model']
        rhs_name='rhs_name'
        n_col=n_col or len(self.model_dir_path[pre_portion][rhs_name])  
 
        # from neld_fun_0.help_dnn_one_hot import model_choice  
        # from neld_pinn_0.run_1 import lang_loss, neld_coef,get_neld_data_train,aka_grad
        from neld_fun_0.help_dnn_one_hot import model_choice  

        run_module = importlib.import_module(path_dict['run'])
 
        lang_loss = run_module.lang_loss
        neld_coef = run_module.neld_coef
        get_neld_data_train = run_module.get_neld_data_train
        aka_grad = run_module.aka_grad

        
 
        # from PINN import PINN, aka_train
        train_neld_param=self.data_mode['pinn']
        train_neld_param['tf_train']=False
        train_neld_param['Samples_nbr']=[[],[]]
        print('[[[[[[[[[]]]]]]]]]',train_neld_param)
        bcs_train_test,sav,par,pm,Nterm=get_neld_data_train(**train_neld_param,self=self)
        print('[[[[[[[[[]]]]]]]]]',par['n_classes'])


        n_classes=par['n_classes']
        mchoice = model_choice(model_type=model_type,n_classes=n_classes)
        custom = mchoice.get_custom_objects(model_type) 
        model = load_model(model_dir, custom_objects=custom)

        print(f'get_hmod_pred started')
        print('--------------------------------')
        print(f"Model Type Gen : {model_type}")
        print(f"Model Type     : {model_sufix}")
        print(f"Model Dir.     : {model_dir}")
        print(f"Model Portion  : {pre_portion}") 
        print(f"Data Dir.      : {path_train['data_hmod_path']}") 
        print(f"Destin. Dir.   : {path_train['dest_hmod_path']}")  
        print('--------------------------------')
        time_start = time.time()
        rhs_name='rhs_name'
        neld_names = neld_names or self.neld_names 
        for  index,neld_name in enumerate(neld_names):
            self.get_neld_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type, )  
            neld_name=self.neld_name  
            smod_portion_path=self.path_file[path_train['data_hmod_path']]  

            train_neld_param['Samples_nbr']=[[index,],[]]
            print('[[[[[[[[[]]]]]]]]]',train_neld_param)
            print('[[[[[[[[[]]]]]]]]]',par['n_classes'])

            bcs_train_test,sav,par,pm,Nterm=get_neld_data_train(**train_neld_param,self=self)

            res_coef_ = neld_coef(par,sav,bcs_train_test[index],Nterm=Nterm)


            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'
            ku='inference'
            smods_logit=  self.intensity_logit_dict[ku]
            for kk,tyy in zip(['initial','final'],smods_logit):
                if res_coef_.bcs[kk]['tf']:
                    rhs0=model(res_coef_.bcs[kk]['coord']).numpy()    
                    np.savetxt(self.path_file_sub[tyy][key], rhs0, fmt='%f') 
                    print('[[[[[[[[[]rhs  0]]]]]]]]',par['n_classes'],rhs0.shape,self.path_file_sub[tyy][key])


            ku='true'
            smods_logit=  self.intensity_logit_dict[ku]
            for kk,tyy in zip(['initial','final'],smods_logit):
                if res_coef_.bcs[kk]['tf']:
                    rhs0=res_coef_.bcs[kk]['u']    
                    np.savetxt(self.path_file_sub[tyy][key], rhs0, fmt='%f') 
 

            
            mytime0 = time.time() - time_start
 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60)
            if disp_infos:
                print(f'hmod Prediction completed on {neld_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
                print(f'data stores in: {smod_portion_path}')  
        print('hmod Prediction completed') 






    def get_roc(self,   
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
        print('ROC started=-------------------------------------')
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
            for  index,neld_name in enumerate(neld_names):
                self.get_neld_name(data_studied=data_studied,index=index ,
                                ) 
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
                            yy,sc = np.sum(snn[:,ntimi:mtimi],axis=1).reshape(-1,1),np.sum(snny[:,ntimi:mtimi],axis=1).reshape(-1,1)
                            print('[[[]]]',[yy.shape,sc.shape])
                            fpr, tpr, _ = roc_curve(yy, y_score=sc) 
                            # roc_auc = auc(fpr, tpr)

                            path=path_train[head_neck_path] 
                            smod_path_save=     self.path_file[f'result_{path}'] 
                            np.savetxt(os.path.join(smod_path_save,f'roc_{mn}_{ii+1}.txt'),np.array([fpr,tpr]).T)






    def get_train_input_pnet(self,  
                        path_train, 
                        pre_portion,  
                        DTYPE=None, 
                        file_path_model_data=None,
                        data_studied=None, 
                        line_num_points_hmod=None,
                        line_num_points_inter_hmod=None,
                        spline_smooth_hmod=None,   
                        model_sufix=None,
                        disp_infos=None,
                        txt_save_file=None,
                        neld_names=None,
                        weight_positive=.5,  
                        weight2=None,
                        list_features=None,
                        base_features_list=None,
                        model_type=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        tf_train=True,
                        entry_names=[None],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features = list_features if list_features is not None else self.list_features
        disp_infos = disp_infos or self.disp_infos  
        model_sufix = model_sufix or self.model_sufix  
        file_path_model_data = file_path_model_data or self.file_path_model_data
        data_studied = data_studied or self.data_studied  
        line_num_points_hmod = line_num_points_hmod or self.line_num_points_hmod
        line_num_points_inter_hmod = line_num_points_inter_hmod or self.line_num_points_inter_hmod
        spline_smooth_hmod = spline_smooth_hmod or self.spline_smooth_hmod
        neld_names = neld_names if neld_names is not None else self.neld_names
        DTYPE=DTYPE or self.DTYPE 

        curv, rhs, weight, indices,adj,neld = [], [], [], [],[],[]

        indexx={neld_name:index for index, neld_name  in enumerate(self.neld_names)} 
        for entry_name in entry_names:
            for neld_name in neld_names: 
                index=indexx[neld_name]
                self.get_neld_name(data_studied=data_studied, index=index,model_type=model_type,entry_name=entry_name) 
                smod_portion_path=self.path_file[path_train['data_smod_path']] 
                hmod_portion_path=self.path_file[path_train['data_hmod_path']] 
                pid = pinn_data(file_path=self.file_path,
                                file_path_feat=self.file_path_feat,
                                path_file=self.path_file,
                                smod_path = smod_portion_path,
                                hmod_path =  hmod_portion_path , 
                                smod_path_pre=smod_portion_path,
                                hmod_path_pre=  hmod_portion_path ,  
                                neld_path_original_m=self.neld_path_original_m,
                                neld_first_name=self.neld_namess[index][1],
                                model_sufix=model_sufix,
                                path_train=path_train,
                                line_num_points_hmod=line_num_points_hmod,
                                line_num_points_inter_hmod=line_num_points_inter_hmod,
                                spline_smooth_hmod=spline_smooth_hmod,
                                thre_target_number_of_triangles=thre_target_number_of_triangles,
                                voxel_resolution=voxel_resolution,
                                dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                                        neld_path_true_final=self.neld_path_true_final,
                                    )
                file_path_feat = self.file_path_feat if entry_name is None else self.file_path_feat_entry
                file_path = self.file_path if entry_name is None else self.file_path_entry
                pid.save_pinn_data(file_path_feat=file_path_feat,
                                file_path=file_path,
                                )    
                pid.get_neld_data(file_path_feat=file_path_feat,
                                file_path=file_path,)
                feat_paths=[]  
                for path_inten,name_inten in list_features:  
                    pathh=os.path.join( file_path_feat ,name_inten)
                    if os.path.exists(pathh):
                        feat_paths.append(pathh)
                        print('train data path ---->>',file_path_feat,pathh) 
                        print(np.loadtxt(pathh))
                    else:
                        print('path doesnt exists ===----->>>',pathh)
                neld.append(pid.neld)     
                if tf_train:
                    rhspp=tf.cast(

                        pid.get_pinn_rhs(pre_portion=pre_portion,
                                        file_path_feat=file_path_feat,
                                        file_path=file_path,
                                        neld_path_true_final=self.neld_path_true_final,
                                        ),  
                        dtype=DTYPE)
                    rhs.append(rhspp)
                if len(list(list_features)+list(base_features_list))==0: 
                    points = tf.expand_dims(tf.convert_to_tensor(pid.neld.vertices), axis=0)   
                    curv.append(points) 
                    continue
                    
                curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,
                                                            base_features_list=base_features_list,
                                                            file_path_feat=file_path_feat,
                                                            file_path=file_path,))) 
        return curv, rhs ,adj,neld
 
 
    def train_model_smod_pnet(self,
                            path_train=None,
                            get_training=True, 
                            pre_portion=None,
                            full_neld=True, 
                            hidden_layers=None, 
                            neurons_per_layer=None, 
                            activation_init=None,
                            activation_hidden=None,
                            activation_last=None, 
                            curv=None,
                            rhs=None,
                            weight=None,
                            indices=None,
                            DTYPE=None, 
                            model_sufix=None,
                            file_path_model_data=None,
                            line_num_points_hmod=None,
                            line_num_points_inter_hmod=None, 
                            spline_smooth_hmod=None,
                            data_studied=None,
                            vv_cts=None,
                            new_model=True, 
                            itime = 10000, 
                            itime_div=1,
                            loss_save_dir=None,
                            iou_save_dir=None,    
                            auc_save_dir=None,
                            dice_save_dir=None,
                            index_save_dir=None,  
                            model_dir=None, 
                            loss_mode="bce",
                            ls=[2,3,4,5],
                            disp_infos=None, 
                            weight_positive= .5,
                            list_features=None,
                            base_features_list=None,
                            train_smods=False,
                            dest_path='dest_smod_path',
                            model_type=None,
                            num_sub_nodes=None,
                            rl_par= 0.5, 
                            dnn_par= 0.5,
                            thre_target_number_of_triangles=None,
                            voxel_resolution=None,
                            l1_values = [0,   1e-6,   1e-4, 1e-2],
                            l2_values = [0,   1e-6,   1e-4, 1e-2],
                            dict_mesh_to_skeleton_finder_mesh=None,
                        entry_names=[],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh 
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features=list_features if list_features is not None else self.list_features
        disp_infos = disp_infos or self.disp_infos 
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        DTYPE = DTYPE or self.DTYPE
        vv_cts = vv_cts or self.vv_cts 
        model_sufix = model_sufix or self.model_sufix 
        self.get_model_opt_name(model_sufix=model_sufix,model_type=model_type)
        line_num_points_hmod=line_num_points_hmod or self.line_num_points_hmod
        line_num_points_inter_hmod=line_num_points_inter_hmod or self.line_num_points_inter_hmod
        file_path_model_data = file_path_model_data or self.file_path_model_data
        data_studied = data_studied or self.data_studied 
        spline_smooth_hmod=self.spline_smooth_hmod
 
 
        hidden_layers=hidden_layers or self.hidden_layers 
        neurons_per_layer=neurons_per_layer or self.neurons_per_layer 
        activation_init=activation_init or self.activation_init
        activation_hidden=activation_hidden or self.activation_hidden
        activation_last=activation_last or self.activation_last
 
        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default'])
  
        loss_save_dir = loss_save_dir or model_dirs['loss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou'] 
        auc_save_dir = auc_save_dir or model_dirs['auc']
        dice_save_dir = dice_save_dir or model_dirs['dice']
        index_save_dir = index_save_dir or  model_dirs['index_save'] 
        model_dir = model_dir or  model_dirs['model']  
   

 
        curv,rhs,adj,neld =self.get_train_input_pnet(   
                                path_train=path_train,  
                                pre_portion=pre_portion, 
                                line_num_points_hmod=line_num_points_hmod,
                                line_num_points_inter_hmod=line_num_points_inter_hmod, 
                                model_sufix=model_sufix, 
                                list_features=list_features,
                                base_features_list=base_features_list, 
                                model_type=model_type,
                                num_sub_nodes=num_sub_nodes, 
                                thre_target_number_of_triangles=thre_target_number_of_triangles,
                                voxel_resolution=voxel_resolution,
                                dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                                tf_train=True,
                                entry_names=entry_names,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                                ) 
        ls =[i for i in ls if i in range(len(curv))]
        curv_train = [curv[i] for i in ls]
        rhs_train = [tf.expand_dims(tf.convert_to_tensor(rhs[i]), axis=0) for i in ls] 
        neld_train=[neld[i] for i in ls] 
        lss=np.arange(len(curv),dtype=int) 
        adj_train =[] 

        from neld_fun_0.help_pnet_one_hot import LOSS,aka_train,Get_iou,model_choice,get_auc
        with tf.device(device):
            mchoice = model_choice()
            model = mchoice.get_model(model_type=model_type, )

                 
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [1e-2, 1e-3, 5e-4])
        optimizer = tf.optimizers.Adam(learning_rate=lr) 
        if new_model:
            print(f"I'm starting a new model")
            print('--------------------------------')
            print(f"Model Type Gen : {model_type}")
            print(f"Model Type     : {model_sufix}")
            print(f"Model Dir.     : {model_dir}")
            print(f"Model Portion  : {pre_portion}")
            print(f"Data Dir.      : {path_train['data_smod_path']}") 
            print(f"Base Feat. list: {[mm for mm in base_features_list] if base_features_list is not None else list(self.base_features_dict.keys())}") 
            print(f"Feat. list     : {[mm for _,mm in list_features]}") 
            print(f"Feature Size   : {curv[0].shape[1]} * {len(curv)}")
            print(f"Target Size    : {rhs[0].shape[1]} * {len(rhs)}")
            print(f"neld Trainer size    : {[len(fv) for fv in curv_train]}")
            print(f"Hidden Layers        : {hidden_layers}")
            print(f"neurons_per_layer    : {neurons_per_layer}")
            print('--------------------------------')
            loss_save = []
            iou_save = {0: [], 1: [], 2: []}
            auc_save = {0: [], 1: [], 2: []}
            loss_tmp=10**10
            head_tmp=0
        else:
            print("This is a continuation of the previous model") 
            loss_save = np.loadtxt(loss_save_dir, dtype=float).tolist()
            iou_save = np.loadtxt(iou_save_dir, dtype=float).tolist() 
            auc_save = np.loadtxt(auc_save_dir, dtype=float).tolist() 
            
        print(f"Using device: {device}------------------ ----------------------------------------------")
        aka_train_ = aka_train()

        
        indd=[[np.where(rhs[i][:,label]==1)[0] for label in range(rhs[i].shape[1])] for i in lss ] 
        fun = LOSS(rhs=rhs_train , 
                   curv=curv_train , 
                   weight=weight,#tf.cast(weight,dtype=DTYPE), 
                   loss_mode=loss_mode,
                   adj=adj_train,
                   dtype=DTYPE,
                   neld=neld_train, 
                   rl_par= rl_par, 
                   dnn_par= dnn_par,
                   )
        
        tf.random.set_seed(0)
        iou_tmp=[0,0,0];auc_tmp=[0,0,0];index_save=[]
        
        pbar = tqdm(range(itime), desc=f"Loss: {loss_tmp:.6f}| IoU: N/A")  
        for i in pbar: 
            loss = aka_train_.train_PINN(optimizer, fun, model)
            iou=Get_iou(model, 
                        curv, 
                        lab=indd,
                        adj=adj, 
                   neld=neld,)
            auc=get_auc(model, curv, rhs=rhs)
            for ii in range(rhs[0].shape[1]):
                iou_save[ii].append(iou[ii]) 
                np.savetxt(iou_save_dir[ii] , np.array(iou_save[ii]), fmt='%f') 
                auc_save[ii].append(auc[ii]) 
                np.savetxt(auc_save_dir[ii] , np.array(auc_save[ii]), fmt='%f') 
            loss_save.append(loss.numpy())
            np.savetxt(loss_save_dir, np.array(loss_save), fmt='%f') 
            # if loss.numpy() < loss_tmp:
            if auc[0] > auc_tmp[0]:
                loss_tmp=loss.numpy()
                model.save(model_dir) 
                iou_tmp=iou  
                auc_tmp=auc

                index_save.append(i)
                np.savetxt(index_save_dir, np.array(index_save), fmt='%d')  
            if pre_portion=='neck_head':
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU sh: {min(iou_tmp[0]):.4f} IoU nk: {min(iou_tmp[1]):.4f} IoU hd: {min(iou_tmp[2]):.4f}")
            elif pre_portion=='smod':
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU sh: {min(iou_tmp[0]):.4f} IoU sp: {min(iou_tmp[1]):.4f} |  AUC sh: {auc_tmp[0]:.4f} AUC sp: {auc_tmp[1]:.4f}")
            else:
                pbar.set_description(f"Loss: {loss_tmp:.6f} ")
   
 
    def get_hmod_pred_pnet(self, 
                        path_train=None, 
                        pre_portion=None, 
                        n_col=None,
                        hidden_layers=None, 
                        neurons_per_layer=None, 
                        activation_init=None,
                        activation_hidden=None,
                        activation_last=None, 
                        model= None,
                        curv=None,
                        size_threshold=None,
                        neld_names=None, 
                        model_dir=None, 
                        loss_save_dir=None,
                        iou_save_dir=None,
                        model_sufix=None, 
                        data_studied=None,
                        file_path_org=None,  
                        line_num_points_hmod=None,
                        line_num_points_inter_hmod=None,
                        spline_smooth_hmod=None ,  
                        disp_infos=None, 
                        DTYPE=None,
                        list_features=None,
                        base_features_list=None,
                        hmod_thre=None,
                        train_smods=False,
                        weight=None,
                        weight2=None,
                        weights=None,
                        neck_lim=None,
                        n_clusters=3,  
                        smooth_tf=False,
                        model_type=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        reconstruction_tf=False, 
                        dict_mesh_to_skeleton_finder=None,
                        dict_wrap=None,
                        tf_skl_hmod_distance=False,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution
        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features=list_features if list_features is not None else self.list_features
        size_threshold=size_threshold or self.size_threshold
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        disp_infos = disp_infos or self.disp_infos  
        file_path_org = file_path_org or self.file_path_org
        model_sufix=model_sufix or self.model_sufix
        self.get_model_opt_name(model_sufix=model_sufix,
                                model_type=model_type,)  
        line_num_points_hmod = line_num_points_hmod or self.line_num_points_hmod
        line_num_points_inter_hmod = line_num_points_inter_hmod or self.line_num_points_inter_hmod
        spline_smooth_hmod = spline_smooth_hmod or self.spline_smooth_hmod
        DTYPE=DTYPE or self.DTYPE 
        data_studied = data_studied or self.data_studied 

        hidden_layers=hidden_layers or self.hidden_layers 
        neurons_per_layer=neurons_per_layer or self.neurons_per_layer 
        activation_init=activation_init or self.activation_init
        activation_hidden=activation_hidden or self.activation_hidden
        activation_last=activation_last or self.activation_last

        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default'])
  
        loss_save_dir = loss_save_dir or model_dirs['loss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou'] 
        model_dir = model_dir or  model_dirs['model']   

        rhs_name='rhs_name'
        n_col=n_col or len(self.model_dir_path[pre_portion][rhs_name]) 
        print('-------------',n_col,model_dir)

 
        if model_type.startswith('pnet'):
            from neld_fun_0.help_pnet_one_hot import model_choice  

        with tf.device(device):
            mchoice = model_choice(model_type=model_type)
            custom = mchoice.get_custom_objects(model_type) 
            model = load_model(model_dir, custom_objects=custom)



        print(f'get_hmod_pred started')
        print('--------------------------------')
        print(f"Model Type Gen : {model_type}")
        print(f"Model Type     : {model_sufix}")
        print(f"Model Dir.     : {model_dir}")
        print(f"Model Portion  : {pre_portion}") 
        print(f"Data Dir.      : {path_train['data_smod_path']}") 
        print(f"Destin. Dir.   : {path_train['dest_smod_path']}")  
        print(f"Feat. list     : {[mm for _,mm in list_features]}")
        print('--------------------------------')
        time_start = time.time()
        rhs_name='rhs_name'
        neld_names = neld_names or self.neld_names 
        for  index,neld_name in enumerate(neld_names):
            self.get_neld_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type, )  
            neld_name=self.neld_name  
            smod_portion_path=self.path_file[path_train['data_smod_path']] 
            hmod_portion_path=self.path_file[path_train['data_hmod_path']] 
            pid = pinn_data(file_path=self.file_path,
                            file_path_feat=self.file_path_feat,
                            path_file=self.path_file,
                            smod_path = smod_portion_path,
                            hmod_path =  hmod_portion_path , 
                            smod_path_pre=smod_portion_path,
                            hmod_path_pre=  hmod_portion_path ,  
                            neld_path_original_m=self.neld_path_original_m,
                            neld_first_name=self.neld_namess[index][1],
                            model_sufix=model_sufix,
                            path_train=path_train,
                            line_num_points_hmod=line_num_points_hmod,
                            line_num_points_inter_hmod=line_num_points_inter_hmod,
                            spline_smooth_hmod=spline_smooth_hmod,
                            thre_target_number_of_triangles=thre_target_number_of_triangles,
                            voxel_resolution=voxel_resolution,
                            dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                        )  
            key,tyy=f'{self.model_type}_{self.model_sufix}_{self.path_dir}', self.intensity_logit_dict[pre_portion][0]  
            path_ex = os.path.join(self.path_file_sub[tyy][key]) 
            if os.path.exists(path_ex) and not param_dic['tf_restart']['get_hmod_pred']: 
                continue
            pid.save_pinn_data() 
            pid.get_neld_data()
            con=0

            curv,_,_,_ =self.get_train_input_pnet(   
                                    neld_names=[neld_name],
                                    path_train=path_train,  
                                    pre_portion=pre_portion, 
                                    line_num_points_hmod=line_num_points_hmod,
                                    line_num_points_inter_hmod=line_num_points_inter_hmod, 
                                    model_sufix=model_sufix, 
                                    list_features=list_features,
                                    base_features_list=base_features_list, 
                                    model_type=model_type,
                                    num_sub_nodes=num_sub_nodes, 
                                    thre_target_number_of_triangles=thre_target_number_of_triangles,
                                    voxel_resolution=voxel_resolution,
                                    dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                                    tf_train=False,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                                    )   
            rhs0=model(curv[0]).numpy()[0,...] 

            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'  
            for kk,tyy in enumerate(self.intensity_logit_dict[pre_portion]):
                np.savetxt(self.path_file_sub[tyy][key], rhs0[:,kk], fmt='%f') 
 
            mytime0 = time.time() - time_start
 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60)
            if disp_infos:
                print(f'hmod Prediction completed on {neld_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
                print(f'data stores in: {smod_portion_path}')  
        print('hmod Prediction completed') 





 




    def model_shap(self, 
                        neld_names_ls=[0,1],
                        n_shap=25,  
                        path_train=None, 
                        pre_portion=None, 
                        n_col=None,
                        hidden_layers=None, 
                        neurons_per_layer=None, 
                        activation_init=None,
                        activation_hidden=None,
                        activation_last=None, 
                        model= None,
                        curv=None,
                        size_threshold=None,
                        neld_names=None, 
                        model_dir=None, 
                        loss_save_dir=None,
                        iou_save_dir=None,
                        model_sufix=None, 
                        data_studied=None,
                        file_path_org=None,  
                        line_num_points_hmod=None,
                        line_num_points_inter_hmod=None,
                        spline_smooth_hmod=None ,  
                        disp_infos=None, 
                        DTYPE=None,
                        list_features=None,
                        base_features_list=None,
                        hmod_thre=None,
                        train_smods=False,
                        weight=None,
                        weights=None,
                        neck_lim=None,
                        n_clusters=3,  
                        smooth_tf=False,
                        model_type=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        reconstruction_tf=False, 
                        dict_mesh_to_skeleton_finder=None,
                        dict_wrap=None,
                        tf_skl_hmod_distance=False,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution
        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features=list_features if list_features is not None else self.list_features
        size_threshold=size_threshold or self.size_threshold
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        disp_infos = disp_infos or self.disp_infos  
        file_path_org = file_path_org or self.file_path_org
        model_sufix=model_sufix or self.model_sufix
        self.get_model_opt_name(model_sufix=model_sufix,
                                model_type=model_type,)  
        line_num_points_hmod = line_num_points_hmod or self.line_num_points_hmod
        line_num_points_inter_hmod = line_num_points_inter_hmod or self.line_num_points_inter_hmod
        spline_smooth_hmod = spline_smooth_hmod or self.spline_smooth_hmod
        DTYPE=DTYPE or self.DTYPE 
        data_studied = data_studied or self.data_studied 

        hidden_layers=hidden_layers or self.hidden_layers 
        neurons_per_layer=neurons_per_layer or self.neurons_per_layer 
        activation_init=activation_init or self.activation_init
        activation_hidden=activation_hidden or self.activation_hidden
        activation_last=activation_last or self.activation_last

        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default'])
  
        loss_save_dir = loss_save_dir or model_dirs['loss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou'] 
        model_dir = model_dir or  model_dirs['model']    

        rhs_name='rhs_name'
        n_col=n_col or len(self.model_dir_path[pre_portion][rhs_name]) 
        print('-------------',n_col,model_dir)
        import shap  

        # feat_name=[] 
        # for  mm in base_features_list:
        #     feat_name.append(mm)
        # for _,name_inten in list_features: 
        #     feat_name.append(name_inten.removesuffix('.txt') )


 
 
                
        rhs_name='rhs_name'
        n_col=  len(self.model_dir_path[pre_portion][rhs_name]) 
        # print('-------------',n_col,model_dir)

        if  model_type.startswith(('cml','ml',)):
            with open(model_dir, 'rb') as f:
                model = pickle.load(f)  

        else:
            model_type_l=model_type
            if model_type.startswith(('gcn',)):
                from neld_fun_0.help_gcn_one_hot import model_choice
                adj_tf=True 
            elif model_type.startswith('dnn'):
                from neld_fun_0.help_dnn_one_hot import model_choice 
                adj_tf=False 
            elif model_type.startswith(('cnn','vol',)):
                from neld_fun_0.help_vol_one_hot import model_choice 
                mmnn=['vol']
                mmnn.extend(model_type.split('_')[1:])
                model_type_l='_'.join(mmnn)
                print('[[[[[[[]]]]]]]',model_type_l) 
                adj_tf=False
            mchoice = model_choice()
            custom = mchoice.get_custom_objects(model_type_l) 
            model = load_model(model_dir, custom_objects=custom)


        feat_name=[] 
        for  mm in base_features_list:
            feat_name.append(mm)
        for _,name_inten in list_features: 
            feat_name.append(name_inten.removesuffix('.txt') )


 


        print(f'SHAP started')
        print('--------------------------------')
        print(f"Model Type     : {model_sufix}")
        print(f"Model Dir.     : {model_dir}")
        print(f"Model Portion  : {pre_portion}") 
        print(f"Data Dir.      : {path_train['data_smod_path']}") 
        print(f"Destin. Dir.   : {path_train['dest_smod_path']}") 
        print(f"Base Feat. list: {[mm for mm in base_features_list] if base_features_list is not None else list(self.base_features_dict.keys())}") 
        print(f"Feat. list     : {[mm for _,mm in list_features]}")
        print('--------------------------------')
        time_start = time.time()
        time_start = time.time()
        rhs_name='rhs_name'
        neld_names = neld_names or [self.neld_names[jj] for jj in neld_names_ls ]
        base_features=[]
        for  index,neld_name  in enumerate( neld_names):
            if index>2:
                break
            self.get_neld_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type, )  
            neld_name=self.neld_name  
            smod_portion_path=self.path_file[path_train['data_smod_path']] 
            hmod_portion_path=self.path_file[path_train['data_hmod_path']] 
            pid = pinn_data(file_path=self.file_path,
                            file_path_feat=self.file_path_feat,
                            path_file=self.path_file,
                            smod_path = smod_portion_path,
                            hmod_path =  hmod_portion_path , 
                            smod_path_pre=smod_portion_path,
                            hmod_path_pre=  hmod_portion_path ,  
                            neld_path_original_m=self.neld_path_original_m,
                            neld_first_name=self.neld_namess[index][1],
                            model_sufix=model_sufix,
                            path_train=path_train,
                            line_num_points_hmod=line_num_points_hmod,
                            line_num_points_inter_hmod=line_num_points_inter_hmod,
                            spline_smooth_hmod=spline_smooth_hmod,
                            thre_target_number_of_triangles=thre_target_number_of_triangles,
                            voxel_resolution=voxel_resolution,
                            dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                            neld_path_true_final=self.neld_path_true_final,
                        )   
            pid.save_pinn_data() 
            pid.get_neld_data()
            con=0
            parami=dict(   
                        neld_names=[neld_name],
                        path_train=path_train,  
                        pre_portion=pre_portion, 
                        line_num_points_hmod=line_num_points_hmod,
                        line_num_points_inter_hmod=line_num_points_inter_hmod, 
                        model_sufix=model_sufix, 
                        list_features=list_features,
                        base_features_list=base_features_list, 
                        model_type=model_type,
                        num_sub_nodes=num_sub_nodes, 
                        thre_target_number_of_triangles=thre_target_number_of_triangles,
                        voxel_resolution=voxel_resolution,
                        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                        tf_train=False,
                        kmean_n_run=kmean_n_run,
                        kmean_max_iter=kmean_max_iter,
                        param_dic=param_dic,
                        ) 

            if  model_type.startswith(('cml','ml',)):
                curv,_,_,_,_=self.get_train_input_cml(**parami)  
                curvd=curv[0]
            else:
                if model_type.startswith(('gcn',)):
                    curv,_,_,_,_=self.get_train_input_gcn(**parami) 
                    curvd=curv[0] 
                elif model_type.startswith('dnn'):
                    curv,_,_,_,_=self.get_train_input_dnn(**parami) 
                    curvd=curv[0]  
                elif model_type.startswith(('cnn','vol',)):
                    curv,rhs,mskls,smod_indexs,idx_originals,weight_mask,rhshr  =self.get_train_input_volume(**parami)  
                    curvd=curv[0].numpy()
            base_features.append(curvd)
            # rhs0=model(curv[0]).numpy() [0] 
        # print([vd.shape for vd in base_features])
        n_shap = n_shap or min([150, base_features[1].shape[0]])
        niuu = min([base_features[0].shape[0] // 100, base_features[0].shape[0]]) 
        explain = shap.Explainer(model.predict, base_features[0][:niuu, :]) 
        shap_values = explain(base_features[1][:n_shap, :]) 
        raw_values = shap_values.values 

        '''
        n_shap = n_shap or min([150, base_features[1].shape[0]])
        niuu = min([base_features[0].shape[0] // 100, base_features[0].shape[0]])

        background = base_features[0][:niuu, :]

        masker = shap.maskers.Independent(background)
        explain = shap.Explainer(model.predict, masker)

        shap_values = explain(base_features[0][:n_shap, :])
        raw_values = shap_values.values
'''



        mean_shap = np.abs(raw_values).mean(axis=0)
        mean_shap = mean_shap if mean_shap.ndim == 2 else mean_shap.reshape(-1, 1) 
        feat = {'Feature': feat_name}
        for idx in range(mean_shap.shape[1]):
            feat[f'Mean SHAP Values {idx+1}'] = mean_shap[:, idx]

        shap_df = pd.DataFrame(feat).sort_values(by='Mean SHAP Values 1', ascending=False)
        # shap_df.to_csv(self.shap_dir, index=False)


        head_neck_path = 'dest_hmod_path'
        path=path_train[head_neck_path]
        smod_path_save=     self.path_file[f'result_{path}']   
        shap_df.to_csv(os.path.join(smod_path_save,'shap.csv') , index=False)







