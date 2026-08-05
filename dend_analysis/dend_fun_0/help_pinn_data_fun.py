

import sys
import os
import numpy as np
import dend_fun_0.help_funn as hff  
from scipy.interpolate import splprep, splev 

from scipy.spatial import  distance_matrix
import networkx as nx 

from sklearn.neighbors import KDTree 
  

def get_model(base_features ,vcv_length, model_sufix, add_param=None ): 
    # Ensure inputs are numpy arrays
    vcv_length =np.asarray(vcv_length).reshape(-1, 1)

    if model_sufix.startswith("opt"):
        base_features.append( hff.normalize(vcv_length) )

    if model_sufix == "opt_mean":
        base_features.append(vcv_length - np.mean(vcv_length)) 

    if model_sufix == "opt_std":
        base_features.append((vcv_length - np.mean(vcv_length)) / np.std(vcv_length))

    if model_sufix == "opt_add_param" and add_param is not None:
        add_param = np.asarray(add_param).reshape(-1, 1)
        base_features.append(add_param)
 
    print(f'I activated {model_sufix}')
    return base_features 
 
def Curve_length(points):  
    return np.sum(np.linalg.norm(np.diff(np.array(points) , axis=0), axis=1) )  

 
class Sorted_distance_matrix:
    def __init__(self, vert_0: np.ndarray, vert_1: np.ndarray) -> None:
        self.vert_0, self.vert_1 = vert_0, vert_1
        self.distances = distance_matrix(self.vert_0, self.vert_1).flatten()
        self.sorted_indices = np.argsort(self.distances)
        
        ind_y, ind_x = np.meshgrid(range(self.vert_0.shape[0]), range(self.vert_1.shape[0]))
        self.ind_x = ind_x.flatten()[self.sorted_indices]
        self.ind_y = ind_y.flatten()[self.sorted_indices]
 
        self.vert_sorted_0 = self.vert_0[self.ind_x, :]
        self.vert_sorted_1 = self.vert_1[self.ind_y, :]
 
    def Min(self):
        self.vert_min_0,self.vert_min_1=self.vert_0[self.ind_x[0],:],self.vert_1[self.ind_y[0],:]
 
    def Max(self):
        self.vert_max_0,self.vert_max_1=self.vert_0[self.ind_x[-1],:],self.vert_1[self.ind_y[-1],:]
        

def get_aligned_points(vertices, line_num_points=100, spline_smooth=0): 

    G = nx.Graph() 
    for i, pos in enumerate(vertices):
        G.add_node(i, pos=pos)
     
    for i in range(len(vertices)-1):
        dist = np.linalg.norm(vertices[i] - vertices[i+1])
        G.add_edge(i, i+1, weight=dist)
     
    path = list(nx.dfs_preorder_nodes(G, 0))
    path_points = vertices[path]
     
    if vertices.shape[0] > 2: 
        k = max(1, min(3, vertices.shape[0] - 1)) 
        tck, _ = splprep([path_points[:, 0],
                         path_points[:, 1],
                         path_points[:, 2]],
                        s=spline_smooth,
                        k=k,
                        )
         
        u = np.linspace(0, 1, line_num_points)
        aligned_points = np.array(splev(u, tck)).T
         
        curve_length = Curve_length(aligned_points)
        
        return aligned_points, curve_length
     
    else:
        aligned_points = np.linspace(vertices[0], vertices[-1], line_num_points)
        curve_length = Curve_length(aligned_points)
        return aligned_points, curve_length
  

import pickle
from dend_fun_0.get_path import get_name,get_param


class pinn_data(get_name,get_param):
    def __init__(self,file_path,
                #  txt_save_file,
                hmod_path,
                smod_path,
                dend_path_original_m,
                dend_first_name,
                name_smod_id=None,
                model_sufix=None,
                hmod_path_pre=None,
                smod_path_pre=None,
                path_mapping=None,
                path_pre=None,
                path_train=None,
                path_file=None,
                pre_portion=None,
                line_num_points_hmod=200,
                line_num_points_inter_hmod=300, 
                spline_smooth_hmod=1, 
                thre_target_number_of_triangles=None,
                voxel_resolution=None, 
                file_path_feat=None,
                dict_mesh_to_skeleton_finder_mesh=None,
                        kmean_n_run=60,
                        kmean_max_iter=600,
                        param_dic=None,
                        dend_path_true_final=None,
                        ):
        get_name.__init__(self)  
        get_param.__init__(self,
                         line_num_points_hmod=line_num_points_hmod,
                         line_num_points_inter_hmod=line_num_points_inter_hmod, 
                         spline_smooth_hmod=spline_smooth_hmod,
                        thre_target_number_of_triangles=thre_target_number_of_triangles,
                        voxel_resolution=voxel_resolution, ) 
        # self.txt_save_file=txt_save_file
        self.hmod_path=hmod_path
        self.smod_path=smod_path
        self.file_path=file_path
        self.hmod_path_pre=hmod_path_pre
        self.smod_path_pre=smod_path_pre
        self.dend_path_original_m=dend_path_original_m
        self.dend_first_name=dend_first_name
        self.name_smod_id=name_smod_id
        self.model_sufix=model_sufix
        self.path_mapping=path_mapping
        self.path_pre=path_pre
        self.path_train=path_train
        self.path_file=path_file
        self.pre_portion=pre_portion
        self.file_path_feat=file_path_feat 
        self.dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh 
        self.kmean_n_run=kmean_n_run
        self.kmean_max_iter=kmean_max_iter
        self.param_dic=param_dic
        self.dend_path_true_final=dend_path_true_final
        pass
 
    def get_intensity_head_neck(self, 
                        dend_path_true_final=None,
						dend_first_name=None,
						smod_path=None,
                        hmod_path=None,
						file_path=None,
						dend_path_original_m=None, 
						radius_threshold=None, 
						disp_infos=None,
						size_threshold=None,
						):
        disp_infos=disp_infos or self.disp_infos
        file_path  = file_path or self.file_path
        smod_path = smod_path or self.smod_path
        hmod_path = hmod_path or self.hmod_path
        dend_path_original_m=dend_path_original_m or self.dend_path_original_m 
        radius_threshold = radius_threshold or self.radius_threshold
        size_threshold=size_threshold or self.size_threshold
        dend_first_name=dend_first_name or self.dend_first_name
        dend_path_true_final=dend_path_true_final or self.dend_path_true_final
        if disp_infos:
            print(f"get_intensity_head_neck: {file_path}")  
        vertices_00 = np.loadtxt(os.path.join(file_path, self.txt_vertices_old), dtype=float)

        intensity  =  np.zeros(vertices_00.shape[0]) 
        intensity_1hot=np.zeros_like(vertices_00[:,:-1],dtype=int)
        count= hff.loadtxt_count(os.path.join(smod_path,self.txt_smod_count)) 
        skeleton_points=[]
        mmm=count.ndim 
        smod_index_all=[]
        count=count if mmm==2 else count.reshape(-1,1) 
        for i in range(count.shape[0]): 
            ii=count[i,0]
            if ii <0:
                continue
            name=f'{ii}_{count[i,1]}' if mmm==2 else f'{count[i,0]}'  
            smod_index = np.loadtxt(os.path.join(smod_path, f'{self.name_smod_index}_{name}.txt'),dtype=int) 
            intensity[smod_index]=1
            smod_index_all.extend(smod_index) 
        intensity_1hot[:,1:2][smod_index_all]=1 
        intensity_1hot[:,0:1][list(set(np.arange(vertices_00.shape[0]))-set(smod_index_all))]=1
        np.savetxt(os.path.join(dend_path_true_final,'intensity_hmod_smod.txt'), intensity, fmt='%d')
        np.savetxt(os.path.join(dend_path_true_final,'intensity_1hot_hmod_smod.txt'), intensity_1hot, fmt='%d')
 





    def get_smod_group(self,smod_path):  
        self.get_dend_data()    
        count= hff.loadtxt_count( os.path.join(smod_path,self.txt_smod_count))
        smod_group=[]
        for clustss in count:
            smod_index = np.loadtxt(os.path.join(smod_path, f'{self.name_smod}_{self.name_index}_{clustss}.txt'),dtype=int) 
            smod_group.extend(smod_index)

        self.cclu.Cluster_index(ln_elm= smod_group)
        self.cclu.Cluster_faces()
        self.cclu.Cluster_faces_unique()   
        if len(self.cclu.cluster_index)>self.size_threshold:
            np.savetxt(os.path.join(smod_path,f'{self.name_smod}_group_{self.name_index}.txt'),self.cclu.cluster_index, fmt='%d')
            np.savetxt(os.path.join(smod_path,f'{self.name_smod}_group_{self.name_faces}.txt'),self.cclu.cluster_faces, fmt='%d')
            np.savetxt(os.path.join(smod_path,f'{self.name_smod}_group_index_unique.txt'),self.cclu.cluster_faces_unique, fmt='%d')



 

    def get_pinn_rhs(self,
                        pre_portion,
						file_path=None, 
                        file_path_feat=None,
                        dend_path_true_final=None,
                        radius_threshold=.03): 
        file_path = file_path or self.file_path 
        file_path_feat=file_path_feat or self.file_path_feat
        dend_path_true_final=dend_path_true_final or self.dend_path_true_final
 
        if os.path.exists(os.path.join(file_path, self.txt_vertices_0)):
            vertices_0  = np.loadtxt(os.path.join(file_path, self.txt_vertices_0), dtype=float)
            vertices_0 -= np.mean(vertices_0, axis=0)
            self.vertices_0=vertices_0
        print('[[[[[[[pre_portion]]]]]]]',pre_portion)
        hmod_path_tmp=os.path.join(dend_path_true_final,f'intensity_1hot_hmod_{pre_portion}.txt')
        # if pre_portion=='head_neck': 
        #     hmod_path_tmp=os.path.join(dend_path_true_final,'intensity_1hot_hmod_neck_head.txt')
        #     if not os.path.exists(hmod_path_tmp):
        #         self.get_intensity_head_neck(radius_threshold=radius_threshold) 
        # elif pre_portion=='smod':  
        #     hmod_path_tmp=os.path.join(dend_path_true_final,'intensity_1hot_hmod_smod.txt')
        #     if not os.path.exists(hmod_path_tmp):
        #         self.get_intensity_head_neck(radius_threshold=radius_threshold)
        # elif pre_portion=='head':  
        #     hmod_path_tmp=os.path.join(dend_path_true_final,'intensity_1hot_hmod_head.txt')
        #     if not os.path.exists(hmod_path_tmp):
        #         self.get_intensity_head_neck(radius_threshold=radius_threshold)
        # elif pre_portion=='neck':  
        #     hmod_path_tmp=os.path.join(dend_path_true_final,'intensity_1hot_hmod_neck.txt')
        #     if not os.path.exists(hmod_path_tmp):
        #         self.get_intensity_head_neck(radius_threshold=radius_threshold)

        return np.loadtxt(hmod_path_tmp,dtype=int) 
 