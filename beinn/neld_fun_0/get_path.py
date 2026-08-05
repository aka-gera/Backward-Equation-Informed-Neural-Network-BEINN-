 
import os
import shutil
import stat
import pickle

def remove_directory(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.chmod(file_path, stat.S_IWRITE)   
            os.remove(file_path) 
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.chmod(dir_path, stat.S_IWRITE)
            os.rmdir(dir_path) 
    shutil.rmtree(path, ignore_errors=True)  

def assign_if_none(self, **kwargs):
    for arg, value in kwargs.items():
        if value is None: 
            value = getattr(self, arg, None) 
        setattr(self, arg, value)
  
 
class get_name:
    def __init__(self): 
        self.txt_vertices_old ='vertices_old.txt'
        self.txt_vertices_0 = 'vertices_0.txt'
        self.txt_vertices_1 = 'vertices_1.txt'
        self.txt_faces = 'faces_1.txt' 
        self.name_count = 'count'
        self.name_smod = 'smod'
        self.name_hmod = 'hmod'
        self.name_head = 'head'
        self.name_neck = 'neck'
        self.name_smod_group = 'smod_group'
        self.name_neld = 'neld'
        self.name_index = 'index'
        self.name_faces = 'faces'
        self.name_intensity = 'intensity'
        self.name_count = 'count'
        self.name_centroid = 'centroid'
        self.name_centroid_curve = 'centroid_curve_vert'
        self.name_centr = 'centr'
        self.name_hmod_iou = 'hmod_iou'
        self.name_center='center'
        self.name_length='length'
        self.txt_intensity_smods_segment='intensity_smods_segment.txt'
        self.txt_neld_org_mean='neld_init_mean.txt'
        self.txt_neld_org_gauss='neld_init_gauss.txt'   


        self.name_smod_new =f'{self.name_smod}_new'
        self.txt_smod_new =f'{self.name_smod}_new.txt'
        self.name_smod_new_index=f'{self.name_smod_new}_{self.name_index}'
        self.name_smod_new_face=f'{self.name_smod_new}_{self.name_faces}'

        self.txt_smod_new_intensity=f'{self.name_smod_new}_{self.name_intensity}.txt'

        self.txt_neld_smooth_mean='neld_smooth_mean.txt'
        self.txt_neld_smooth_gauss='neld_smooth_gauss.txt'

        self.txt_smod_intensity=f'{self.name_smod}_{self.name_intensity}.txt'
        self.txt_head_intensity= f'{self.name_head}_{self.name_intensity}.txt'
        self.txt_neck_intensity= f'{self.name_neck}_{self.name_intensity}.txt'
        self.txt_smod_volume='smod_volume.txt'

        self.name_smod_metric=f'{self.name_smod}_metric'
        self.name_head_metric=f'{self.name_head}_metric'
        self.name_neck_metric=f'{self.name_neck}_metric'
        self.name_hmod_metric=f'{self.name_hmod}_metric'

        self.txt_smod_metric=f'{self.name_smod_metric}.txt'
        self.txt_head_metric=f'{self.name_head_metric}.txt'
        self.txt_neck_metric=f'{self.name_neck_metric}.txt'
        self.txt_hmod_metric=f'{self.name_hmod_metric}.txt'
        self.txt_metric=f'metric.txt'

        self.txt_smod_center_length=f'{self.name_smod}_{self.name_center}_{self.name_length}.txt'
        self.txt_smod_hmod_length=f'{self.name_smod}_{self.name_hmod}_{self.name_length}.txt'
        self.txt_hmod_init_index=f'{self.name_hmod}_init_{self.name_index}.txt' 
        self.txt_hmod_init_faces=f'{self.name_hmod}_init_{self.name_faces}.txt'
        self.txt_hmod_init_index_unique=f'{self.name_hmod}_init_{self.name_index}_unique.txt'
        # Indexed names
        self.name_smod_index = f'{self.name_smod}_{self.name_index}'
        self.name_hmod_index = f'{self.name_hmod}_{self.name_index}'
        self.name_head_index = f'{self.name_head}_{self.name_index}'
        self.name_neck_index = f'{self.name_neck}_{self.name_index}'
        self.name_smod_group_index = f'{self.name_smod_group}_{self.name_index}'

        # Face names
        self.name_smod_faces = f'{self.name_smod}_{self.name_faces}'
        self.name_hmod_faces = f'{self.name_hmod}_{self.name_faces}'
        self.name_head_faces = f'{self.name_head}_{self.name_faces}'
        self.name_neck_faces = f'{self.name_neck}_{self.name_faces}'
        self.name_smod_group_faces = f'{self.name_smod_group}_{self.name_faces}'

        # Unique indices
        self.name_smod_group_index_unique = f'{self.name_smod_group_index}_unique'
        self.name_index_unique = f'{self.name_index}_unique'
        self.name_hmod_index_unique = f'{self.name_hmod_index}_unique'
        self.name_smod_group_faces_unique = f'{self.name_smod_group_faces}_unique'
        self.name_smod_index_unique = f'{self.name_smod_index}_unique'
        self.name_head_index_unique = f'{self.name_head_index}_unique'
        self.name_neck_index_unique = f'{self.name_neck_index}_unique'
        self.txt_hmod_index_unique = f'{self.name_hmod_index}_unique.txt'
        # Centroid-related names
        self.name_smod_centr = f'{self.name_smod}_{self.name_centr}'
        self.name_smod_centroid = f'{self.name_smod}_{self.name_centroid}'
        self.name_smod_centroid_curve = f'{self.name_smod}_{self.name_centroid_curve}'
        self.name_smod_count = f'{self.name_smod}_count'
        self.name_head_count = f'{self.name_head}_count'
        self.name_neck_count = f'{self.name_neck}_count'
        # File names
        self.txt_smod_count = f'{self.name_smod}_count.txt'
        self.txt_head_count = f'{self.name_head}_count.txt'
        self.txt_neck_count = f'{self.name_neck}_count.txt'
        self.txt_count = f'{self.name_count}.txt'
        self.txt_hmod_iou = f'{self.name_hmod_iou}.txt'
        self.txt_hmod_index = f'{self.name_hmod_index}.txt'
        self.txt_hmod_faces = f'{self.name_hmod_faces}.txt'
        self.txt_hmod_index_unique = f'{self.name_hmod_index_unique}.txt'
        self.txt_hmod_vertices_center = f'{self.name_hmod}_vertices_center.txt'
        
        self.txt_hmod_vertices_center_to_vertices_length = f'{self.name_hmod}_vertices_center_to_vertices_length.txt'
        self.txt_smod_count=f'{self.name_smod}_count.txt'
        self.txt_head_count=f'{self.name_head}_count.txt'
        self.txt_neck_count=f'{self.name_neck}_count.txt'



        self.txt_smod_iou='iou.txt'

        self.name_smod_div =f'{self.name_smod}_div'
        self.txt_smod_div_intensity  =f'{self.name_smod}_{self.name_intensity}_div.txt'
        self.txt_smod_new_intensity = f'{self.name_smod}_new_intensity.txt'
        self.txt_hmod_intensity = f'{self.name_hmod}_intensity.txt' 
        self.txt_smod_count_name =f'{self.name_smod_count}_name.txt'
        self.txt_hmod_vertices_center='hmod_vertices_center.txt'
        self.txt_hmod_vertices_center_to_vertices_length='hmod_vertices_center_to_vertices_length.txt'

        self.name_vcv_length = 'vcv_length'
        self.txt_neld_vcv_length=f'{self.name_neld}_{self.name_vcv_length}.txt'
        self.name_smod_vcv_length = f'{self.name_smod}_{self.name_vcv_length}'
        self.name_head_vcv_length = f'{self.name_head}_{self.name_vcv_length}'
        self.name_neck_vcv_length = f'{self.name_neck}_{self.name_vcv_length}'
        self.txt_hmod_vcv_length = f'{self.name_hmod}_{self.name_vcv_length}.txt' 
        self.txt_hmod_vcv_vertices_center = f'{self.name_hmod}_vcv_vertices_center.txt'
        self.txt_hmod_vcv_length_improved=f'hmod_vcv_length_improved.txt'
 
        self.txt_gauss_curv_init='gauss_curv_init.txt'
        self.txt_mean_curv_init='mean_curv_init.txt'
        self.txt_gauss_curv_smooth='gauss_curv_smooth.txt'
        self.txt_mean_curv_smooth='mean_curv_smooth.txt'
        self.txt_faces_class_faces='faces_class_faces.txt'
        self.txt_vertex_neighbor='vertex_neighbor.txt'
        self.txt_skl_distance='skl_distance.txt'
        self.txt_skl_vertices='skl_vertices.txt'
        self.txt_skl_index='skl_index.txt'

        self.txt_skl_distance_org='skl_distance_org.txt'
        self.txt_skl_vertices_org='skl_vertices_org.txt'
        self.txt_skl_index_org='skl_index_org.txt'

        self.txt_skl_distance_true='skl_distance_true.txt'
        self.txt_skl_vertices_true='skl_vertices_true.txt'
        self.txt_skl_index_true='skl_index_true.txt'
        
        
        self.txt_skl_distance_con='skl_distance_con.txt'
        self.txt_skl_vertices_con='skl_vertices_con.txt'
        self.txt_skl_index_con='skl_index_con.txt'

        self.txt_skl_hmod_distance='skl_hmod_distance.txt'
        self.txt_skl_hmod_vertices='skl_hmod_vertices.txt'

        self.txt_skl_hmod_distance_org='skl_hmod_distance_org.txt'
        self.txt_skl_hmod_vertices_org='skl_hmod_vertices_org.txt'

        self.txt_gauss_sq_curv_smooth='gauss_sq_curv_smooth.txt'
        self.txt_mean_sq_curv_smooth='mean_sq_curv_smooth.txt' 
        self.txt_gauss_sq_curv_init='gauss_sq_curv_init.txt'
        self.txt_mean_sq_curv_init='mean_sq_curv_init.txt' 


        self.pkl_vertex_neighbor='vertex_neighbor.pkl'
        self.pkl_mp='mp.pkl'

        gff=['skl', ]#,'mean_qd''mean_gauss','skl']
        gffindex=['skl',]  #22-23
 
        gfff=['curv_k','curv_v','curv_kv' ,'curv_k2','curv_v2','curv_kv22'] #23-28
        gfffindex=['x','v','xv' ,'x2','v2','x2v2']  

        gf=['gauss','mean','gauss_sq','mean_sq','gauss_qd','mean_qd']# 0-5   ,'mean_qd''mean_gauss','skl']
        gfindex=['g','m','g2','m2','g4','m4' ]


        gfi=['igauss','imean','igauss_sq','imean_sq']#29-32    ,'mean_qd''mean_gauss','skl']
        gfiindex=['ig','im','ig2','im2' ]
         
        self.kmean_list=[100,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] # 6-21
        gf2=[f'kmean_smooth_{rf}' for rf in self.kmean_list]
        gfindex2=[f'k{rf}' for rf in self.kmean_list] 
        gf22=[f'kmean_init_{rf}' for rf in self.kmean_list]# 33-48
        gfindex22=[f'n{rf}' for rf in self.kmean_list] 
        gffkk=['skl_hmod_distance', ]#,49
        gffkkindex=['sklsh',]  #49
        gfg=[]
        gkg=[]
        gfg.extend(gf)
        gfg.extend(gf2)
        gfg.extend(gff)
        gfg.extend(gfff)
        gfg.extend(gfi)
        gfg.extend(gf22)
        gfg.extend(gffkk)
        gkg.extend(gfindex)
        gkg.extend(gfindex2)
        gkg.extend(gffindex)
        gkg.extend(gfffindex)
        gkg.extend(gfiindex)
        gkg.extend(gfindex22)
        gkg.extend(gffkkindex)
        self.base_features_dict= {}
        for nam,nm in zip(gfg,gkg):
            self.base_features_dict[nam]={}
            self.base_features_dict[nam]['index']=nm

        self.inten_pinn=['hmod_vcv_length','skl_hmod_distance','smod_hmod_length','pca_1_norm','pca_2_norm','pca_3_norm','division','smod_intensity_division']
        self.inten_pinn_name=['vcv','shkl','sh','p1','p2','p3','di','sid']
        self.inten_pinn_dic={key:val for key,val in zip(self.inten_pinn,self.inten_pinn_name)}
        
        self.metrics_keys=['smod_vol','head_vol','neck_vol','smod_area','head_area','neck_area','head_diameter','neck_diameter','head_length','neck_length','smod_length']
 
        self.metrics={}
        for val in self.metrics_keys:
            self.metrics[val]={}

        self.metrics_combine={'diam_head_neck_length' :['head_diameter','neck_diameter','smod_length'],
                              'vol_head_neck_smod'   :['head_vol','neck_vol','smod_vol'],
                              'area_head_neck_smod'  :['head_area','neck_area','smod_area'],
                              'length_head_neck_smod':['head_length','neck_length','smod_length'],
                              }

        # self.file_model_train=['smod','head_neck','hmod']
        # self.path_heads=['pinn','rpinn','ML','true']
        self.pre_portions=['head_neck','smod']
        
        self.inten_file_model_train_smod_iou=[ 'iou_smod_sh','iou_smod_sp' ]
        self.inten_file_model_smod_iou=['model_sp_iou'] 
        self.inten_file_model_train_smod_loss=['loss_smod']
        self.inten_file_model_smod_loss=['model_sp_loss']
        self.inten_file_model_train_smod_auc=[ 'auc_smod_sh','auc_smod_sp' ]
        self.inten_file_model_train_smod_dice=[ 'dice_smod_sh','dice_smod_sp' ]
        self.inten_file_model_smod_auc=['model_sp_auc'] 
        self.inten_file_model_smod_dice=['model_sp_dice'] 

        self.inten_file_model_train_iou=['iou_head_neck_hd','iou_head_neck_nk','iou_head_neck_sh' ]
        self.inten_file_model_head_neck_iou=['model_hn_iou'] 
        self.inten_file_model_train_loss=['loss_head_neck']
        self.inten_file_model_head_neck_loss=['model_hn_loss']
 
        self.inten_file_model_train_shap=[ 'shap']
        self.inten_file_model_shap=['model_shap']
        self.inten_file_model_head_neck=[] 
        self.inten_file_model_head_neck.extend(self.inten_file_model_smod_iou)
        self.inten_file_model_head_neck.extend(self.inten_file_model_smod_loss)
        self.inten_file_model_head_neck.extend(self.inten_file_model_smod_auc)
        self.inten_file_model_head_neck.extend(self.inten_file_model_smod_dice)
        self.inten_file_model_head_neck.extend(self.inten_file_model_shap)
        self.inten_pca=['pca_1_norm','pca_2_norm','pca_3_norm','volume','area','energy','division',]
        # self.inten_file_sub=["path","smod_intensity",  "intensity_head_neck_segm",  'intensity_smods_segment','intensity_smods_segment_hmod','intensity_smods_logit_sh','intensity_smods_logit_sp','smod_annot','smod_match','smod_match_dice',]
        # self.inten_file_sub_name=["path","Segmentation","Head Neck Segm.", 'intensity_smods_segment','intensity_smods_segment_hmod','intensity_smods_logit_sh','intensity_smods_logit_sp','smod_annot','smod_match','smod_match_dice', ] 
        self.inten_file_sub=["path","smod_intensity",  "intensity_head_neck_segm",  'intensity_smods_segment','intensity_smods_segment_hmod','smod_annot','smod_match','smod_match_dice','hmod_vcv_length',]
        self.inten_file_sub_name=["path","Segmentation","Head Neck Segm.", 'intensity_smods_segment','intensity_smods_segment_hmod', 'smod_annot','smod_match','smod_match_dice','length sh. skl. vert.', ] 
        self.inten_file=[] 
        self.inten_file_train=['skl_distance','skl_hmod_distance',  ]#'smod_hmod_length',  'skl hmod Distance',  'Length sp. skl to sh. skl.',"gauss_curv_smooth",'Annotation','smod_annot',"mean_curv_smooth","gauss_curv_init","mean_curv_init","intensity_hmod_neck_head",'intensity_hmod_smod','intensity_1hot_hmod_smod', 'intensity_1hot_hmod_neck_head']
        self.inten_file_train.extend(['intensity_hmod_smod','intensity_hmod_neck','intensity_hmod_head','intensity_hmod_neck_head','intensity_hmod.neck_head'])
        self.intensity_smods_logit=[f'intensity_inference_logit_{yuy}' for yuy in ['ini','fin']]
        self.intensity_neck_logit=[f'intensity_true_logit_{yuy}' for yuy in ['ini','fin']]
        self.intensity_head_logit=[f'intensity_head_logit_{yuy}' for yuy in ['sh','hd']]
        self.intensity_head_neck_logit=[f'intensity_neck_head_logit_{yuy}' for yuy in ['sh','nk','hd']]
        m_logit=[self.intensity_smods_logit,self.intensity_neck_logit]#,self.intensity_head_logit,self.intensity_head_neck_logit]
        self.intensity_logit_dict={key:val for key,val in zip(['inference','true',],m_logit)}
        # self.inten_file_train=['skl_distance', 'Annotation','smod_annot','smod_match','smod_match_dice' ]
        self.intensity_logit=[mm[0] for mm in m_logit]
        self.intensity_logit_ext=sum(m_logit,[])
        self.neld_file=['vertices_head','vertices_neck','vertices_smod','faces_head','faces_neck','faces_smod', "vertices_1","vertices_0",'faces_0']  
        self.inttt=[]
        self.inttt.extend(self.inten_file_sub)  
        # self.inttt.extend(self.inten_pca)  
        self.inttt.extend(self.inten_file_model_head_neck) 
        self.inttt.extend(self.intensity_logit_ext) 
        self.inttt.extend(self.inten_file_train) 
        # self.inttt.extend(self.intensity_smods_logit[:1])
        # self.inttt.extend(self.intensity_neck_logit[:1])
        # self.inttt.extend(self.intensity_head_logit[:1])
        # self.inttt.extend(self.intensity_head_neck_logit[:1])
        self.inttt.extend(gfg) 
        # self.path_file_sub={ty for ty in self.inttt}  
        self.dropdown_options_style = {'color': 'white', 'background-color': 'gray'}
 


        
        pre_portion='smod'
        pre_portion='head_neck'

        pinn_dir_data= 'save' 
        pinn_dir_dest= 'save' 


        self.data_mode={}
        self.data_mode['pinn_dir_data_all']=[]
        self.data_mode['model_sufix_all']=[]
        self.data_mode['mode_id']=[] 



class get_model_name:
    def __init__(self,pre_opt,train_test,seg_neld,dest_head,pre_portion): 
        self.seg_neld,self.dest_head,self.pre_portion=seg_neld,dest_head,pre_portion
        self.pre_opt,self.train_test=pre_opt,train_test
        get_name.__init__(self) 
    def vals(self, 
            inten_pinn_index ,
            base_features_index ,
            pre_opt=None,
            train_test=None,
            seg_neld=None,
            dest_head=None,
            pre_portion=None): 
        seg_neld=seg_neld or self.seg_neld
        dest_head=dest_head or self.dest_head
        pre_portion=pre_portion or self.pre_portion 
        pre_opt=pre_opt or self.pre_opt
        train_test=train_test or self.train_test  
        path_head,model_suf,path='pinn','pre_gmg2m2','save'
        id_path=f'{path_head}_{model_suf}_{path}' 
        self.inten_pinn_name_tmp=[self.inten_pinn_name[val] for val in inten_pinn_index]
        self.inten_pinn_tmp={
                                val:{
                                    'name':f'{self.inten_pinn[val]}.txt',
                                    'id_path':id_path
                                  } 
                                for val in inten_pinn_index
                                }

        self.inten_pinn_path=id_path
        # self.model_name=  '' if len(inten_pinn_index)==0 else '_'.join([self.inten_pinn[hh] for hh in self.inten_pinn])
        self.model_name = '_'.join([self.inten_pinn_name[i] for i in inten_pinn_index])# if inten_pinn_index else ''_{pre_portion[:2]}
        self.base_features_list=[list(self.base_features_dict.keys())[hhh] for hhh in base_features_index]
        self.dest_sufix = f"{pre_opt}_{''.join([self.base_features_dict[hh]['index'] for hh in self.base_features_list])}_{self.model_name}" if inten_pinn_index else f"{pre_opt}_{''.join([self.base_features_dict[hh]['index'] for hh in self.base_features_list])}"
        self.mode_id=f'{train_test}_{seg_neld}_{pre_portion[:2]}_{dest_head}_{self.dest_sufix}' 
        self.dest_dir=f'{seg_neld}'  

nh=24




'''

def get_configs():
    return {
            "DNN-0": {
                "pre_opt": "pre",
                "base_features_index": [],
                "inten_pinn_index": [],
                "data_sufix": "DNN-0",
                "dest_sufix": "DNN-0",
            },
            "DNN-1": {
                "pre_opt": "pre",
                "base_features_index": [0,1,2,3],
                "inten_pinn_index": [],
                "data_sufix": "DNN-1",
                "dest_sufix": "DNN-1",
            },
            "DNN-4": {
                "pre_opt": "pre",
                "base_features_index": [0,1,2,3,49],
                "inten_pinn_index": [],
                "data_sufix": "DNN-4",
                "dest_sufix": "DNN-4",
            },
            "DNN-2": {
                "pre_opt": "opt",
                "base_features_index": [0,1,2,3],
                "inten_pinn_index": [1], 
                # "inten_pinn_index": [0], 
                "data_sufix": "DNN-1",
                "dest_sufix": "DNN-2",
            },
            "DNN-5": {
                "pre_opt": "opt",
                "base_features_index": [0,1,2,3],
               #  "inten_pinn_index": [1], 
                "inten_pinn_index": [0], 
                "data_sufix": "DNN-1",
                "dest_sufix": "DNN-5",
            },
            "DNN-3": {
                "pre_opt": "pre",
                "base_features_index": [27+i for i in [0-27,2-27,7,8,9,10,11,12,13,14,15]],
                "inten_pinn_index": [],
                "data_sufix": "DNN-3",
                "dest_sufix": "DNN-3",
            }, 
            "DNN-6": {
                "pre_opt": "pre",
                "base_features_index": [0,2,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [],
                "data_sufix": "DNN-6",
                "dest_sufix": "DNN-6",
            },
            "mode2": {
                "pre_opt": "pre",
                "base_features_index": [0,1,2,3,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode3": {
                "pre_opt": "pre",
                "base_features_index": [0,2,4,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode4": {
                "pre_opt": "pre",
                "base_features_index": [0,2,4,7,11,15],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode5": {
                "pre_opt": "pre",
                "base_features_index": [0,2,3,4,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode6": {
                "pre_opt": "opt",
                "base_features_index": [0,1,2,3],
                "inten_pinn_index": [0],
                "model_init": None,
            },
            "mode7": {
                "pre_opt": "pre",
                "base_features_index": [0,1,2,3,22],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode9": {
                "pre_opt": "opt",
                "base_features_index": [0,2,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [1],
                "model_init": None,
            },
            "mode10": {
                "pre_opt": "pre",
                "base_features_index": [0,2,3,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode11": {
                "pre_opt": "pre",
                "base_features_index": [23,24,25,26,27, 7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode12": {
                "pre_opt": "opt",
                "base_features_index": [23,24,25,26,27 ],
                "inten_pinn_index": [0],
                "model_init": None,
            },
            "mode13": {
                "pre_opt": "opt",
                "base_features_index": [23,24,25,26,27, 7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [0],
                "model_init": None,
            },
            "mode14": {
                "pre_opt": "pre",
                "base_features_index": [23,24,25,26,27 ],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode15": {
                "pre_opt": "opt",
                "base_features_index": [0,2,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [0],
                "model_init": None,
            },
            "mode16": {
                "pre_opt": "pre",
                "base_features_index": [1,23,24,25,26,27, 7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [0],
                "model_init": None,
            },
            "mode17": {
                "pre_opt": "pre",
                "base_features_index": [0,1,2,3,22],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode18": {
                "pre_opt": "pre",
                "base_features_index": [0,2,10,11,12,13,14,15],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode19": {
                "pre_opt": "pre",
                "base_features_index": [0,1,2,3,10,11,12,13,14,15],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode20": {
                "pre_opt": "pre",
                "base_features_index": [7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode21": {
                "pre_opt": "pre",
                "base_features_index": [29,31,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode22": {
                "pre_opt": "pre",
                "base_features_index": [0,2,29,31,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [],
                "model_init": None,
            },
            "mode23": {
                "pre_opt": "opt",
                "base_features_index": [0,1,2,3,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [1],
                "model_init": None,
            },
        }

'''

# def drop_part(s, sep="_", index=1, name=None):
#     parts = s.split(sep) 
#     namme=[name] if name is None else name.split(sep)
#     if name is not None:
#         for nam in namme:
#             if nam in parts:
#                 parts.remove(nam)
#             return sep.join(parts) 
#     if index < 0 or index >= len(parts):
#         return s

#     return sep.join(parts[:index] + parts[index+1:])

def drop_part(s, sep="_", index=1, name=None): 
    parts = s.split(sep)
 
    # print('==============---------111224455100',)
    if name is not None: 
        if isinstance(name, str):
            names = name.split(sep) if sep in name else [name]
        else: 
            names = list(name) 
        parts = [p for p in parts if p not in names]
        # print('==============---------',parts)
        return sep.join(parts) 
    if index < 0 or index >= len(parts):
        return s 
    return sep.join(parts[:index] + parts[index+1:])

def metric_paths(base, prefix):
    return {
        'loss': os.path.join(base, f'loss_{prefix}.txt'),
        'oloss': os.path.join(base, f'oloss_{prefix}.txt'),
        'iou': {
            1: os.path.join(base, f'iou_{prefix}_sp.txt'),
            0: os.path.join(base, f'iou_{prefix}_sh.txt'),
        },
        'auc': {
            1: os.path.join(base, f'auc_{prefix}_sp.txt'),
            0: os.path.join(base, f'auc_{prefix}_sh.txt'),
        },
        'dice': {
            1: os.path.join(base, f'dice_{prefix}_sp.txt'),
            0: os.path.join(base, f'dice_{prefix}_sh.txt'),
        },
        'index_save': os.path.join(base, f'index_{prefix}.txt'),
    }
def model_paths(base, name, mo,mlname):
    # return {
    #     'model_oloss': os.path.join(base, f'model_oloss_{name}.{mo}'),
    #     'model_iou': os.path.join(base, f'model_iou_{name}.{mo}'),
    #     'model_dice': os.path.join(base, f'model_dice_{name}.{mo}'),
    #     'model_auc': os.path.join(base, f'model_auc_{name}.{mo}'),
    #     'model_loss': os.path.join(base, f'model_loss_{name}.{mo}'),
    # }
    
    return dict(zip([f'model_{mm}' for mm in mlname],
                    [os.path.join(base, f'model_{mm}_{name}.{mo}') for mm in mlname]
                    ))

class get_param:
    def __init__(self,  
                    hidden_layers=4, 
                    neurons_per_layer=100, 
                    n_col=2,
                    activation_init="relu",
                    activation_hidden="relu",
                    activation_last="sigmoid",
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
                    disp_infos=False,
                    thre_gauss=45,
                    thre_mean=15,
                    thre_gen=100,
                    weight_positive=0.5,
                    txt_save=False,
                    txt_save_pred=False,
                    DTYPE='float32',
                    txt_true_file=None,
                    zoom_thre=10,  
                    thre_target_number_of_triangles=None,
                    voxel_resolution=None,
                     ): 

        self.hidden_layers=hidden_layers
        self.neurons_per_layer=neurons_per_layer
        self.n_col=n_col
        self.activation_init=activation_init
        self.activation_hidden=activation_hidden
        self.activation_last=activation_last
        self.thre_target_number_of_triangles=thre_target_number_of_triangles
        self.voxel_resolution=voxel_resolution
                         
                         
                         
                         
        self.disp_infos=disp_infos
        self.radius_threshold=radius_threshold
        self.name_path_fin=name_path_fin
        self.thre_gauss=thre_gauss 
        self.thre_mean=thre_mean
        self.thre_gen=thre_gen
        self.weight_positive=weight_positive 
        self.zoom_thre=zoom_thre

        self.DTYPE=DTYPE
        self.txt_save=txt_save 
        self.txt_save_pred=txt_save_pred
        self.txt_true_file=txt_true_file

        self.neld_cla=None
        self.neld_saves=None 
 
        self.line_num_points_hmod= line_num_points_hmod 
        self.line_num_points_inter_hmod= line_num_points_inter_hmod 
        self.spline_smooth_hmod= spline_smooth_hmod
        self.numNeighbours=numNeighbours
        self.gauss_threshold=gauss_threshold
        self.size_threshold=size_threshold 
        self.zoom_threshold=zoom_threshold 
        self.cts=cts 
        self.stoppage=stoppage 
        self.zoom_threshold_min=zoom_threshold_min 
        self.zoom_threshold_max=zoom_threshold_max 
        self.smod_filter=smod_filter
        self.radius_threshold=radius_threshold
        vv_fin=[name_path_fin_save_index]
        if txt_save:
            if txt_save_pred: 
                vv=[ii for ii in range(self.cts)] 
                vv.extend(vv_fin)
                self.vv_cts=vv 
            else:
                self.vv_cts=vv_fin
        else:
            self.vv_cts=vv_fin
        self.name_path_fin_save=f'{name_path_fin}_{name_path_fin_save_index}' 
        self.name_path_fin_save_index=name_path_fin_save_index



class get_files(get_name,get_param):
    def __init__(self,
                    file_path_org,
                    model_sufix,
                    neld_data=None, 
                    neld_names=None,
                    neld_namess=None, 
                    data_studied=None, 
                    neld_path_inits=None,
                    name_smod_id=None,
                    name_head_id=None,
                    name_neck_id=None,
                    name_hmod_id=None,
                    size_threshold=10,
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
                    zoom_threshold_max=3, 
                    line_num_points_hmod=200,
                    line_num_points_inter_hmod=300, 
                    spline_smooth_hmod=1, 
                    disp_infos=False,
                    thre_gauss=45,
                    thre_mean=15,
                    thre_gen=1000,
                    weight_positive=0.5,
                    txt_save=False,
                    txt_save_pred=False,
                    DTYPE='float32',
                    txt_true_file=None,
                    path_train=None,
                    pre_portion=None,
                    pinn_dir_data=None,
                    path_file=None,  
                    path_file_sub=None,
                    pinn_dir_data_all=None,
                    model_sufix_all=None, 
                    list_features=None,
                    base_features_list=None,
                    model_init=None,
                    metrics={},
                    model_type=None,
                    data_mode=None,
                    path_heads=None, 
                    true_keys=None,
                    thre_target_number_of_triangles=None,
                    voxel_resolution=None,
                    obj_org_path=None,
                    obj_org_path_dict=None,
                    model_sufix_dic=None,
                    path_display_dic=None,
                    path_file_dir=None, 
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
        ) :  
        get_name.__init__(self)  
        get_param.__init__(self,
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
                        thre_gauss=thre_gauss ,
                        thre_mean=thre_mean,
                        thre_gen=thre_gen,
                        weight_positive=weight_positive, 
                        disp_infos=disp_infos,
                        thre_target_number_of_triangles=thre_target_number_of_triangles,
                        voxel_resolution=voxel_resolution,
                        ) 


        self.path_file_dir=path_file_dir
        self.thre_target_number_of_triangles=thre_target_number_of_triangles
        self.voxel_resolution=voxel_resolution
        self.model_type=model_type
        self.base_features_list=base_features_list
        self.list_features=list_features
        self.pinn_dir_data_all=pinn_dir_data_all
        self.model_sufix_all=model_sufix_all
        self.path_heads=path_heads
        self.true_keys=true_keys
        self.path_train=path_train
        self.path_file=path_file
        self.pinn_dir_data=pinn_dir_data
        self.pre_portion=pre_portion
        self.file_path_org=file_path_org
        self.neld_names=neld_names 
        self.neld_namess=neld_namess  
        self.neld_path_inits=neld_path_inits 
        self.neld_data=neld_data  
        self.kmean_n_run=kmean_n_run
        self.param_dic=param_dic

        self.kmean_max_iter=kmean_max_iter
        if neld_data is not None:
            self.neld_names=neld_names if neld_names is not None else neld_data['neld_names']
            self.neld_namess=neld_namess if neld_namess is not None else neld_data['neld_namess']
            self.neld_path_inits=neld_path_inits if neld_path_inits is not None else neld_data['neld_path_inits']
            self.name_smod_id =name_smod_id if name_smod_id is not None else neld_data['name_smod_id']
            self.name_head_id =name_head_id if name_head_id is not None else neld_data['name_head_id']
            self.name_neck_id = name_neck_id if name_neck_id is not None else neld_data['name_neck_id']
            self.name_hmod_id = name_hmod_id if name_hmod_id is not None else neld_data['name_hmod_id']
            self.obj_org_path_dict = obj_org_path_dict if obj_org_path_dict is not None else neld_data['obj_org_path_dict']
            self.obj_org_path = obj_org_path if obj_org_path is not None else neld_data['obj_org_path']
            self.part_id_mapping = {
                self.name_smod: self.name_smod_id,
                self.name_head: self.name_head_id,
                self.name_neck: self.name_neck_id,
                self.name_hmod: self.name_hmod_id
            }

        if data_mode is not None:
            self.path_train=path_train if path_train is not None else data_mode['path_train']
            self.pre_portion=pre_portion if pre_portion is not None else data_mode['pre_portion']
            # self.pinn_dir_data=pinn_dir_data if pinn_dir_data is not None else data_mode['pinn_dir_data']
            # self.list_features=list_features if list_features is not None else data_mode['list_features']
            # self.base_features_list=base_features_list if base_features_list is not None else data_mode['base_features_list']
            self.model_init=model_init if model_init is not None else data_mode['model_init']
        
        self.data_studied=data_studied
        self.txt_true_file=txt_true_file
        self.model_sufix=model_sufix
        self.data_mode=data_mode
        self.obj_org_path=obj_org_path or self.obj_org_path
        self.obj_org_path_dict=obj_org_path_dict or self.obj_org_path_dict
        self.model_sufix_dic=model_sufix_dic 
        self.path_dir=self.model_sufix_dic.get('path_dir',None) 
        self.path_display_dic=path_display_dic
        # self.file_path_model=self.file_path_save =os.path.join(file_path_org,'pinn')
        # os.makedirs(self.file_path_model, exist_ok=True) 
        # print('[[[[[[]]]]]]',self.file_path_org,file_path_org)
        self.file_diff=os.path.relpath(self.obj_org_path, os.path.dirname(self.file_path_org)) 
        # self.file_path_model_data= os.path.join(file_path_org,'data',self.file_diff)
        # mnn=neld_data['cpath']
        self.file_path_model_data= os.path.join(file_path_org,'data',*self.neld_data['cpath'])
        # print('[[[[[[[[]]]]]]]]',self.file_path_model_data, mnn)
        os.makedirs(self.file_path_model_data, exist_ok=True)  


        self.pkl_path_model_data = os.path.join(self.file_path_model_data,fr'neld_{self.data_studied}_data.pkl')
        self.dash_pages_path=os.path.join(file_path_org,'app','pages') 
   


        self.path_file={} if path_file is None else path_file   
        # print('[[[[[[[[]]]]]]]]',None if path_file_sub is None else path_file_sub.keys())
        self.path_file_sub ={mm:{} for mm in  self.inttt} if path_file_sub is None else path_file_sub
        # self.path_file={}    
        # self.path_file_sub ={mm:{} for mm in  self.inttt} 
            
        self.metrics=metrics
        for val in self.metrics_keys:
            self.metrics[val]={} 
        self.pinn_dir_data_all=pinn_dir_data_all= list(set(pinn_dir_data_all if pinn_dir_data_all is not None else self.pinn_dir_data_all ))
        self.model_sufix_all= list(set(model_sufix_all if model_sufix_all is not None else self.model_sufix_all  ))
        self.path_heads= list(set(path_heads if path_heads is not None else self.path_heads  )) 

        self.dropdown_true_keys_option=[]
        for intt in self.true_keys:
            self.dropdown_true_keys_option.append({'label': intt,          'value': intt,           'style': self.dropdown_options_style})   


        self.path_heads_show=self.model_sufix_dic.get('path_heads_show',None) 
        self.path_heads_show = self.path_heads_show if self.path_heads_show is not None else path_heads
        self.dropdown_path_head_option = []
        for nam in self.path_heads_show:
            # self.dropdown_path_head_option.append({'label': self.model_sufix_dic['path_heads_dic'][nam], 'value': nam, 'style': self.dropdown_options_style})
            self.dropdown_path_head_option.append({'label': self.model_sufix_dic['path_heads_dic_sec'].get(nam,nam), 'value': nam, 'style': self.dropdown_options_style})
        self.dropdown_path_head_option.append({'label': f'Annotation',   'value': 'true', 'style': self.dropdown_options_style})

        self.dropdown_model_suf_option=[]
        for intt in self.model_sufix_dic['model_sufix_show']: 
                self.dropdown_model_suf_option.append({'label': self.model_sufix_dic['model_sufix_dic'][intt],          'value': intt,           'style': self.dropdown_options_style})   

  
        self.dropdown_path_option=[]
        for intt in self.model_sufix_dic['path_dirs_show']:
            self.dropdown_path_option.append({'label': intt,          'value': intt,           'style': self.dropdown_options_style})   


        # self.dropdown_path_option=[]
        # for intt in ['save',]:
        #     self.dropdown_path_option.append({'label': intt,          'value': intt,           'style': self.dropdown_options_style})   






    def assign_if_none(self, **kwargs):
        for arg, value in kwargs.items():
            if value is None: 
                value = getattr(self, arg, None) 
            setattr(self, arg, value)

    def get_model_opt_name(self,  
                      model_type ,
                    file_path_model=None,
                    file_path_org=None, 
                    model_sufix=None,  
                     name_path_fin=None, 
                     mlname=['oloss', 'loss','iou','auc','model']):   
        file_path_org = file_path_org or self.file_path_org
        name_path_fin = name_path_fin or self.name_path_fin
        self.name_path_fin = name_path_fin
        model_sufix = model_sufix or self.model_sufix 

  
        if file_path_model is None:
            file_path_model = os.path.join(file_path_org , 'model',model_type,model_sufix ) 
            os.makedirs(file_path_model, exist_ok=True)
        self.file_path_model = file_path_model


        self.iou_save_dir_all = [os.path.join(self.file_path_model, f'iou_{ii}.txt') for ii in range(3)]

        self.shap_dir = os.path.join(self.file_path_model, 'shap.csv')
        self.df_metric_algorithms_dir = os.path.join(self.file_path_model, 'df_metric_algorithms.csv')
        
        mo = 'pkl' if model_type == 'ML' else 'pth' if model_type.endswith('cnn') else 'keras'
 

        self.model_dir_path = {
            'smod': {
                **metric_paths(self.file_path_model, 'smod',),
                **model_paths(self.file_path_model, 'smod', mo,mlname),
                'oloss': os.path.join(self.file_path_model, 'oloss_smod.txt'),
                'rhs_name': ['hmod_pre_sp', 'smod_pre_sp'],
            },
            'head': {
                **metric_paths(self.file_path_model, 'head'),
                **model_paths(self.file_path_model, 'head', mo,mlname),
                'rhs_name': ['hmod_pre_sp', 'head_pre_sp'],
            },
            'neck': {
                **metric_paths(self.file_path_model, 'neck'),
                **model_paths(self.file_path_model, 'neck',mo, mlname),
                'rhs_name': ['hmod_pre_sp', 'head_pre_sp'],
            },
            'neck_head': {
                'loss': os.path.join(self.file_path_model, 'loss_neck_head.txt'),
                'oloss': os.path.join(self.file_path_model, 'oloss_neck_head.txt'),
                'iou': {
                    2: os.path.join(self.file_path_model, 'iou_neck_head_hd.txt'),
                    1: os.path.join(self.file_path_model, 'iou_neck_head_nk.txt'),
                    0: os.path.join(self.file_path_model, 'iou_neck_head_sh.txt'),
                },
                'auc': {
                    2: os.path.join(self.file_path_model, 'auc_neck_head_hd.txt'),
                    1: os.path.join(self.file_path_model, 'auc_neck_head_nk.txt'),
                    0: os.path.join(self.file_path_model, 'auc_neck_head_sh.txt'),
                },
                'dice': {
                    2: os.path.join(self.file_path_model, 'dice_neck_head_hd.txt'),
                    1: os.path.join(self.file_path_model, 'dice_neck_head_nk.txt'),
                    0: os.path.join(self.file_path_model, 'dice_neck_head_sh.txt'),
                },
                'index_save': os.path.join(self.file_path_model, 'index_neck_head.txt'),
                **model_paths(self.file_path_model, 'neck_head', mo,mlname),
                'rhs_name': ['hmod_pre', 'neck_pre', 'head_pre'],
            },
            'hmod': {
                'loss': os.path.join(self.file_path_model, 'loss_hmod.txt'),
                'oloss': os.path.join(self.file_path_model, 'oloss_hmod.txt'),
                'model': os.path.join(self.file_path_model, f'model_hmod.{mo}'),
                'rhs_name': ['hmod_pre'],
            },
            'default': {
                'loss': os.path.join(self.file_path_model, 'loss.txt'),
                'oloss': os.path.join(self.file_path_model, 'oloss.txt'),
                'iou': os.path.join(self.file_path_model, 'iou.txt'),
                'auc': os.path.join(self.file_path_model, 'auc.txt'),
                'dice': os.path.join(self.file_path_model, 'dice.txt'),
                **model_paths(self.file_path_model, 'default', mo,mlname),
                'rhs_name': ['smod_pre'],
            }
        }

    def get_path(self, *names, name='pinn',
                 neld_path=None,
                 file_path_model_data=None,
                 neld_path_true=None,
                 file_path_org=None,
                file_path=None,
                file_path_feat=None, 
                neld_path_true_final=None,
                 ):
        neld_path=neld_path if neld_path is not None else self.neld_path
        file_path_model_data =file_path_model_data if file_path_model_data is not None else self.file_path_model_data
        neld_path_true=neld_path_true if neld_path_true is not None else self.neld_path_true
        base = os.path.join(neld_path, name)

        if name.startswith( 'pinn'):
            if names:
                base = os.path.join(base, f'{names[0]}', *names[1:])
            else:
                base = os.path.join(base, f'{self.name_path_fin}')
        elif name == 'result':
            base = file_path_model_data#os.path.join(self.file_path_org, 'data',self.file_diff)
            if names: 
                if names[0]=='true': 
                    base = os.path.join(base, name, f'{names[0]}') 
                    # base = os.path.join(base,self.neld_path_inits[0],  name, f'{names[0]}') 
                else:  
                    base = os.path.join(base,  name, f'{names[0]}',    *names[1:])  
                    # base = os.path.join(base,self.neld_path_inits[0],  name, f'{names[0]}',    *names[1:]) 

        elif name.startswith('true'): 
            base = os.path.join( neld_path_true, name) 
        elif name.startswith('resized'): 
            base = os.path.join(self.neld_path_resized, names[0])
        else:
            if names:
                base = os.path.join(base, f'{names[0]}', *names[1:])
            else:
                base = os.path.join(base, f'{self.name_path_fin}')

        return base
    

    def get_paths(self, names,name=None,
                 neld_path=None,
                 file_path_model_data=None,
                 neld_path_true=None,
                  file_path_org=None,
                file_path=None,
                file_path_feat=None,
                 neld_path_true_final=None, ):
        name=name if name is not None else names[0]
        return self.get_path(*names[1:],name=name,
                 neld_path=neld_path,
                 file_path_model_data=file_path_model_data,
                 neld_path_true=neld_path_true,
                 file_path_org=file_path_org,
                file_path=file_path,
                file_path_feat=file_path_feat,)
 


    def get_neld_name(self, index,
                        neld_names=None,
                        neld_namess=None,
                        file_path_org=None, 
                        data_studied=None,
                        file_path_model_data=None,
                        name_path_fin=None,
                        name_path_fin_save=None, 
                        neld_path_inits=None,
                        model_sufix=None,
                        pinn_dir_data=None, 
                        pinn_dir_data_all=None,
                        model_sufix_all=None, 
                        path_heads =None,
                        model_type=None,
                        obj_org_path=None,
                        entry_name=None,
                        exit_name=None,
                        path_file_dir=None,
                        data_org='data_org', 
                        wrap_part='hmod_wrap',
                        dict_neld_path='current',
                        drop_dic_name=None, 
                        old_path=None,
                        nam_gen=None,


        ):  
        path_file_dir=path_file_dir if path_file_dir is not None else self.path_file_dir
        obj_org_path = obj_org_path or self.obj_org_path
        self.obj_org_path=obj_org_path 

        model_type=model_type or self.model_type
        if file_path_org is None:
            file_path_org = self.file_path_org 
        self.model_sufix=model_sufix = model_sufix or self.model_sufix 
        self.get_model_opt_name(model_sufix=model_sufix,model_type=model_type)
        neld_names = neld_names or self.neld_names
        file_path_model_data = file_path_model_data or self.file_path_model_data
        neld_namess = neld_namess or self.neld_namess
        name_path_fin = name_path_fin or self.name_path_fin
        neld_path_inits = neld_path_inits or self.neld_path_inits
        self.data_studied = data_studied or self.data_studied
        name_path_fin_save = name_path_fin_save or self.name_path_fin_save
        pinn_dir_data=pinn_dir_data or self.pinn_dir_data
        # print('[[[[[]]]]]',index,neld_path_inits)
        self.neld_path_init =neld_path_inits[index]
        # Assigning to self
        # self.file_path_org_init= os.path.join(self.file_path_model_data, self.neld_path_init,path_dir)
        self.file_path_org_init= os.path.join(self.file_path_model_data,'data' )
        self.file_path_org_true=os.path.join(self.file_path_model_data,'true' )
        self.file_path_org_temp=os.path.join(self.file_path_model_data,'temp' )
        self.name_path_fin = name_path_fin
        self.name_path_fin_save = name_path_fin_save


        self.neld_name = f'{neld_names[index]}' 
        self.last_name,self.sp_name_hmod=neld_namess[index][0],neld_namess[index][1]  
 
        # print('self.obj_org_path, self.neld_name',self.obj_org_path, self.file_path_org_init,self.neld_name)
        self.neld_path_original =self.neld_path_original_m = os.path.join(self.file_path_org_init, self.neld_name, data_org) 
        self.neld_path_original_new = os.path.join(self.obj_org_path, self.neld_name, data_org)   
        self.neld_path_org_new = os.path.join(self.obj_org_path, self.neld_name, 'data')   
        self.neld_path_original_new_smooth = os.path.join(self.obj_org_path, self.neld_name, 'data_smooth') 
        # os.makedirs(self.neld_path_original_new_smooth, exist_ok=True)

 
        drop_dic=  self.model_sufix_dic['drop_dic'] 
        # print('[[[[[[[[[[00]]]]]]]]]]',drop_dic)
        if (drop_dic is None) or (len(drop_dic)==0):
            drop_dic=dict(index=1,sep="_", name='smooth' )
        drop_dic['name']=drop_dic['name'] if drop_dic_name is None else drop_dic_name
        self.drop_dic=drop_dic
        '''
        diffo = self.file_diff.split('/')
        print('[[[[[[[[[]]]]]]]]]',diffo)
        difff = drop_part(diffo[1], **self.drop_dic) 
        diffo[1] = difff 
        self.file_diff_smooth =self.file_diff= '/'.join(diffo)
        if nam_gen is not None:
            self.file_diff_smooth =self.file_diff= '/'.join([nam_gen,drop_dic_name])'''



        from pathlib import Path

        p = Path(self.file_diff)

        parts = list(p.parts)
        print(parts)

        # Modify the last directory name
        parts[-1] = drop_part(parts[-1], **self.drop_dic)

        self.file_diff = self.file_diff_smooth = str(Path(*parts))

        if nam_gen is not None:
            self.file_diff = self.file_diff_smooth = str(Path(nam_gen) / drop_dic_name)









        # print('[[[[[[[[[[[[[[[[[[[[nam  gen]]]]]]]]]]]]]]]]]]]]',nam_gen)

        self.neld_path_org=self.neld_path = os.path.join(self.file_path_org_init, self.neld_name )
        # self.neld_path_temp=  os.path.join(self.file_path_org_temp, self.neld_name )
        self.file_path= os.path.join(self.file_path_org_temp, self.neld_name )
        # self.neld_path_resized = os.path.join(self.file_path_model_data, f'{self.neld_path_init}_resized','data',f'{self.neld_name}')
        os.makedirs(self.neld_path, exist_ok=True)
        # self.file_path = os.path.join(self.neld_path_temp, 'data')
        os.makedirs(self.file_path, exist_ok=True)

         

        self.obj_org_path_entry=self.obj_org_path if entry_name is None else f'{self.obj_org_path}_{entry_name}'
        self.file_path_model_data_entry=self.file_path_model_data if entry_name is None else f'{self.file_path_model_data}_{entry_name}'
        self.neld_path_org_entry=os.path.join(self.obj_org_path_entry, self.neld_name, data_org)
        self.neld_path_entry = os.path.join(self.file_path_model_data_entry,'data',f'{self.neld_name}')
        self.file_path_entry= os.path.join(self.file_path_model_data_entry,'temp',f'{self.neld_name}')
        # self.file_path_temp_entry= os.path.join(self.neld_path_temp_entry, 'data' )  
        self.file_path_feat_entry= os.path.join(self.file_path_entry, 'feat' ) 

        

        self.obj_org_path_exit=self.obj_org_path if exit_name is None else f'{self.obj_org_path}_{exit_name}'
        self.file_path_model_data_exit=self.file_path_model_data if exit_name is None else f'{self.file_path_model_data}_{exit_name}'
        self.neld_path_org_exit=os.path.join(self.obj_org_path_exit, self.neld_name, data_org)
        self.neld_path_exit = os.path.join(self.file_path_model_data_exit,'data',f'{self.neld_name}')
        # self.file_path_exit= os.path.join(self.neld_path_exit, 'data' )  
        # self.file_path_feat_exit= os.path.join(self.file_path_exit, 'feat' ) 
        self.file_path_exit = os.path.join(self.file_path_model_data_exit,'temp',f'{self.neld_name}') 
        self.file_path_feat_exit= os.path.join(self.file_path_exit, 'feat' ) 


        self.file_path_org_true=os.path.join(self.file_path_model_data,'true' )
        self.neld_path_true = os.path.join(self.file_path_org_true,f'{self.neld_name}')
        self.file_path_org_true_exit =self.file_path_org_true if exit_name is None else os.path.join(f'{self.file_path_model_data}_{exit_name}','true')
        self.file_path_org_true_entry=self.file_path_org_true if exit_name is None else os.path.join(f'{self.file_path_model_data}_{entry_name}','true')
        self.neld_path_true_exit  = os.path.join(self.file_path_org_true_exit,f'{self.neld_name}')
        self.neld_path_true_entry = os.path.join(self.file_path_org_true_entry,f'{self.neld_name}')



        self.obj_org_path_old=os.path.join(os.path.dirname(os.path.dirname(self.obj_org_path)),self.file_diff_smooth)
        self.file_path_model_data_old=os.path.join(os.path.dirname(os.path.dirname(self.file_path_model_data)),self.file_diff_smooth)
        # if old_path =='entry':
        #     self.obj_org_path_old=self.obj_org_path_entry
        #     self.file_path_model_data_old=self.file_path_model_data_entry
        # elif old_path =='current':
        #     self.obj_org_path_old=self.obj_org_path
        #     self.file_path_model_data_old=self.file_path_model_data
        # elif old_path =='exit':
        #     self.obj_org_path_old=self.obj_org_path_exit
        #     self.file_path_model_data_old=self.file_path_model_data_exit
        #     print('((((((((((((((((((((((((((((((()))))))))))))))))))))))))))))))')
        if old_path is not None:
            if old_path=='current':
                self.obj_org_path_old=f'{self.obj_org_path}'
                self.file_path_model_data_old=f'{self.file_path_model_data}'
            else:
                self.obj_org_path_old=f'{self.obj_org_path}_{old_path}'
                self.file_path_model_data_old=f'{self.file_path_model_data}_{old_path}'

        # print('[[[[[[[[[[[[[[[[[[diff_smooth]]]]]]]]]]]]]]]]]]',self.file_diff_smooth)
        # print('[[[[[[[[[[[[[[[[[[obj_org_path_old]]]]]]]]]]]]]]]]]]',self.obj_org_path_old)
        # print('[[[[[[[[[[[[[[[[[[file_path_model_data_old]]]]]]]]]]]]]]]]]]',self.file_path_model_data_old)
        # print('((((((((((((((((()))))))))))))))))))))',drop_dic_name,self.file_diff_smooth)
        # print('[[[[[[[[[self.obj_org_path]]]]]]]]]======',self.obj_org_path_entry)
        # print('[[[[[[[[[self.obj_org_path_old]]]]]]]]]======',self.obj_org_path_old) 
        self.neld_path_org_old=os.path.join(self.obj_org_path_old, self.neld_name, data_org) 
        self.neld_path_old = os.path.join(self.file_path_model_data_old,'data',f'{self.neld_name}')
        # self.file_path_old= os.path.join(self.neld_path_old, 'data' )  
        # self.file_path_feat_old= os.path.join(self.file_path_old, 'feat' ) 
        self.file_path_old= os.path.join(self.file_path_model_data_old,'temp',f'{self.neld_name}') 
        self.file_path_feat_old= os.path.join(self.file_path_old, 'feat' ) 
        self.file_path_org_old_true=os.path.join(self.file_path_model_data_old,'true' )
        self.neld_path_true_old = os.path.join(self.file_path_org_old_true,f'{self.neld_name}')


        self.file_path_feat = os.path.join(self.file_path, 'feat')
        os.makedirs(self.file_path_feat, exist_ok=True)
 
        self.vertices_1_path=os.path.join(self.file_path,  self.txt_vertices_1)
        self.vertices_0_path=os.path.join(self.file_path,   self.txt_vertices_0)
        self.faces_path=os.path.join(self.file_path,   self.txt_faces)   
        self.neld_first_name=self.neld_namess[index][1]
  

        self.dict_neld={va:{
            la:{} for la in ['current','old','entry','exit']
        } for va in ['path','key', ]
        }
        self.dict_neld['path']['current']=dict(neld_path=self.neld_path,
                    file_path_model_data=self.file_path_model_data,
                    neld_path_true=self.neld_path_true,
                    file_path_org=self.file_path_org,
                    file_path=self.file_path,
                    file_path_feat=self.file_path_feat, 
                            )
        self.dict_neld['path']['old']=dict(neld_path=self.neld_path_old,
                    file_path_model_data=self.file_path_model_data_old,
                    neld_path_true=self.neld_path_true_old , 
                    file_path_feat=self.file_path_feat_old,
                    file_path=self.file_path_old,  
                    )
        # pinn_dir_data_all= pinn_dir_data_all or self.pinn_dir_data_all 
        # pinn_dir_data_all=list(self.model_sufix_dic['path_dirs'].keys())
        # model_sufix_all= model_sufix_all or self.model_sufix_all 

        path_heads=self.model_sufix_dic['path_heads_show'] 
        model_sufix_all=self.model_sufix_dic['model_sufix_show'] 
        pinn_dir_data_all=self.model_sufix_dic['path_dirs_show']

        # path_heads = path_heads or self.path_heads
        path_headss=list(path_heads)
        path_headss.append('true') 
        if (path_file_dir is not None) :
            if path_file_dir is not None:
                with open(path_file_dir, "rb") as f: 
                    loaded_dict = pickle.load(f) 
                self.path_file_dir=loaded_dict['path_file_dir']
                self.path_train=loaded_dict['path_train']
                self.path_file=loaded_dict['path_file']
                self.path_file_sub=loaded_dict['path_file_sub']
                self.pinn_dir_data=loaded_dict['pinn_dir_data']
                # self.neld_data=neld_data=loaded_dict['neld_data']
                self.obj_org_path_dict=loaded_dict['obj_org_path_dict']
                self.model_sufix_dic=loaded_dict['model_sufix_dic']
                self.path_display=loaded_dict['path_display']
                self.path_display_dic=loaded_dict['path_display_dic']
                self.neld_path_original_mm=loaded_dict['neld_path_original_mm']
                self.path_heads_show=self.model_sufix_dic.get('path_heads_show',None)
    

            # if neld_data is not None: 
            #     neld_names=neld_names if not None else neld_data['neld_names']
            #     neld_namess=neld_namess if not None else neld_data['neld_namess']
            #     neld_path_inits=neld_path_inits if not None else neld_data['neld_path_inits'] 
            return 


        for pa in pinn_dir_data_all:
            if pa is not None:
                for model_sufi in model_sufix_all:
                    for path_head in path_headss:
                        modd=os.path.join(self.file_path_org, 'model',path_head,model_sufi) 
                        key=f'{path_head}_{model_sufi}_{pa}'# if dict_neld_path is 'current' else f'{dict_neld_path}_{path_head}_{model_sufi}_{pa}'
                        self.path_file_sub[self.inten_file_sub[0]][key] =self.get_paths([path_head, model_sufi, pa],
                                                                                            **self.dict_neld['path'][dict_neld_path],)
                        self.path_file[key]=self.get_paths([path_head, model_sufi, pa],
                                                            **self.dict_neld['path'][dict_neld_path],)
                        os.makedirs(self.path_file[key], exist_ok=True) 
                        for tyy in self.inten_file_sub[1:]: 
                            self.path_file_sub[tyy][key]= os.path.join(self.path_file[key],f'{tyy}.txt')  
                        for tyy in self.intensity_logit_ext: 
                            self.path_file_sub[tyy][key]= os.path.join(self.path_file[key],f'{tyy}.txt') 
                            # self.path_file_sub[tyy][key]= os.path.join(os.path.dirname(self.path_file[key]),f'{tyy}.txt') 
                        # for tyy in self.intensity_smods_logit: 
                        #     self.path_file_sub[tyy][key]= os.path.join(os.path.dirname(self.path_file[key]),f'{tyy}.txt') 
                        # for tyy in self.intensity_head_neck_logit: 
                        #     self.path_file_sub[tyy][key]= os.path.join(os.path.dirname(self.path_file[key]),f'{tyy}.txt')  


                        # for tyy in self.inten_pca: 
                        #     self.path_file_sub[tyy][key]= os.path.join(self.path_file[key],f'{tyy}.txt') 


                        for tyy in self.inten_file: 
                            self.path_file_sub[tyy][key]= os.path.join(self.path_file[key],f'{tyy}.txt') 

                        for tyy in self.inten_file_train: 
                            self.path_file_sub[tyy][key]= os.path.join(self.file_path_feat,f'{tyy}.txt') 

                        # self.file_model_train=['smod','head_neck','hmod']
                        # self.inten_file_model_train=['loss','iou'] 
                        '''
                        tyy =self.inten_file_model_head_neck_loss[0]
                        tyyy=self.inten_file_model_train_loss[0]
                        self.path_file_sub[tyy][key]= os.path.join(modd,f'{tyyy}.txt')

                        tyy =self.inten_file_model_head_neck_iou[0]
                        self.path_file_sub[tyy][key]={}
                        for tyyy in  self.inten_file_model_train_iou:  
                                self.path_file_sub[tyy][key][tyyy]= os.path.join(modd,f'{tyyy}.txt')

'''
                        tyy =self.inten_file_model_smod_loss[0]
                        tyyy=self.inten_file_model_train_smod_loss[0]
                        self.path_file_sub[tyy][key]= os.path.join(modd,f'{tyyy}.txt')
                        '''
                        tyy =self.inten_file_model_smod_iou[0]
                        self.path_file_sub[tyy][key]={}
                        for tyyy in  self.inten_file_model_train_smod_iou:  
                                self.path_file_sub[tyy][key][tyyy]= os.path.join(modd,f'{tyyy}.txt')

                        tyy =self.inten_file_model_smod_auc[0]
                        self.path_file_sub[tyy][key]={}
                        for tyyy in  self.inten_file_model_train_smod_auc:  
                                self.path_file_sub[tyy][key][tyyy]= os.path.join(modd,f'{tyyy}.txt')

                        tyy =self.inten_file_model_smod_dice[0]
                        self.path_file_sub[tyy][key]={}
                        for tyyy in  self.inten_file_model_train_smod_dice:  
                                self.path_file_sub[tyy][key][tyyy]= os.path.join(modd,f'{tyyy}.txt')
'''
        # self.inten_file_model_smod_auc=[ 'auc_smod_sh','auc_smod_sp' ]
                        tyy =self.inten_file_model_shap[0]
                        tyyy=self.inten_file_model_train_shap[0]
                        self.path_file_sub[tyy][key]= os.path.join(modd,f'{tyyy}.csv')
 

                        for tyy in self.base_features_dict.keys(): 
                            self.path_file_sub[tyy][key]= os.path.join(self.file_path_feat,f'{tyy}.txt') 

  


        # if len(self.obj_org_path_dict)>0:
        #     for ii,(keys,val) in enumerate(self.obj_org_path_dict.items()): 
        #         path_head ,pa,model_sufi=f'true_{ii}','save',f'save' 
        #         key=f'{path_head}_{model_sufi}_{pa}' 
        #         pasd=self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([path_head, model_sufi, pa])
        #         remove_directory(self.path_file[key] )
        path_headss=[]
        self.neld_path_original_mm={ky:{} for ky in ['keys','dir']}  
        if len(self.obj_org_path_dict)>0:
            for ii,(keys,val) in enumerate(self.obj_org_path_dict.items()): 
                path_head ,model_sufi,pa=keys,'save',f'save' 
                key=f'{path_head}_{model_sufi}_{pa}'
                path_headss.extend(path_head) 
                self.neld_path_original_mm['dir'][key]=os.path.join(val, self.neld_name, 'data_org')
                self.neld_path_original_mm['keys'][key]=keys
 
                # for tyy in list(set(self.inten_file_sub[1:]+list(self.base_features_dict.keys()))): 

                path_head,model_sufi ,pa=keys,'save',f'save' 
                # key=f'resized_{path_head}_{model_sufi}_{pa}' 
                # pasd=self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths(['resized',path_head, model_sufi, pa],name='resized')
                self.neld_path_true_final_current= os.path.join(self.neld_path_true,keys)
                self.neld_path_true_final_entry = os.path.join(self.neld_path_true_entry,keys)
                self.neld_path_true_final_exit  = os.path.join(self.neld_path_true_exit,keys)
                self.neld_path_true_final_old = os.path.join(self.neld_path_true_old,keys)
                self.dict_neld['path']['old']['neld_path_true_final']=self.neld_path_true_final_old
                self.dict_neld['path']['current']['neld_path_true_final']=self.neld_path_true_final_current
                self.neld_path_true_final=self.dict_neld['path'][dict_neld_path]['neld_path_true_final']
                self.path_file[key]=self.neld_path_true_final
                # pasd=self.path_file_sub[self.inten_file_sub[0]][key]=
                # print('[[[[]]]]',self.neld_path_true_final,)
                os.makedirs(self.neld_path_true_final, exist_ok=True)
                for tyy in self.inttt:
                    self.path_file_sub[tyy][key]= os.path.join(self.neld_path_true_final,f'{tyy}.txt')

 
        for keyss,path_head in  self.neld_path_original_mm['keys'].items():
            pasd= self.path_file[keyss]  
            for pa in pinn_dir_data_all:
                if pa is not None:
                    for model_sufi in model_sufix_all: 
                            modd=os.path.join(self.file_path_org, 'model',path_head,model_sufi)
                            key=f'{path_head}_{model_sufi}_{pa}' # if dict_neld_path is 'current' else f'{dict_neld_path}_{path_head}_{model_sufi}_{pa}'
                            # key=f'{path_head}_{model_sufi}_{pa}' 
                            self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=pasd
                            # os.makedirs(self.path_file[key], exist_ok=True) 
                            for tyy in self.inten_file_sub[1:]: 
                                self.path_file_sub[tyy][key]= os.path.join(self.path_file[key],f'{tyy}.txt') 

                            # for tyy in self.inten_pca: 
                            #     self.path_file_sub[tyy][key]= os.path.join(self.path_file[key],f'{tyy}.txt') 


                            for tyy in self.inten_file: 
                                self.path_file_sub[tyy][key]= os.path.join(self.path_file[key],f'{tyy}.txt') 

                            for tyy in self.inten_file_train: 
                                self.path_file_sub[tyy][key]= os.path.join(self.file_path_feat,f'{tyy}.txt') 

                            # self.file_model_train=['smod','head_neck','hmod']
                            # self.inten_file_model_train=['loss','iou'] 
                            '''
                            tyy =self.inten_file_model_head_neck_loss[0]
                            tyyy=self.inten_file_model_train_loss[0]
                            self.path_file_sub[tyy][key]= os.path.join(modd,f'{tyyy}.txt')

                            tyy =self.inten_file_model_head_neck_iou[0]
                            self.path_file_sub[tyy][key]={}
                            for tyyy in  self.inten_file_model_train_iou:  
                                    self.path_file_sub[tyy][key][tyyy]= os.path.join(modd,f'{tyyy}.txt')
'''

                            tyy =self.inten_file_model_smod_loss[0]
                            tyyy=self.inten_file_model_train_smod_loss[0]
                            self.path_file_sub[tyy][key]= os.path.join(modd,f'{tyyy}.txt')

                            tyy =self.inten_file_model_smod_iou[0]
                            self.path_file_sub[tyy][key]={}
                            for tyyy in  self.inten_file_model_train_smod_iou:  
                                    self.path_file_sub[tyy][key][tyyy]= os.path.join(modd,f'{tyyy}.txt')

                            tyy =self.inten_file_model_smod_auc[0]
                            self.path_file_sub[tyy][key]={}
                            for tyyy in  self.inten_file_model_train_smod_auc:  
                                    self.path_file_sub[tyy][key][tyyy]= os.path.join(modd,f'{tyyy}.txt')

                            tyy =self.inten_file_model_smod_dice[0]
                            self.path_file_sub[tyy][key]={}
                            for tyyy in  self.inten_file_model_train_smod_dice:  
                                    self.path_file_sub[tyy][key][tyyy]= os.path.join(modd,f'{tyyy}.txt')


                            tyy =self.inten_file_model_shap[0]
                            tyyy=self.inten_file_model_train_shap[0]
                            self.path_file_sub[tyy][key]= os.path.join(modd,f'{tyyy}.csv')

         
        for pa in pinn_dir_data_all:
            if pa is not None:
                for model_sufi in model_sufix_all:
                    for res in ['result',]:
                        for path_head in path_heads:
                            key=f'{res}_{path_head}_{model_sufi}_{pa}' 
                            key=f'{res}_{path_head}_{model_sufi}_{pa}' # if dict_neld_path is 'current' else f'{dict_neld_path}_{res}_{path_head}_{model_sufi}_{pa}'
                            self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([res,path_head, model_sufi, pa],
                                                                                                               **self.dict_neld['path'][dict_neld_path])
                            os.makedirs(self.path_file[key], exist_ok=True)  

        for nn in 'true_0':
            remove_directory(os.path.join(self.neld_path_org,  nn))
            # remove_directory(os.path.join(self.file_path_model_data, self.neld_path_init,'result',nn)), self.neld_path_inits[index]
            remove_directory(os.path.join(self.file_path_model_data,'result',nn))

        key=f'result_true'  if dict_neld_path == 'current' else f'{dict_neld_path}_result_true'
        self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths(['result', 'true_0'],
                                                                                           **self.dict_neld['path'][dict_neld_path]
                                                                                           )
        os.makedirs(self.path_file[key], exist_ok=True) 

    
        key=f'result_appr'#  if dict_neld_path is 'current' else f'{dict_neld_path}_result_appr'
        self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths(['result', 'appr'],
                                                                                           **self.dict_neld['path'][dict_neld_path]
                                                                                           )
        os.makedirs(self.path_file[key], exist_ok=True) 
 

        self.neld_data_all ={jj:
                                    {ii:{} 
                                        for ii in ['path','distance','vertices',f'distance_{wrap_part}',f'vertices_{wrap_part}']
                                    }
                                    for jj in ['entry','old','exit']
                                    }
        # neld_path_entry_data_all['path'] = {
        #                                     "entry": self.neld_path_org_entry,
        #                                     "old":   self.neld_path_org_old,
        #                                     "exit":  self.neld_path_org_exit
        #                                 }
        mkm = [
            [self.neld_path_org_entry,        self.neld_path_org_old,        self.neld_path_org_exit],
            [self.txt_skl_distance,           self.txt_skl_distance_org,     self.txt_skl_distance],
            [self.txt_skl_vertices,           self.txt_skl_vertices_org,     self.txt_skl_vertices],
            [self.txt_skl_hmod_distance,     self.txt_skl_hmod_distance_org,self.txt_skl_hmod_distance],
            [self.txt_skl_hmod_vertices,     self.txt_skl_hmod_vertices_org,self.txt_skl_hmod_vertices],
            [self.txt_skl_index,              self.txt_skl_index_org,         self.txt_skl_index], 
        ]

        for mm, lol in zip(
                ['path', 'distance', 'vertices', f'distance_{wrap_part}', f'vertices_{wrap_part}','skl_index',],
                mkm): 
            for idx, nam in zip(["entry", "old", "exit"], lol):
                self.neld_data_all[idx][mm] = nam




    def get_dash_pages_name(self,index,data_studied,
                        dict_neld_path='current',
                        drop_dic_name=None,):
        self.get_neld_name(data_studied=data_studied,
                            index=index,
                        dict_neld_path=dict_neld_path,
                        drop_dic_name=drop_dic_name,
                               ) 
        
        dash_path_neld=os.path.join(self.dash_pages_path,self.data_studied,self.model_type, self.model_sufix,self.file_diff)  
        os.makedirs(dash_path_neld, exist_ok=True)
        self.dash_pages_name=os.path.join(dash_path_neld,f'{self.neld_names[index]}.py')



    def clean_neld_name(self, 
                      path_head_clean,
                    neld_names=None,
                    neld_namess=None,
                    file_path_org=None, 
                    data_studied=None,
                    file_path_model_data=None,
                    name_path_fin=None,
                    name_path_fin_save=None, 
                    neld_path_inits=None,
                    model_sufix=None,
                    pinn_dir_data=None, 
                    pinn_dir_data_all=None,
                    model_sufix_all=None,
                    path_heads=None,
                    model_type=None,


        ):  
        model_type=model_type or self.model_type
        file_path_org =file_path_org or self.file_path_org 
        self.model_sufix=model_sufix = model_sufix or self.model_sufix 
        self.get_model_opt_name(model_sufix=model_sufix,model_type=model_type,)
        neld_names = neld_names or self.neld_names
        file_path_model_data = file_path_model_data or self.file_path_model_data
        neld_namess = neld_namess or self.neld_namess
        name_path_fin = name_path_fin or self.name_path_fin
        neld_path_inits = neld_path_inits or self.neld_path_inits
        self.data_studied = data_studied or self.data_studied
        name_path_fin_save = name_path_fin_save or self.name_path_fin_save
        pinn_dir_data=pinn_dir_data or self.pinn_dir_data


        pinn_dir_data_all= pinn_dir_data_all or self.pinn_dir_data_all# or ['save','new','new_pre',pinn_dir_data,f'{pinn_dir_data}_pre']
        model_sufix_all= model_sufix_all or self.model_sufix_all# or ['pre','opt' ,model_sufix] 
        path_heads = path_heads or self.path_heads

        for path_head_clean  in  path_head:
            if path_head not in ['true','result']:
                for pa in pinn_dir_data_all:
                    if pa is not None:
                        for model_sufi in model_sufix_all: 
                            key=f'{path_head}_{model_sufi}_{pa}' 
                            if key in self.path_file:
                                remove_directory(self.path_file[key] )   

        if path_head_clean =='result'  :
            for pa in pinn_dir_data_all:
                if pa is not None:
                    for model_sufi in model_sufix_all:
                        for res in ['result']:
                            for path_head in path_heads:
                                key=f'{res}_{path_head}_{model_sufi}_{pa}'
                                self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([res,path_head, model_sufi, pa])
                                if key in self.path_file:
                                    remove_directory(self.path_file[key] )  

        if path_head_clean=='all':
            remove_directory(self.file_path_org_init)
            remove_directory(os.path.join(self.dash_pages_path,self.data_studied))



            path_headss=[pa for pa in path_heads if pa.startswith(('pinn','rpinn','gcn'))]
            for pa in pinn_dir_data_all:
                if pa is not None:
                    for model_sufi in model_sufix_all:
                        for res in ['data']:
                            for path_head in path_headss:
                                key=f'{path_head}_{model_sufi}_{pa}'
                                path= os.path.join(self.neld_path,  path_head)
                                # path= os.path.join(self.neld_path, res,path_head),'true'
                                #{res}_ self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([res,path_head, model_sufi, pa])
                                print('Removing ----------',path)
                                remove_directory(path) 

                        for res in ['result']:
                            for path_head in path_heads:
                                key=f'{res}_{path_head}_{model_sufi}_{pa}'
                                self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([res,path_head, model_sufi, pa])
                                if key in self.path_file:
                                    remove_directory(self.path_file[key] )  


            print('Removing ---------',path)
            remove_directory(self.file_path_feat)
            for nn in 'true_0':
                remove_directory(os.path.join(self.neld_path_org,  nn))

            if len(self.obj_org_path_dict)>0:
                for ii,(keys,val) in enumerate(self.obj_org_path_dict.items()): 
                    path_head ,pa,model_sufi=f'true_{ii}','save',f'save' 
                    key=f'{path_head}_{model_sufi}_{pa}'
                    path_heads.extend(path_head) 
                    self.neld_path_original_mm[key]=os.path.join(val, self.neld_name, 'data_org')
                    pasd=self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([path_head, model_sufi, pa])
                    remove_directory(self.path_file[key] )


            key=f'result_true'
            if key in path_head_clean:  
                remove_directory(self.path_file[key] )

            dash_path_neld=os.path.join(self.dash_pages_path,self.data_studied)
            os.makedirs(dash_path_neld, exist_ok=True)
            dash_path_neld=os.path.join(dash_path_neld,self.model_sufix)
            os.makedirs(dash_path_neld, exist_ok=True) 
            remove_directory(dash_path_neld) 
 

        for nn in 'true_0':
            remove_directory(os.path.join(self.neld_path_true,  nn))
            # remove_directory(os.path.join(self.file_path_model_data, self.neld_path_init,'result',nn))
            remove_directory(os.path.join(self.file_path_model_data, 'result',nn))

class neld_dataset:
    def __init__(self, neld_path_inits=None, neld_names=None, neld_namess=None, 
                 line_num_points_hmod_thre=8000,
                 line_num_points_inter_hmod_thre=800,
                 cts=6,
                 stoppage=3,
                 zoom_threshold=5000,
                 size_threshold=50):
        self.neld_path_inits = neld_path_inits if neld_path_inits else []
        self.neld_names = neld_names if neld_names else []
        self.neld_namess = neld_namess if neld_namess else []
        self.line_num_points_hmod_thre = line_num_points_hmod_thre
        self.line_num_points_inter_hmod_thre = line_num_points_inter_hmod_thre
        self.cts = cts
        self.stoppage = stoppage
        self.zoom_threshold = zoom_threshold
        self.size_threshold = size_threshold

    def add_data(self, neld_path_init, neld_name, neld_names_set):
        """Adds a new model dataset."""
        self.neld_path_inits.append(neld_path_init)
        self.neld_names.append(neld_name)
        self.neld_namess.extend(neld_names_set)

    def __add__(self, other):
        """Combines two neld_dataset instances using the + operator."""
        if not isinstance(other, neld_dataset):
            raise TypeError("Can only add two neld_dataset instances.")

        return neld_dataset(
            neld_path_inits=self.neld_path_inits + other.neld_path_inits,
            neld_names=self.neld_names + other.neld_names,
            neld_namess=self.neld_namess + other.neld_namess,
            line_num_points_hmod_thre=self.line_num_points_hmod_thre,  # Keep the same params
            line_num_points_inter_hmod_thre=self.line_num_points_inter_hmod_thre,
            cts=self.cts,
            stoppage=self.stoppage,
            zoom_threshold=self.zoom_threshold,
            size_threshold=self.size_threshold
        )

    def __repr__(self):
        return (f"neld_dataset(neld_names={self.neld_names}, "
                f"neld_path_inits={self.neld_path_inits}, "
                f"neld_namess={self.neld_namess})")







import re

def safe_id(raw: str) -> str: 
    return re.sub(r'[.\{\}/\\ ]+', '_', raw)





class get_app_param(get_name ):
    def __init__(self,
                 dropdown_path_head_option=None,
                 dropdown_model_suf_option=None,
                 dropdown_path_option=None,
                 dropdown_true_keys_option=None,):
        self.dropdown_path_head_option=dropdown_path_head_option
        self.dropdown_model_suf_option=dropdown_model_suf_option
        self.dropdown_path_option=dropdown_path_option
        self.dropdown_true_keys_option=dropdown_true_keys_option
        pass

        get_name.__init__(self) 
        self.dropdown_options_style=dropdown_options_style = {'color': 'white', 'background-color': 'gray'} 
        self.styles = {
            'pre': {
                'border': 'thin lightgrey solid',
                'overflowX': 'scroll'
            }
        }
        self.box_style = {
            'width': '100%',
            'padding': '3px',
            'font-size': '20px',
            'text-align-last': 'center',
            'margin': 'auto',  # Center-align the dropdown horizontally
            'background-color': 'black',
            'color': 'black'
        }



        self.template=template= [ "seaborn", "plotly_dark","plotly", "plotly_white", "ggplot2", "simple_white", "none"]
        # Color and style settings
        self.tcouleur = 'plotly_dark' 
        self.bcouleur = 'navy'
        self.fcouleur = ['white','black','grey']
        self.fsize = 16
        self.dropdown_template_option = []  
        for temp in template:
            self.dropdown_template_option.append({'label': f'{temp}', 
                                                    'value': f'{temp}', 
                                                    'style': dropdown_options_style})  


    '''

    def model_train(self):
        dropdown_options_style=self.dropdown_options_style
        self.dropdown_mode_option=dropdown_mode_option= [
                            {'label': 'Algorithm',      'value': 'algorithm', 'style': dropdown_options_style},
                            {'label': 'Comparison',     'value': 'comparison','style': dropdown_options_style}, 
                            {'label': 'Skeleton',      'value': 'skeleton', 'style': dropdown_options_style},
                            {'label': 'Results',        'value': 'result',    'style': dropdown_options_style}, 
                            {'label': 'Results Optimum','value': 'result_opt','style': dropdown_options_style}, 
                    ] 

        action_name='mode'
        self.dropdown_mode=dropdown_mode ={
                        'option':dropdown_mode_option,
                        'id'         :f'dropdown_{action_name}',
                        'value'      :'result',
                        'placeholder':f'Select {action_name}', 
        }


        ################################ METRIC #########################################


        self.dropdown_plot_option=[
                            # {'label': 'Distribution', 'value': 'dist',     'style': dropdown_options_style} ,
                            # {'label': 'Density',      'value': 'den',      'style': dropdown_options_style},
                            {'label': 'Accuracy',              'value': 'accuracy','style': dropdown_options_style},
                            {'label': 'IOU',                   'value': 'iou',     'style': dropdown_options_style},
                            {'label': 'Confusion Matrix',      'value': 'conf',    'style': dropdown_options_style},
                            {'label': 'Classification Report', 'value': 'report',  'style': dropdown_options_style},
                            {'label': 'Compare',               'value': 'compare', 'style': dropdown_options_style},
                            {'label': 'Logit',                 'value': 'logit',   'style': dropdown_options_style},
        ]
        ################################ INTENSITY #########################################
        self.dropdown_intensity_option= [
                            {'label': 'Clusterization',  'value': 'kmean',       'style': dropdown_options_style},
                            {'label': 'Segmentation',    'value': 'cluu_rad',    'style': dropdown_options_style},
                            {'label': 'smods',          'value': 'smods',      'style': dropdown_options_style},
                            {'label': 'smods True',     'value': 'smods true', 'style': dropdown_options_style},
                            {'label': 'Gauss Curvature', 'value': 'gauss',       'style': dropdown_options_style},
                            {'label': 'Mean Curvature',  'value': 'mean',        'style': dropdown_options_style},
                    ]

'''
    def model_test(self):
        dropdown_options_style=self.dropdown_options_style
        self.dropdown_mode_option= [
                            # {'label': 'Image',         'value': 'algorithm', 'style': dropdown_options_style},
                            # {'label': 'Comparison',        'value': 'comparison',    'style': dropdown_options_style},  
                            # {'label': 'Skeleton',      'value': 'skeleton', 'style': dropdown_options_style},
                            # {'label': 'Accuracy',              'value': 'accuracy','style': dropdown_options_style}, 
                            # {'label': 'ROC Curve',              'value': 'roc_curve','style': dropdown_options_style}, 
                            {'label': 'BE',                 'value': 'logit',   'style': dropdown_options_style},
                            {'label': 'Invar_Measure',                 'value': 'INV_MEASURE',   'style': dropdown_options_style},
                    ]

        for mmnn in  self.inten_file_model_smod_loss:
            self.dropdown_mode_option.append({'label': mmnn,      'value': mmnn, 'style': dropdown_options_style},)
 
        '''
        # dropdown_mode_option.append({'label': 'IOU',      'value': 'IOU', 'style': dropdown_options_style},)INV_MEASURE
        for mmnn in self.inten_file_model_head_neck:
            dropdown_mode_option.append({'label': mmnn,      'value': mmnn, 'style': dropdown_options_style},)
 

        # metric_name=self.metrics_keys
        metric_name=[]


        metric_name.extend(['heatmap_cylinder','heatmap_cylinder_surface'])
        metric_name.extend(self.metrics_keys)
        # metric_name.extend(['heatmap_iou','heatmap_iou_union','histogram_iou'])
        self.metric_mapping = {
            "name": metric_name,
            "title": [f"{mm} Histogram" for mm in metric_name],
            "index": {mm: ii for ii, mm in enumerate(metric_name)},
            "xtitle": "Length",
            "ytitle": "Count",
        }
        self.metrics_combine={'diam_head_neck_length'  :{'key':  ['head_diameter','neck_diameter','smod_length'],
                                                       'label':['Head Diameter','Neck Diameter','smod Length']},
                              'vol_area_length_smod':{'key':['smod_vol','smod_area','smod_length'],
                                                        'label':['smod Volume','smod Area','smod Length']},
                              'vol_head_neck_smod'    :{'key':['head_vol','neck_vol','smod_vol'],
                                                     'label':['Head Volume','Neck Volume','smod Volume'],
                                                     },
                              'area_head_neck_smod'  :{'key':['head_area','neck_area','smod_area'],
                                                      'label':['Head Area','Neck Area','smod Area']},
                              'length_head_neck_smod':{'key':['head_length','neck_length','smod_length'],
                                                        'label':['Head Length','Neck Length','smod Length']},
                              }
        keys=list(self.metrics_combine.keys())
        self.metric_mapping_combine = {
            "name": keys,
            "title": [f"{self.metrics_combine[mm]['label'][0]} vs {self.metrics_combine[mm]['label'][1]} vs {self.metrics_combine[mm]['label'][2]}" for mm in keys],
            "index": {mm: ii for ii, mm in enumerate(keys)},
            "xtitle": [f"{self.metrics_combine[mm]['label'][0]}" for mm in keys],
            "ytitle":[f"{self.metrics_combine[mm]['label'][1]}" for mm in keys],
        }
        for mo in self.metric_mapping_combine['name']:
            dropdown_mode_option.append({'label': mo,        'value': mo,    'style': dropdown_options_style}) 
        for mo in self.metric_mapping['name']:
            dropdown_mode_option.append({'label': mo,        'value': mo,    'style': dropdown_options_style})
            self.dropdown_plot_option=
 '''
    
        action_name='mode'
        self.dropdown_mode ={
                        'option':self.dropdown_mode_option,
                        'id'         :f'dropdown_{action_name}',
                        'value'      :self.dropdown_mode_option[0]['value'],
                        'placeholder':f'Select {action_name}', 
        }
        ################################ METRIC #########################################


        # self.dropdown_plot_option=[ 
        #                     {'label': 'Accuracy',              'value': 'accuracy','style': dropdown_options_style},
        #                     {'label': 'IOU',                   'value': 'iou',     'style': dropdown_options_style},
        #                     {'label': 'Confusion Matrix',      'value': 'conf',    'style': dropdown_options_style},
        #                     {'label': 'Classification Report', 'value': 'report',  'style': dropdown_options_style},
        #                     {'label': 'Compare',               'value': 'compare', 'style': dropdown_options_style},
        #                     {'label': 'Logit',                 'value': 'logit',   'style': dropdown_options_style},
        # ] 

        self.dropdown_intensity_option=[]
        for intt,name in zip(self.inten_file_sub[1:],self.inten_file_sub_name[1:]):
            self.dropdown_intensity_option.append({'label': name,   'value': intt,       'style': dropdown_options_style})
        for intt in self.inten_file_train:
            self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style}) 
        # for intt in self.inten_pca:
        #     self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style})  
        for intt in self.inten_file:
            self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style})  
        for intt in self.base_features_dict.keys():
            self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style}) 
        for intt in self.intensity_logit:
            self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style}) 
        for intt in self.intensity_head_neck_logit:
            self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style}) 



    def more_param(self,id_name_end,model_type,model_sufix,path_dir,neld_name): 
        dropdown_options_style=self.dropdown_options_style 
        self.neld_name=neld_name 

        action_name='template' 
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_template={
            'option'     :self.dropdown_template_option,
            'id'         :id_name,
            'value'      :self.dropdown_template_option[1]['value'],
            'placeholder':f'Select {action_name}', 
        }
        action_name='intensity'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_intensity ={
        'option':self.dropdown_intensity_option,
        'id'         :id_name,
        'value'      :self.dropdown_intensity_option[0]['value'],
        'placeholder':f'Select {action_name}', 
        }

        action_name='metric'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_plot={
        'option':self.dropdown_mode_option,
        'id'          :id_name,
        'value'       :self.dropdown_mode_option[0]['value'],
        'placeholder' :f'Select {action_name}', 
        } 

        self.output_graph_1={
            'id':f'output-graph1_{action_name}_{id_name_end}', 
            'style':{'display': 'flex', 'justify-content': 'center'}
        }

        self.output_text_1={
            'id':f'output-text1_{action_name}_{id_name_end}', 
            # 'style':{'display': 'flex', 'justify-content': 'center'}
        } 

        self.height_slider={
                        'id':f'height-slider_{action_name}_{id_name_end}',
                        'min':300,
                        'max':900,
                        'step':50,
                        'value':750, 
                        'marks':{i: f'{i}px' for i in range(300, 1000, 200)}, 
        }

        self.width_slider={
                        'id':f'width-slider_{action_name}_{id_name_end}',
                        'min':400,
                        'max':1200,
                        'step':50,
                        'value':1000, 
                        'marks':{i: f'{i}px' for i in range(400, 1200, 200)}, 
        }

        self.iou_slider={
                        'id':f'iou-slider_{action_name}_{id_name_end}',
                        'min':0,
                        'max':100,
                        'step':1,
                        'value':70, 
                        'marks':{i: f'{i}%' for i in range(0, 100, 20)}, 
        } 
 

        self.hist_slider={
                        'id':f'iou-slider_{action_name}_{id_name_end}',
                        'min':-1,
                        'max':10,
                        'step':1,
                        'value':0, 
                        'marks':{i: f'{i}' for i in range(0, 10, 2)}, 
        } 
 
   
        action_name='true_keys'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_true_keys={
            'option'     :self.dropdown_true_keys_option,
            'id'         :id_name,
            'value'      :self.dropdown_true_keys_option[0]['value'],
            'placeholder':f'Select {action_name}', 
        }

        action_name='path_head'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_path_head={
            'option'     :self.dropdown_path_head_option,
            'id'         :id_name,
            # 'value'      :self.dropdown_path_head_option[0]['value'],
            'value' : model_type,
            'placeholder':f'Select Type of Model', 
        }

        action_name='model_suf'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_model_suf={
            'option'     :self.dropdown_model_suf_option,
            'id'         :id_name,
            'value'      :f'{model_sufix}',
            'placeholder':f'Select Feature Class', 
        }
 
        action_name='path'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_path={
            'option'     :self.dropdown_path_option,
            'id'         :id_name,
            # 'value'      :self.dropdown_path_option[0]['value'],
            'value':  path_dir,
            'placeholder':f'Select Weight', 
        } 
        dropdown_neld_option = [{'label': 'Initial', 'value': 'init', 'style': dropdown_options_style} ,
                                {'label': f'Smoothed', 'value': 'smooth', 'style': dropdown_options_style}]
        action_name='model curve state'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_neld={
            'option'     :dropdown_neld_option,
            'id'         :id_name,
            'value'      :dropdown_neld_option[0]['value'],
            'placeholder':f'Select {action_name}', 
        }
 

    def get_dropdown_cluster_test(self,id_name_end,count):
        dropdown_options_style=self.dropdown_options_style 
        dropdown_cluster_option = [{'label': 'model', 'value': 0, 'style': dropdown_options_style}] 
        dropdown_cluster_option.append({'label': f'hmod', 'value': 1, 'style': dropdown_options_style}) 
        ik=0
        jj=0
        scatter=[]
        for iii in count:  
                dropdown_cluster_option.append({'label': f'sp({iii:02})', 'value': iii+2, 'style': dropdown_options_style}) 
        
        action_name='cluster'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_cluster={
            'option'     :dropdown_cluster_option,
            'id'         :id_name,
            'value'      :dropdown_cluster_option[1]['value'],
            'placeholder':f'Select {action_name}', 
        }


    def get_dropdown_cluster(self,id_name_end,count):
        dropdown_options_style=self.dropdown_options_style 
        dropdown_cluster_option =     [{'label': 'model', 'value': 0, 'style': dropdown_options_style}] 
        dropdown_cluster_option.append({'label': f'hmod'  , 'value': 1, 'style': dropdown_options_style})
        jjjj=2 
        for iii in count: 
            dropdown_cluster_option.append({
                'label': f'sp({iii:02})' if isinstance(iii, int) else f'sp_ap({iii[0]:02})~sp_an({iii[1]:02})', 
                'value':jjjj,
                'style': dropdown_options_style
            })
            jjjj+=1
        action_name='cluster'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_cluster={
            'option'     :dropdown_cluster_option,
            'id'         :id_name,
            'value'      :dropdown_cluster_option[0]['value'],
            'placeholder':f'Select {action_name}', 
        }



    def get_dropdown_index(self,id_name_end,neld_names,index):
        dropdown_options_style=self.dropdown_options_style  
        dropdown_index_option=[{
                'label': f'{iii}', 
                'value':ii,
                'style': dropdown_options_style
            } for ii,iii in enumerate(neld_names)] 
        action_name='index'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_index={
            'option'     :dropdown_index_option,
            'id'         :id_name,
            'value'      :index, 
            'placeholder':f'Select {action_name}', 
        } 






def get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir,path_list=['hmod_path','smod_path',] ): 
    model = {
        path: {'data': {}, 'dest': {}}
        # for path in ['hmod_path','smod_path', 'smod_path_pre', ] 
        for path in path_list
                     
    }
    path_train = {}  
    for md in model:
        path_train[f'data_{md}'] = f"{data_head}_{data_sufix}_{data_dir}"
        path_train[f'dest_{md}'] = f"{dest_head}_{dest_sufix}_{dest_dir}"


    # path_train['data_hmod_vertices_center_path']=f"{model['hmod_path']['data']['head']}_{model['hmod_path']['data']['sufix']}_{model['hmod_path']['data']['dir']}"
    # path_train['dest_hmod_vertices_center_path']=f"{model['hmod_path']['dest']['head']}_{model['hmod_path']['dest']['sufix']}_{model['hmod_path']['dest']['dir']}"
    return path_train


 


 
from neld_fun_0.get_path import get_path_train,get_name,get_model_name


class get_data_mode(get_name,get_model_name):
    def __init__(self,mode_ids=[],
                pre_portion=None,
                path_list=['hmod_path','smod_path',  ] ,
                data_mode={}):
        super().__init__() 
        self.data_mode=data_mode
        self.data_mode['model_sufix_all']=[]
        self.data_mode['pinn_dir_data_all']=[]
        self.model_sufix_all=set()
        self.pinn_dir_data_all=set()
        self.data_mode['mode_id']=[]
        self.mode_ids=mode_ids
        self.path_list=path_list
        
        pinn_dir_dest= data_sufix=  data_dir= 'save'  


    def train_pre(self,data_mode=None,
                pre_portion='head_neck',
                pre_opt='pre',
                train_test='train', 
                data_head='true',
                dest_head='true', 
                data_dir=None,
                data_sufix=None,
                dest_dir=None,
                dest_sufix=None,
                seg_neld='save',
                base_features_index=[0,1,2,3] ,
                path_list=None,
                model_init=None,
                ):
        mon=get_model_name(
            pre_opt=pre_opt,
            train_test=train_test, 
            seg_neld=seg_neld,
            dest_head=dest_head,
            pre_portion=pre_portion)
        data_mode=data_mode if data_mode is not None else self.data_mode 
        path_list=path_list if path_list is not None else self.path_list

        

        inten_pinn_index=[]
        mon.vals(base_features_index=base_features_index, inten_pinn_index=inten_pinn_index)
        list_features=[]
        base_features_list=mon.base_features_list
        dest_sufix_pr= mon.dest_sufix 

 

 
        pinn_dir_data=seg_neld 
        dest_dir_pr=mon.dest_dir# 
        
        data_sufix=data_sufix or seg_neld
        dest_sufix=dest_sufix or dest_sufix_pr
        data_dir=data_dir or dest_dir_pr
        dest_dir=dest_dir or dest_dir_pr
 
        self.mode_id=mode_id=mon.mode_id#f'train_{seg_neld}_{pre_portion[:2]}_{dest_head}_{dest_sufix}'
        self.mode_ids.append(mode_id)
        data_mode['mode_id'].append(mode_id)
        data_mode[mode_id]={}
        path_train_pre=get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir,path_list=path_list) 
        data_mode[mode_id]['path_train']= path_train_pre 
        data_mode[mode_id]['model_sufix']=[dest_sufix ]
        data_mode[mode_id]['dest_dir']=dest_dir 
        data_mode[mode_id]['base_features_list']=base_features_list
        data_mode[mode_id]['list_features']=list_features
        data_mode[mode_id]['model_init']=model_init
        data_mode[mode_id]['pre_portion']= pre_portion 
        data_mode[mode_id]['pinn_dir_data']= pinn_dir_data   
        data_mode[mode_id]['seg_neld']=seg_neld
        data_mode[mode_id]['get_training']=True 
        data_mode[mode_id]['get_hmod']=True  
        data_mode[mode_id]['get_segm']=True 
        data_mode[mode_id]['get_head_neck_segm']=True  
        data_mode[mode_id]['get_segss_group']=False
        data_mode[mode_id]['train_smods']=False  
        self.dest_sufix_pr=dest_sufix_pr
        self.dest_dir_pr=dest_dir_pr
        self.pinn_dir_data=pinn_dir_data 
        self.model_sufix_all.update([dest_sufix, data_sufix])
        self.pinn_dir_data_all.update([data_dir, dest_dir])


        return data_mode





    def train_opt(self,data_mode=None,
            pre_opt='opt',
            train_test='train', 
                pre_portion='head_neck',
                data_head='true',
                dest_head='true', 
                seg_neld='save',
                data_dir=None,
                data_sufix=None,
                dest_dir=None,
                dest_sufix=None,
                path_list=None,
                base_features_index=[0,1,2,3],
        inten_pinn_index=[0,1],
        model_init=None,
                ):
        mon=get_model_name(
            pre_opt=pre_opt,
            train_test=train_test, 
            seg_neld=seg_neld,
            dest_head=dest_head,
            pre_portion=pre_portion)
        data_mode=data_mode if data_mode is not None else self.data_mode
        path_list=path_list if path_list is not None else self.path_list


        data_sufix=dest_sufix_pr=data_sufix or self.dest_sufix_pr
        data_dir=dest_dir_pr=data_dir or self.dest_dir_pr
        pinn_dir_data=self.pinn_dir_data
        # data_hmod_train_path=self.data_hmod_train_path
        # data_hmod_vertices_center_path=self.data_hmod_vertices_center_path
        # dest_hmod_vertices_center_path_opt=self.dest_hmod_vertices_center_path_opt 
 
   
        mon.vals(base_features_index=base_features_index,
                 inten_pinn_index=inten_pinn_index) 
        base_features_list=mon.base_features_list
        dest_sufix=dest_sufix or  mon.dest_sufix
        self.mode_id=mode_id=mon.mode_id
        self.mode_ids.append(mode_id)
        list_features=[[mon.inten_pinn_tmp[nam]['id_path'], mon.inten_pinn_tmp[nam]['name'] ] for nam in inten_pinn_index] 
        dest_dir=mon.dest_dir  

  
        data_mode['mode_id'].append(mode_id)
        data_mode[mode_id]={}


        path_train_opt=get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir,path_list=path_list) 
        # path_train_opt=get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir) 
        data_mode[mode_id]['path_train']= path_train_opt#get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir)
        data_mode[mode_id]['model_sufix']=[dest_sufix ]
        data_mode[mode_id]['dest_dir']=dest_dir  
        data_mode[mode_id]['base_features_list']=base_features_list
        data_mode[mode_id]['list_features']=list_features
        data_mode[mode_id]['model_init']=model_init
        data_mode[mode_id]['pre_portion']= pre_portion  
        data_mode[mode_id]['pinn_dir_data']= pinn_dir_data  
        data_mode[mode_id]['seg_neld']=seg_neld
        data_mode[mode_id]['get_training']=True  
        data_mode[mode_id]['get_hmod']=True 
        data_mode[mode_id]['get_segm']=True 
        data_mode[mode_id]['get_head_neck_segm']=True 
        data_mode[mode_id]['dest_path']='dest_smod_path' 
        data_mode[mode_id]['get_segss_group']=True
        data_mode[mode_id]['train_smods']=False  

        '''
        path_train_opt['data_hmod_vertices_center_path']=path_train_opt['data_true_iou'] 
        data_mode[mode_id]['path_train']['data_train_path']=data_hmod_train_path 
        data_mode[mode_id]['path_train']['data_hmod_path']=data_hmod_vertices_center_path
        data_mode[mode_id]['path_train']['data_smod_path_center']=dest_hmod_vertices_center_path_opt
        data_mode[mode_id]['path_train']['dest_smod_path_center']=data_mode[mode_id]['path_train']['dest_smod_path'] '''
 
        # self.model_sufix_all.extend([dest_sufix,data_sufix]) 
        # self.model_sufix_all=list(set(self.model_sufix_all))
        # self.pinn_dir_data_all.extend([data_dir ,dest_dir, ])
        # self.pinn_dir_data_all=list(set(self.pinn_dir_data_all))
        self.model_sufix_all.update([dest_sufix, data_sufix])
        self.pinn_dir_data_all.update([data_dir, dest_dir])
        
        return data_mode




    def test_pre(self,data_mode=None,
                pre_opt='pre',
                train_test='test',
                 pre_portion='head_neck',
                data_head='pinn',
                dest_head='pinn', 
                seg_neld='save' ,
                data_dir=None,
                dest_dir=None,
                data_sufix=None,
                dest_sufix=None,
                path_list=None,
                base_features_index=[0,1,2,3] ,
                model_init=None,
                Nmarkov=None,
                Nsample=None,
                nSample_interval=1,
                nTest=1,
                g_name=None,
                tf_initial_bcs=True,
                tf_final_bcs=True,
                Samples_nbr=None,
                tf_train=True,
                ):
        mon=get_model_name(
            pre_opt=pre_opt,
            train_test=train_test, 
            seg_neld=seg_neld,
            dest_head=dest_head,
            pre_portion=pre_portion)
        
        data_mode=data_mode if data_mode is not None else self.data_mode
        path_list=path_list if path_list is not None else self.path_list

  
        #____________________________________'pre_non_full'________________________________ 
    
        inten_pinn_index=[]
        '''
        mon.vals(base_features_index=base_features_index, inten_pinn_index=inten_pinn_index) 
        self.mode_id=mode_id=mon.mode_id
        self.mode_ids.append(mode_id)
        base_features_list=mon.base_features_list
        dest_sufix=dest_sufix_pr=data_sufix_pr=mon.dest_sufix
        list_features=[[mon.inten_pinn_tmp[nam]['id_path'], mon.inten_pinn_tmp[nam]['name'] ] for nam in inten_pinn_index] 
 
        dest_dir=dest_dir or mon.dest_dir  

        data_sufix=data_sufix or data_sufix_pr#'save'
        dest_sufix=dest_sufix or dest_sufix_pr
        data_dir=data_dir or dest_dir#data_dir_pr #'save'#
        # dest_dir=dest_dir#dest_dir_pr 
        # path_train_pre=get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir) 
        # mode_id=f'test_{seg_neld}_{pre_portion[:2]}_{dest_head}_{dest_sufix}'
'''
        self.mode_id=mode_id='_'.join([mm for mm in [data_head,data_sufix,data_dir] if mm is not None])

        data_mode['mode_id'].append(mode_id)
        data_mode[mode_id]={} 
        data_mode[mode_id]['path_train']= get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir,path_list=path_list) 
        data_mode[mode_id]['model_sufix']=[dest_sufix] 
        data_mode[mode_id]['dest_dir']=dest_dir 
        # data_mode[mode_id]['base_features_list']=base_features_list
        # # data_mode[mode_id]['list_features']=[] 
        # data_mode[mode_id]['model_init']=model_init
        # data_mode[mode_id]['pre_portion']= pre_portion 
        # data_mode[mode_id]['pinn_dir_data']= seg_neld 
        # data_mode[mode_id]['seg_neld']=seg_neld
        # data_mode[mode_id]['get_training']=False 
        # data_mode[mode_id]['get_hmod']=True  
        # data_mode[mode_id]['get_segm']=True 
        # data_mode[mode_id]['get_head_neck_segm']=True 
        # data_mode[mode_id]['get_segss_group']=False
        # data_mode[mode_id]['train_smods']=False 
        data_mode[mode_id]['pinn']=dict(            
                                        Nmarkov=Nmarkov,
                                        Nsample=Nsample,
                                        nSample_interval=nSample_interval,
                                        nTest=nTest,
                                        g_name=g_name,
                                        tf_initial_bcs=tf_final_bcs,
                                        tf_final_bcs=tf_final_bcs,
                                        Samples_nbr=Samples_nbr,
                                        tf_train=tf_train,
                                )
        # self.dest_sufix_pr=dest_sufix_pr
        self.dest_dir_pr=dest_dir
        self.pinn_dir_data=seg_neld 

        self.model_sufix_all.update([dest_sufix, data_sufix])
        self.pinn_dir_data_all.update([data_dir, dest_dir])
        return data_mode




    def test_opt(self,
                data_mode=None,
                pre_opt='opt',
                train_test='test',
                pre_portion='head_neck',
                data_head='pinn',
                dest_head='pinn', 
                seg_neld='save' ,
                data_dir=None,
                data_sufix=None,
                dest_dir=None,
                dest_sufix=None,
                base_features_index=[0,1,2,3] ,
                inten_pinn_index=[0],
                path_list=None,
                model_init=None,
                Nmarkov=None,
                Nsample=None,
                nSample_interval=1,
                nTest=1,
                g_name=None,
                tf_initial_bcs=True,
                tf_final_bcs=True,
                Samples_nbr=None,
                tf_train=True,
                inter=None,
                pm=None,
                rhs_dim=None,
                tf_mean=None,
                 ): 
        data_mode=data_mode if data_mode is not None else self.data_mode 
        path_list=path_list if path_list is not None else self.path_list
        '''
        data_sufix=data_sufix or self.dest_sufix_pr
        data_dir=data_dir or self.dest_dir_pr
        pinn_dir_data=self.pinn_dir_data'''
        # data_hmod_train_path=self.data_hmod_train_path
        # data_hmod_vertices_center_path=self.data_hmod_vertices_center_path
        # dest_hmod_vertices_center_path_opt=self.dest_hmod_vertices_center_path_opt 


 
        #--------------------------- OPT opt_non_full_gmg2m2_sh_vcv_di_sid_pca123________________________________ 
  

        # model_name='vcv'
        # inten_pca=['hmod_vcv_length']
        # list_features=[['data_smod_path_center',f'{nam}.txt'] for nam in inten_pca] 

        
        # mon.vals(base_features_index=base_features_index,
        #          inten_pinn_index=inten_pinn_index) 
        # base_features_list=mon.base_features_list
        # dest_sufix=dest_sufix or mon.dest_sufix
        # dest_dir=dest_dir or mon.dest_dir
        # list_features=[[mon.inten_pinn_tmp[nam]['id_path'], mon.inten_pinn_tmp[nam]['name'] ] for nam in inten_pinn_index] 
   
        # self.mode_id=mode_id=mon.mode_id  
        self.mode_id=mode_id='_'.join([mm for mm in [data_head,data_sufix,data_dir] if mm is not None])
        self.mode_ids.append(mode_id)
 

        data_mode['mode_id'].append(mode_id)
        data_mode[mode_id]={}
 
        
        data_mode[mode_id]['path_train']= get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir,path_list=path_list)
        data_mode[mode_id]['model_sufix']=[dest_sufix] 
        # data_mode[mode_id]['base_features_list']=base_features_list
        # data_mode[mode_id]['list_features']=list_features
        data_mode[mode_id]['model_init']=model_init
        data_mode[mode_id]['pre_portion']= pre_portion  
        # data_mode[mode_id]['pinn_dir_data']= pinn_dir_data  
        # data_mode[mode_id]['seg_neld']=seg_neld
        data_mode[mode_id]['dest_path']='dest_smod_path' 
        # data_mode[mode_id]['get_training']=False  
        # data_mode[mode_id]['get_hmod']=True 
        # data_mode[mode_id]['get_segm']=True 
        # data_mode[mode_id]['get_segm']=True 
        # data_mode[mode_id]['get_head_neck_segm']=True 
        # data_mode[mode_id]['get_segss_group']=False
        # data_mode[mode_id]['train_smods']=False   
        data_mode[mode_id]['pinn']=dict(            
                                        Nmarkov=Nmarkov,
                                        Nsample=Nsample,
                                        nSample_interval=nSample_interval,
                                        nTest=nTest,
                                        g_name=g_name,
                                        tf_initial_bcs=tf_final_bcs,
                                        tf_final_bcs=tf_final_bcs,
                                        Samples_nbr=Samples_nbr,
                                        tf_train=tf_train,
                                        inter=inter,
                                        pm=pm,
                                        rhs_dim=rhs_dim,
                                        tf_mean=tf_mean,
                                )
        # data_mode['pinn_dir_data_all']=list(set(data_mode['pinn_dir_data_all']))f'{dest_dir}_pre' 
        # data_mode['model_sufix_all']=list(set(data_mode['model_sufix_all']))f'{data_dir}_pre', 
        # self.model_sufix_all.extend([dest_sufix,data_sufix]) 
        # self.model_sufix_all=list(set(self.model_sufix_all))
        # self.pinn_dir_data_all.extend([data_dir,dest_dir])
        # self.pinn_dir_data_all=list(set(self.pinn_dir_data_all))
        self.model_sufix_all.update([dest_sufix, data_sufix])
        self.pinn_dir_data_all.update([data_dir, dest_dir])


        return data_mode

 

 

