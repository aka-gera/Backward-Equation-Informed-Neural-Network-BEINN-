 
import time 
import numpy as np
import copy
from neld_pinn_0.CompFun import Fun
from neld_pinn_0.param import PBCs,particles,saves 

'''
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

from help_fun import aka_grad
from Graph import mygraph
from help_fun import aka_fun,help_fun
from NELD_PINN import lang_loss, neld_coef
from PINN import PINN, aka_train
'''


import pickle
import os
path_0=os.getcwd()


# np.random.seed(1)
            # choose the type of the flow (i.e 'eld', 'shear',  or 'pef')
                # choose the type of the flow (i.e 'eld', 'shear',  or 'pef')

np.random.seed(1062)
               # choose the type of the flow (i.e 'eld', 'shear',  or 'pef')
                   # choose the type of the flow (i.e 'eld', 'shear',  or 'pef')

'''
force='zero'
force='harmonic'
flow = 'eld' 
flow = 'shear'  
flow='pef'    
nPart = 1                         # Number of particles
epsilon = 2.0                     # rate of the deformation of the background flow
rcut = 30                         # radius cut
beta,gamma=1,1
N = 50                             # number of steps in a period
Nperiod =  10000                   # number of periods
Nperiod =  1                  # number of periods
N = 200                             # number of steps in a period

N_grid = 10
'''
def get_neld_data(self,flow, epsilon, nPart, rcut, N, Nperiod,force,beta,gamma,
                # path_gen,
                  dt_r_0=3,
                  dt_r1=1,                  
                fmt='%.4f',
                Nmarkov=1000,
                # Nsample=50,
                disp_infos=True,
    ):
    pm = PBCs(flow, epsilon, nPart, rcut, N, Nperiod,force=force,
            beta=beta,
            gamma=gamma)  # get the parameters
    # pm.dt = 1.0/50
    X = particles(pm)
    X_init_0=Fun().initializez(pm,X)
    dt_r=3
    X_init={ii:X_init_0 for ii in range(dt_r_0)}
    X={ii:particles(pm) for ii in range(dt_r_0)}
    sav_part = saves(pm)

    # sav_part={ii:saves(pm) for ii in range(dt_r)}
    Fun_integ={mm:nn for mm,nn in zip(['fine','coarse'],[Fun().EM,Fun().EM])}
    Fun_integ={mm:nn for mm,nn in zip(['fine','coarse'],[Fun().SOILE_B,Fun().SOILE_B])}
    Fun_integ={mm:nn for mm,nn in zip(['fine','coarse'],[Fun().SOILE_A,Fun().SOILE_A])} 
    Fun_integ={mm:nn for mm,nn in zip(['fine','coarse'],[Fun().SOILE_B,Fun().EM])}
    Fun_integ={mm:nn for mm,nn in zip(['fine','coarse'],[Fun().SOILE_B,Fun().SOILE_A])}

    # path_gen=os.path.join(path_0,f'data_{flow}_{nPart}_{force}',)





    print(f'get_neld_data started')
    print('--------------------------------')


    path_gen=self.obj_org_path
    os.makedirs(path_gen,exist_ok=True)
    path = os.path.join(path_gen,"simulation_parameters.txt")

    with open(path, "w") as f:
        f.write(f"flow = {flow}\n")
        f.write(f"nPart = {nPart}\n")
        f.write(f"epsilon = {epsilon:.3f}\n")
        f.write(f"beta = {beta:.3f}\n")
        f.write(f"gamma = {gamma:.3f}\n")
        f.write(f"force = {force}\n")
        f.write(f"rcut = {rcut}\n")
        f.write(f"N = {N}\n")
        f.write(f"Nperiod = {Nperiod}\n")
        f.write(f"Nmarkov = {Nmarkov}\n")

    print(f"Parameters saved to {path}")






    path_init_param=os.path.join(path_gen,f"param.pkl")
    with open(path_init_param, "wb") as f:
            pickle.dump(pm, f) 








    time_start=time.time() 
    # for index in range(Nsample):            
    for  index,dend_name in enumerate(self.dend_names):
        self.get_dend_name(index=index, ) 
        # path_1=os.path.join(path_gen,f'data_{index}')
        path_1=self.dend_path_org_new
        os.makedirs(path_1,exist_ok=True)



        path_init=os.path.join(path_1,f"save.pkl")

        with open(path_init_param, "rb") as f:
            pm = pickle.load(f)
        

        Ntime= pm.N
        Ndim = pm.dim
        Nperiod = pm.Nperiod
        Nlast=(Nperiod-1)*Ntime

        get_new_data=True
        get_new_data=os.path.exists(path_init)
        if not get_new_data:
            # pm.dt = 1.0/50
            X = particles(pm)
            X_init_0=Fun().initializez(pm,X)
            # dt_r=3
            X_init={ii:X_init_0 for ii in range(dt_r)}
            X={ii:particles(pm) for ii in range(dt_r)}
            sav_part = saves(pm)

            _,XX=Fun().Simulation_init(pm, X_init,sav_part,  dt_r=dt_r1,Fun_integ=Fun_integ)
            Tinit=dict(count=0,
                    range_0=0,
                    range_N=1)
            sav_all,XX = Fun().Simulation_prof(pm,XX, sav_part,Tinit=Tinit,dt_r=dt_r,Fun_integ=Fun_integ)




            path_time=os.path.join(path_1,f"time.txt")
            path_qq=os.path.join(path_1,f"qq_init.txt")
            path_pp=os.path.join(path_1,f"pp_init.txt")
            path_ff=os.path.join(path_1,f"ff_init.txt")  
            np.savetxt(path_qq,sav_all.qq[:Ntime,:],fmt=fmt)
            np.savetxt(path_pp,sav_all.pp[:Ntime,:],fmt=fmt)
            np.savetxt(path_ff,sav_all.fDist[:Ntime,:],fmt=fmt)
            np.savetxt(path_time,sav_all.time[:Ntime,:],fmt=fmt)

            
            with open(path_init, "wb") as f:
                pickle.dump([sav_all,XX], f) 


    

        with open(path_init, "rb") as f:
            sav_all,XX = pickle.load(f)



        mytime0 = time.time() - time_start

        hours, rem = divmod(mytime0, 3600)
        minutes, seconds = divmod(rem, 60)
        if disp_infos:
            print(f'Data generation completed on {dend_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')



        # dt_r=1 
        Tinit=dict(count=pm.N,
                range_0=1,
                range_N=pm.Nperiod) 

        for ii in range(Nmarkov):
            sav_part = saves(pm) 

            XXc=copy.deepcopy(XX)
            sav=copy.deepcopy(sav_all)
            sav,_ = Fun().Simulation_prof(pm,XXc, sav,Tinit=Tinit,dt_r=dt_r,Fun_integ=Fun_integ) 

            path_qq=os.path.join(path_1,f"qq_{ii}.txt")
            path_pp=os.path.join(path_1,f"pp_{ii}.txt")
            path_ff=os.path.join(path_1,f"ff_{ii}.txt")
            np.savetxt(path_qq,sav.qq[Ntime:,:],fmt=fmt)
            np.savetxt(path_pp,sav.pp[Ntime:,:],fmt=fmt)
            np.savetxt(path_ff,sav.fDist[Ntime:,:],fmt=fmt)

            '''
            path_qq=os.path.join(path_1,f"qq_{ii}.txt")
            path_pp=os.path.join(path_1,f"pp_{ii}.txt")
            path_ff=os.path.join(path_1,f"ff_{ii}.txt")
            np.savetxt(path_qq,sav.qq[Nlast:,:],fmt=fmt)
            np.savetxt(path_pp,sav.pp[Nlast:,:],fmt=fmt)
            np.savetxt(path_ff,sav.fDist[Nlast:,:],fmt=fmt)'''
            
            # path=os.path.join(path_1,f"sample_{ii}.pkl")
            # with open(path, "wb") as f:
            #     pickle.dump(sav, f)





 
import numpy as np
import pickle



# from neld_pinn_0.help_fun import aka_fun,help_fun,get_sample_indices
# from neld_pinn_0.NELD_PINN import neld_coef


from neld_pinn_0.CompFun import Fun
from neld_pinn_0.help_fun import get_sample_indices

def rhs_g(qq=None,pp=None,ff=None,name=None,vol=None): 
    if name =='momen':
        return (1/vol)*pp
    elif name=='posi':
        return (1/vol)*qq
    elif name=='press':
        return (1/vol)*(pp**2+qq**2)
    elif name=='press_0':
        return  (1/vol)*(pp**2+ff) 



def lhs_g(time=None,qq=None,pp=None,qq2=None,pp2=None,ff=None,name='pp',vol=None): 
    base=[time,qq,pp]
    # if name in (['pp','qq']):
    #     pass  
    # elif name=='pressure':
    #     base.extend([qq2,pp2]) 
    # elif name=='pressure_pre_0':
    #     base.extend([pp2,ff]) 
    return base
 

def get_configs(Nmarkov=1000,nTest=5,nSample_interval=1,pm=None,rhs_param=None):


    rhs_dim_i,rhs_dim_j=0,1
    Model_all={  }


 
    for ii,g_name in enumerate(rhs_param['g_names']):
        mo=f'{g_name}'
        Model_all[mo]=dict(
                        data_sufix= mo,
                        dest_sufix= mo,
                        Nmarkov=Nmarkov,
                        Nsample=None,
                        nSample_interval=nSample_interval,
                        nTest=nTest,
                        g_name=g_name,
                        tf_initial_bcs=True,
                        tf_final_bcs=True,
                        Samples_nbr=None,
                        tf_train=True, 
                        inter=None,
                        pm=pm,
                        rhs_dim=dict(zip(['i','j'],[rhs_dim_i,rhs_dim_j])),
                )
         
 

    rhs_dim_is,rhs_dim_js=[[0,0],[1,2]] if rhs_param is None else [rhs_param['dim'][mm] for mm in  ['i','j',]]
    for rhs_dim_i,rhs_dim_j in zip(rhs_dim_is,rhs_dim_js):
        for tf_mean in [True,False]:    
            for ii,g_name in enumerate(rhs_param['g_names']):
                nma=f'mean_{rhs_dim_i}{rhs_dim_j}' if tf_mean else f'{rhs_dim_i}{rhs_dim_j}'
                mo=f'{g_name}--{nma}' 
                Model_all[mo]=dict(
                            data_sufix= mo,
                            dest_sufix= mo,
                            Nmarkov=Nmarkov,
                            Nsample=None,
                            nSample_interval=nSample_interval,
                            nTest=nTest,
                            g_name=g_name,
                            tf_initial_bcs=True,
                            tf_final_bcs=True,
                            Samples_nbr=None,
                            tf_train=True, 
                            inter=None,
                            pm=pm,
                            rhs_dim=dict(zip(['i','j'],[rhs_dim_i,rhs_dim_j])),
                            tf_mean=tf_mean,
                        )
             



    return Model_all













# path_gen=os.path.join(path_0,f'data_{flow}_{nPart}_{force}',)

 






import copy




def get_neld_data_train(self,
                data_sufix= None,
                dest_sufix= None,
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
                    tf_mean=False,

    ):

    path_gen=self.obj_org_path
    
    path_init_param=os.path.join(path_gen,f"param.pkl") 

    with open(path_init_param, "rb") as f:
        pm = pickle.load(f)



    # nSample=1
    if inter is None:
        inter=np.arange(0,pm.N,nSample_interval)  



    rhs_dim=rhs_dim if rhs_dim is not None else dict(zip(['i','j'],[0,pm.Nperiod-1]))
    ntimi,mtimi=pm.nPart*pm.dimm*rhs_dim['i'],pm.nPart*pm.dimm*rhs_dim['j']

    pressure_f,time_f,qq_f,pp_f=[],[],[],[]
    keyss= ['train','test']
    coords,pressure_means,pressure_fs,coords_0=[{nn:[] for nn in keyss} for _ in range(4)]
    Xall=dict(zip(['time','q','p','force'],[{nn:[] for nn in keyss} for _ in range(4)]))
    Xfin={nn:[] for nn in keyss} 
    # Nmarkov = 1000
    coords_all,coords_all=[],[]
    Ntime= pm.N
    Ndim = pm.dim
    Nperiod = pm.Nperiod
    Nlast=(Nperiod-1)*Ntime


    Nterm,Ndc=get_sample_indices(pm.N,pm.Nperiod,nSample=nSample_interval)



    # pm.epsilon = 2.0 
    par=dict(
            flow=pm.flow, 
            epsilon=pm.epsilon, 
            nPart=pm.nPart, 
            rcut=pm.rcut, 
            N=Ndc, 
            Nperiod=pm.Nperiod,
            g_name=g_name,
            Xall=Xall,
            n_classes=pm.nPart*pm.dimm if tf_mean else pm.nPart*pm.dimm*(1+rhs_dim['j']-rhs_dim['i']),
            )
 


    timm=None
    # Nmarkov=1000
    Nsample=Nsample if Nsample is not None else len(self.dend_names) 
    print('[[[[[[[[Ndd]]]]]]]]',Nsample,nTest)
    Samples_nbr = Samples_nbr if Samples_nbr is not None else [np.arange(Nsample-nTest),np.arange(Nsample-nTest,Nsample)]
    bcs_test={}
    print('[[[[[[]]]]]]',Samples_nbr)
    bcs_train_test={}
    for key,val in zip(keyss,Samples_nbr):
        # coords_0[key]=[]
        # coords[key]=[]
        # pressure_means[key]=[]
        # pressure_fs[key]=[]
        print('----------',key)
        time_f,qq_f,pp_f,qq_f2,pp_f2,u_f,ff_f=[],[],[],[],[],[],[]
        for index in val:
            self.get_dend_name(index=index, )  
            path_1=self.dend_path_org_new
            path_init=os.path.join(path_1,f"save.pkl")

            path_time=os.path.join(path_1,f"time.txt")
            path_qq=os.path.join(path_1,f"qq_init.txt")
            path_pp=os.path.join(path_1,f"pp_init.txt")
            path_ff=os.path.join(path_1,f"ff_init.txt")
            time=np.loadtxt(path_time,dtype=float).reshape(-1,1) 
            ff=np.loadtxt(path_ff,dtype=float)
            qq=np.loadtxt(path_qq,dtype=float)
            pp=np.loadtxt(path_pp,dtype=float)
            Xall['time'][key].append(time)
            Xall['q'][key].append(qq)
            Xall['p'][key].append(pp)
            Xall['force'][key].append(ff)
            if timm is None:
                timm = np.tile(time, (1, (pm.Nperiod-1))).T
                timm=timm.ravel().reshape(-1,1)
            # pressure = np.sum(qq**2 , axis=1).reshape(-1,1) 
            pressure = rhs_g(name=g_name,pp=pp,qq=qq,ff=ff,vol=pm.vol)
            print('[[[[[[]]]]]]',pressure.shape,timm.shape)
            pressure_0,time_0,qq_0,pp_0,ff_0=np.array(pressure)[inter],time[inter],np.array(qq)[inter],np.array(pp)[inter],np.array(ff)[inter]

            pressure_f=pressure[inter]*0
            # co = lhs_g(name=g_name,pp=pp_0,qq=qq_0,qq2=qq_0**2,pp2=pp_0**2,ff=ff_0,vol=pm.vol,time=time_0)
            co = lhs_g(name=g_name,pp=pp_0,qq=qq_0,time=time_0)
            co=np.concatenate([np.vstack([val for val in vals]) for vals in co], axis=1)
            # co=np.column_stack(lhs)
            # co=np.concatenate([np.vstack([val for val in vals]) for vals in [time_0,qq_0,pp_0]], axis=1)
            # co=np.concatenate([np.vstack([val for val in vals]) for vals in [time_0,qq_0,pp_0]], axis=1)
            press=np.zeros((pm.N* (pm.Nperiod-1),pm.dimm*pm.nPart))
            for ii in range(Nmarkov):
                path_qq=os.path.join(path_1,f"qq_{ii}.txt")
                path_pp=os.path.join(path_1,f"pp_{ii}.txt")
                path_ff=os.path.join(path_1,f"ff_{ii}.txt")
                qqt=np.loadtxt(path_qq,dtype=float)
                ppt=np.loadtxt(path_pp,dtype=float)
                fft=np.loadtxt(path_ff,dtype=float) 
                # qq,pp,ff=[hh[Nlast:,:] for hh in [qqt,ppt,fft]]

                # pressure = rhs_g(name=g_name,pp=pp,qq=qq,ff=ff,vol=pm.vol)
                pressure = rhs_g(name=g_name,pp=ppt,qq=qqt,ff=fft,vol=pm.vol)
                # pressure = qq
                # pressure = np.sum(qq**2 , axis=1).reshape(-1,1)
                if ii>Nmarkov-10:
                    time_f.append(time[inter])
                    qq_f.append(qqt[inter])
                    pp_f.append(ppt[inter])
                    qq_f2.append(qqt[inter]**2)
                    pp_f2.append(ppt[inter]**2)
                    ff_f.append(fft[inter])
                    u_f.append(pressure[inter])
                    Xall['time'][key].append(timm)
                    Xall['q'][key].append(qqt)
                    Xall['p'][key].append(ppt)
                    Xall['force'][key].append(fft) 
                pressure_f+=pressure[inter] 
                press+=pressure
            # pressure_mean=np.sum(np.array([val.flatten() for val in pressure_f]).T,axis=1,keepdims=True)/len(pressure_f)
            # pressure_mean=pressure_f/Nmarkov
            press/=Nmarkov
            # lhs = lhs_g(name=g_name,pp=pp,qq=qq,ff=ff,vol=pm.vol,time=time_f)
            # cood = lhs_g(name=g_name,pp=pp_f,qq=qq_f,pp2=pp_f2,qq2=qq_f2,ff=ff_f,vol=pm.vol,time=time_f) 
            cood = lhs_g(name=g_name,pp=pp_f,qq=qq_f,time=time_f) 
            cood=np.concatenate([np.vstack([val for val in vals]) for vals in cood], axis=1)
            xPress=press.reshape((pm.Nperiod-1), pm.N, pm.dimm*pm.nPart).transpose(1, 0, 2).reshape(pm.N, (pm.Nperiod-1)*pm.dimm*pm.nPart)
            xPress=np.concatenate([pressure_0,xPress[:,ntimi:mtimi]], axis=1) 

            if tf_mean: 
                mio=pressure_0
                for ii in range(rhs_dim['i'],rhs_dim['j']):
                    ntimii,mtimii=1+pm.nPart*pm.dimm*ii-1,pm.nPart*pm.dimm*(ii+1) 
                    mio += xPress[:,ntimii:mtimii]
                xPress=mio/pm.Nperiod



            # xPress=xPress[:,ntimi:mtimi]
            # cood=np.concatenate([np.vstack([val for val in vals]) for vals in [time_f,qq_f,pp_f]], axis=1)
            vvf=np.vstack(u_f)
            if tf_train:
                coords_0[key].append(co)
                pressure_means[key].append(xPress)
                # pressure_means[key].append(pressure_mean)
                coords[key].append(cood)
                pressure_fs[key].append(pressure_0)


            else:
                coord_bound_0=co
                # u_bound=pressure_mean
                u_bound=xPress

                coord_bound_f=cood
                u_pressure_f=vvf

                bcs_train_test[index]={
                    'initial':{
                        'tf':tf_initial_bcs,
                        'coord':coord_bound_0,
                        'u':u_bound,
                    },
                    'final':{
                         'tf':False,
                        #'tf':tf_final_bcs,
                        'coord':coord_bound_0,
                        'u':pressure_0,
                    }
                }   

 


    if tf_train:
        for key,val in zip(keyss,Samples_nbr):
            if len(val)<1:
                continue
            coord_bound_0=np.vstack(coords_0[key])
            u_bound=np.vstack(pressure_means[key])

            coord_bound_f=np.vstack(coords[key])
            u_pressure_f=np.vstack(pressure_fs[key])


            bcs_train_test[key]={
                'initial':{
                    'tf':tf_initial_bcs,
                    'coord':coord_bound_0,
                    'u':u_bound,
                },
                'final':{
                     #'tf':tf_final_bcs,
                    'tf':False,
                    'coord':coord_bound_f,
                    'u':u_pressure_f,
                }
            }  






    sav=None
    '''
    if tf_train:
        with open(path_init, "rb") as f:
            sav_all,XX = pickle.load(f)

        dt_r=1 
        Tinit=dict(count=pm.N,
                range_0=1,
                range_N=pm.Nperiod)  
        Fun_integ={mm:nn for mm,nn in zip(['fine','coarse'],[Fun().SOILE_B,Fun().SOILE_A])}

        XXc=copy.deepcopy(XX)
        sav=copy.deepcopy(sav_all)
        sav,_ = Fun().Simulation_prof(pm,XXc, sav,Tinit=Tinit,dt_r=dt_r,Fun_integ=Fun_integ) 
'''


    for key,val in Xall.items():
        for ke,va in val.items():
            if len(va)>0:
                Xall[key][ke]=np.concatenate(Xall[key][ke],axis=0)




    return bcs_train_test,sav,par,pm,Nterm







import tensorflow as tf

# Derivative
class aka_grad():
    def __init__(self):
        pass

    def grad_q(self,model,fun_coef):  
        t = fun_coef.t
        q = fun_coef.q
        with tf.GradientTape(persistent=True) as tape:
                tape.watch(t)
                tape.watch(q)
                u = model(tf.concat([t,q],axis=1))
                u_q  =  tape.gradient(u,q)
        u_qq = tape.gradient(u_q,q)
        u_t = tape.gradient(u,t)

        return u,u_t,u_q,u_qq

    def grad_qp(self,model,fun_coef):  
        t = fun_coef.t
        q = fun_coef.q
        p = fun_coef.p
        f=fun_coef.force
        g_name=fun_coef.g_name

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(q)
            tape.watch(p)
            coord = lhs_g(name=g_name,pp=p,qq=q,qq2=q**2,pp2=p**2,ff=f,time=t)
            u = model(tf.concat(coord,axis=1))
            # u = model(tf.concat([t,q,p],axis=1))
            u_p  =  tape.gradient(u,p)
        u_pp = tape.gradient(u_p,p)
        u_q = tape.gradient(u,q)
        u_t = tape.gradient(u,t)
        del tape
        
        return u,u_t,u_q,u_p,u_pp
 
    def loss_fn(self,output, target):
        return tf.reduce_mean(tf.square(output - target))





 
class lang_loss(aka_grad):
    def __init__(self):
        aka_grad.__init__(self)
        pass
 
    def call(self,model,dc):  

        u,u_t,u_q,u_p,u_pp = self.grad_qp(model,dc) 
        '''
        if dc.flow == 'shear':
            uu_pp = u_pp[:,0:1] +  dc.t*dc.t*u_pp[:,1:2] 
            for i in range(1,dc.dim*dc.nPart):
                if i%dc.nPart==0:
                    uu_pp = tf.concat([uu_pp,u_pp[:,i:i+1]+ dc.t*dc.t*u_pp[:,i+1:i+2]],axis=1)
                else:
                    uu_pp = tf.concat([uu_pp,u_pp[:,i,i+1]],axis=1)
            uu_pp = (dc.gamma/dc.beta)*uu_pp  
        elif dc.flow == 'pef':
            uu_pp = (dc.gamma/dc.beta)*dc.expA*u_pp
        else:'''
        uu_pp = (dc.gamma/dc.beta)*u_pp
        # Gamma_p = tf.matmul(dc.p, tf.transpose(dc.Gamma))
        # print('------------------',dc.Gamma.shape,Gamma_p.shape)
        u_pp =  tf.reduce_sum(
            u_pp,
            axis=1,
            keepdims=True
        )
        # f_rhs = (-dc.p*u_q  -Gamma_p*u_p + uu_pp) 
        f_rhs = (-dc.p*u_q  -dc.Gamma*u_p + uu_pp)  
        '''
        # dyn_rhs =   tf.reshape(tf.reduce_sum(-dc.p*u_q - dc.force*u_p +dc.Gamma*u_p + uu_pp,axis=1),(-1,1))
        # l =   tf.reshape(tf.reduce_sum(u[1:,:]-u[:-1,:]-dc.dt*f_rhs[:-1,:],axis=1),(-1,1))
        loss = aka_grad().loss_fn(u[1:,:],u[:-1,:]+dc.dt*f_rhs[:-1,:]) 
        loss2 = aka_grad().loss_fn((1/u.shape[0])*tf.reduce_sum(u,axis=0),1) 
        loss = loss + loss2 
        
        '''
        loss = self.loss_fn(u_t,f_rhs)

        # loss = aka_grad().loss_fn(u_t,f_rhs)
        # loss = loss + aka_grad().loss_fn(model(dc.coord_bound),dc.u_bound)
        ii=0
        '''
        ntim,mtim=1+dc.N*ii-1,dc.N*(ii+1)-1
        timei=dc.t[ntim:mtim,:]
        coord_0 =  tf.concat([timei,dc.q[ntim:mtim,:],dc.p[ntim:mtim,:]], axis=1)

        for ii in range(7*dc.Nperiod//9,dc.Nperiod-1):
            ntim,mtim=1+dc.N*ii-1,dc.N*(ii+1)-1 
            coord_ii =  tf.concat([timei,dc.q[ntim:mtim,:],dc.p[ntim:mtim,:]], axis=1)
            loss+=  aka_grad().loss_fn(model(coord_0),model(coord_ii))
'''
        '''
        if dc.coord_bound_init is not None:
            loss+=  aka_grad().loss_fn(model(dc.coord_bound_init),dc.u_bound_init)
        if dc.coord_bound is not None:
            loss +=  aka_grad().loss_fn(model(dc.coord_bound),dc.u_bound)
'''

        for val in dc.bcs.values():
            if val['tf']:  
                loss+= self.loss_fn(model(val['coord']),val['u'])
                
                
                
                       # loss = loss_pde +loss_bound# + 0.1*loss_norm

 

        return  loss
 






import numpy as np
from neld_pinn_0.param import PBCs

class neld_coef(PBCs):
    def __init__(self,par,
                 sav=None,
                 bcs=None,
                 DTYPE='float32', 
                 Nterm =None,
                      ):
        PBCs.__init__(self,
                      flow=par['flow'], 
                       epsilon=par['epsilon'], 
                       nPart=par['nPart'], 
                       rcut=par['rcut'], 
                       N=par['N'], 
                       Nperiod=par['Nperiod'],
                       )
        if Nterm is None:
            Nterm=np.arange(sav.qq.shape[0])
            
        # if sav is not None:
        #     self.t = tf.cast(sav.time[Nterm,:],dtype=DTYPE)
        #     self.q = tf.cast(sav.qq[Nterm,:],dtype=DTYPE)
        #     self.p = tf.cast(sav.pp[Nterm,:],dtype=DTYPE)
        #     self.force = tf.cast(sav.fo[Nterm,:],dtype=DTYPE)
        if par is not None:
            self.t = tf.cast(par['Xall']['time']['train'],dtype=DTYPE)
            self.q = tf.cast(par['Xall']['q']['train'],dtype=DTYPE)
            self.p = tf.cast(par['Xall']['p']['train'],dtype=DTYPE)
            self.force = tf.cast(par['Xall']['force']['train'],dtype=DTYPE)


            # Gamma=np.zeros_like(sav.qq[Nterm,:])
            Gamma=np.zeros_like(par['Xall']['q']['train'])
            A=self.A+self.gamma
            for i,ii in enumerate(Nterm):
                Gamma[i,:]=( A@par['Xall']['q']['train'][ii,:].reshape(self.dimm,self.nPart)).ravel()
                # Gamma[i,:]=( A@sav.qq[ii,:].reshape(self.dimm,self.nPart)).ravel()

            self.Gamma =   tf.cast(Gamma,dtype=DTYPE) 
        for key,val in bcs.items():
            if val['tf']:
                for key2 in val:
                    if key2 !='tf':
                        bcs[key][key2]=tf.cast(bcs[key][key2],dtype=DTYPE)
        self.bcs=bcs
        self.gamma = tf.cast(self.gamma,dtype=DTYPE) 
        self.beta = tf.cast(self.beta,dtype=DTYPE) 
        self.g_name = par['g_name']
    



























