 
from time import time
import numpy as np
import copy
from beinn.neld_pinn_0.CompFun import Fun
from beinn.neld_pinn_0.param import PBCs,particles,saves 

'''
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

from help_fun import aka_grad
from Graph import mygraph
from help_fun import aka_fun,help_fun
from beinn.neld_pinn import lang_loss, neld_coef
from PINN import PINN, aka_train
'''


import pickle
import os
path_0=os.getcwd()


np.random.seed(0)
            # choose the type of the flow (i.e 'eld', 'shear',  or 'pef')
                # choose the type of the flow (i.e 'eld', 'shear',  or 'pef')

# np.random.seed(0)
               # choose the type of the flow (i.e 'eld', 'shear',  or 'pef')
                   # choose the type of the flow (i.e 'eld', 'shear',  or 'pef')


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

pm = PBCs(flow, epsilon, nPart, rcut, N, Nperiod,force=force,
          beta=beta,
          gamma=gamma)  # get the parameters
# pm.dt = 1.0/50
X = particles(pm)
X_init_0=Fun().initializez(pm,X)
dt_r=3
X_init={ii:X_init_0 for ii in range(dt_r)}
X={ii:particles(pm) for ii in range(dt_r)}
sav_part = saves(pm)

# sav_part={ii:saves(pm) for ii in range(dt_r)}
Fun_integ={mm:nn for mm,nn in zip(['fine','coarse'],[Fun().EM,Fun().EM])}
Fun_integ={mm:nn for mm,nn in zip(['fine','coarse'],[Fun().SOILE_B,Fun().SOILE_B])}
Fun_integ={mm:nn for mm,nn in zip(['fine','coarse'],[Fun().SOILE_A,Fun().SOILE_A])} 
Fun_integ={mm:nn for mm,nn in zip(['fine','coarse'],[Fun().SOILE_B,Fun().EM])}
Fun_integ={mm:nn for mm,nn in zip(['fine','coarse'],[Fun().SOILE_B,Fun().SOILE_A])}

path_gen=os.path.join(path_0,f'data_{flow}_{nPart}_{force}',)
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

print(f"Parameters saved to {path}")





path_init_param=os.path.join(path_gen,f"param.pkl")
with open(path_init_param, "wb") as f:
        pickle.dump(pm, f) 







fmt='%.4f'
Nmarkov=1000
Nsample=50
for index in range(Nsample):
    path_1=os.path.join(path_gen,f'data_{index}')
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
        dt_r=3
        X_init={ii:X_init_0 for ii in range(dt_r)}
        X={ii:particles(pm) for ii in range(dt_r)}
        sav_part = saves(pm)

        XX=Fun().Simulation_init(pm, X_init, dt_r=dt_r,Fun_integ=Fun_integ)
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




    dt_r=1 
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
        np.savetxt(path_qq,sav.qq[Nlast:,:],fmt=fmt)
        np.savetxt(path_pp,sav.pp[Nlast:,:],fmt=fmt)
        np.savetxt(path_ff,sav.fDist[Nlast:,:],fmt=fmt)
        
        # path=os.path.join(path_1,f"sample_{ii}.pkl")
        # with open(path, "wb") as f:
        #     pickle.dump(sav, f)










