


import numpy as np
 

from param import PBCs,particles,saves,paramFig
from Graph import mygraph
from CompFun import Fun as myFun
from help_fun import aka_fun


tcouleur = 'plotly_dark'
bcouleur = 'navy'
fcouleur = 'white'
fsize = 20
   

np.random.seed(0)

 
mygraph = mygraph(tcouleur=tcouleur,
                  bcouleur=bcouleur,
                  fcouleur=fcouleur,
                  fsize=fsize)

 




flow = 'pef'                     # choose the type of the flow (i.e 'eld', 'shear',  or 'pef')
nPart = 3                         # Number of particles
epsilon = 1.0                     # rate of the deformation of the background flow
rcut = 30                         # radius cut
N = 50                          # number of steps in a period
Nperiod = 500                   # number of periods



pm = PBCs(flow, epsilon, nPart, rcut, N, Nperiod)  # get the parameters 
X = particles(pm.dim,pm.nPart)
sav_part = saves(pm)
sav = myFun().Simulation(pm,X, sav_part)
# datF = paramFig(flow=pm.flow,sbox=pm.a)

bheight = 600
bwidth = 600



mygraph.plot_history_matrixxy(sav.Q1,sav.Q2,sav.time,bheight,bwidth).show()


Ntime= pm.N
Ndim = pm.dim
Nperiod = pm.Nperiod
qq = sav.qq
xint = 0
yint = xint+pm.nPart
qq_x = aka_fun().vec_to_mat_(qq,xint,Ntime,Nperiod)
qq_y = aka_fun().vec_to_mat_(qq,yint,Ntime,Nperiod) 

fig = mygraph.plot_history_matrixxy(qq_x,qq_y,sav.time,bheight,bwidth)
fig.show()

