 
import numpy as np  
from scipy.linalg import expm
  


def get_sample_indices(nrow,ncol,nSample=10):
    # nSample=10 
    one=np.zeros((nrow,ncol),dtype=int)
    inter=np.arange(0,nrow,nSample) 
    for ii in inter:
        one[ii,:]=1
    return np.where(one.ravel()==1)[0],inter.shape[0]




class my_force:
    def __init__(self):
        pass
        self.force={
            'poten':{
                'force_single':self.poten
            },
            'zero':{
                'force_single':self.f_zeros
            },
            'fLJ':{
                'force_pair':self.fLJ
            },
            'harmonic':{
                'force_pair':self.fHarmonic,
                'force_single':self.f_harmonic
            },
        }

    def poten(self,pm,q):
        x,y = q[0],q[1]
        return [-4*np.sin(2*x)-np.sin(x-y),-np.sin(y)+ np.sin(x-y) ]



    def fLJ(self,pm,rr): 

        if rr > pm.rcut:
            return 0.0
        else:
            return 4 * pm.eps * ((12 * pm.sigm ** 6) / rr ** 7 - (12 * pm.sigm  ** 12) / rr ** 13) 


    def f_zeros(self,pm,q=None):
        return np.zeros((pm.dimm,1))


    def f_harmonic(self,pm,q=None):
        return -np.ones_like(q)
    
    def get_force(self,name,type): 
        return self.force[name][type]


    def fHarmonic(self, pm, rr):

        if rr > pm.rcut:
            return 0.0
        else:
            return -pm.k * (rr - pm.r0)





class help_fun(my_force):
    def __init__(self):
        pass    
        my_force.__init__(self)

    def MyExp(self,M):
        if len(M.shape) > 1:  
            return expm(M)
        else:
            return np.diag(np.exp(M)) 


    def MyRound(self,x):
        return x - np.round(x) 


    def remap_q(self, q, L,Linv): 
        return L @ self.MyRound( Linv @ q  ) 
     
    def ComputeForce(self,pm,X):  
        X.f = np.zeros((pm.dim+1, pm.nPart))  
        X.fDist = np.zeros((pm.dim+1, pm.nPart)) 
        X.ff = 0.0
        X.virial =0.0

        X.pressure=0
        if pm.nPart >1:
            mm1 = 1
            mm2 = 1
            for i in range(pm.nPart-1):
                for j in range(i+1, pm.nPart):
                    X.qDist = X.q[:, i] - X.q[:, j]
                    X.qDist= self.remap_q(X.qDist, pm.L,pm.Linv)
                    normqD = np.linalg.norm(X.qDist)
                    # ff = self.fLJ(normqD,pm)
                    if normqD<1e-12:
                        continue
                    ff= self.get_force(pm.force,'force_pair')(pm,normqD)
                    fij= ff * X.qDist / normqD
                    X.f[:, i] -= fij 
                    X.f[:, j] += fij 
                    X.fDist[:,i]-= X.qDist*fij
                    X.fDist[:,j]+= X.qDist*fij
                    # X.virial+= -0.5*np.dot(X.qDist ,fij)
                    X.virial += -0.5 * normqD * ff
                    if mm1 < abs(ff):
                        mm1 = abs(ff)
                        mm2 = ff
 
            X.ff = mm2
            return X
        else:
            # X.q= self.remap_q(X.q, pm.L,pm.Linv)
            X.qDist = X.q  
            X.qDist= self.remap_q(X.qDist, pm.L,pm.Linv)
            pF= self.get_force(pm.force,'force_single')(pm,X.q)
            X.f[0,0],X.f[1,0] = pF[0],pF[1] 
            # print('[[[[[[[[[99999999]]]]]]]]]',pF)
            # jk 
            X.ff = np.linalg.norm(X.f)
            X.virial+= np.dot(X.qDist[:,0],pF) 
            X.fDist+= X.qDist*pF


            '''
            i=0
            X.qDist = X.q[:, i]  
            X.qDist= help_fun().remap_q(X.qDist, pm.L,pm.Linv)
            normqD = np.linalg.norm(X.qDist)
            # ff = self.fLJ(normqD,pm)
            X.ff=ff= self.get_force(pm.force)(pm,normqD)
            fij= ff * X.qDist / normqD
            X.f[:, i] -= fij  
            X.fDist[:,i]-= X.qDist*fij

            X.virial += -0.5 * normqD * ff
'''


            return X
        



    def remap_box(self,pm,dt):    
        thetaTmp = pm.theta + pm.Sigma * pm.dt
        pm.theta = thetaTmp - np.floor(thetaTmp)
        pm.n += pm.theta - thetaTmp  
        pm.L = self.MyExp(pm.Y * pm.theta) @ pm.L0
        pm.Linv = pm.L0inv @ self.MyExp(-pm.Y * pm.theta) 

    def remap_coord(self,X,pm):    
        X.q= pm.L @ self.MyRound( pm.Linv @ X.q  )
        X.p= X.prel + pm.A @ X.q



    def BCnorm(self,q,L,Linv): 
        return np.linalg.norm(self.remap_q(q, L,Linv))
    '''
    def remap_coord(self,X,pm):    
        X.q= X.L @ self.MyRound( X.Linv @ X.q  )
        X.p= X.prel + pm.A @ X.q



    def BCnorm(self,q,L,Linv): 
        return np.linalg.norm(self.remap_q(q, L,Linv))
    '''



import tensorflow as tf

class aka_fun():
    def __init__(self,DTYPE='float32'):
        self.DTYPE = DTYPE


    # def delta_function(self,x):
    #     return np.where(x == 0, np.inf, 0)

    def delta_function(self,x):
        uu = 10**10
        return (uu/np.pi)*np.exp(-uu**2*tf.reduce_sum(x*x,axis=1))

    def fun_u_0(self,q):    
        return   tf.reshape(self.delta_function(q-q) ,(-1,1)) 

    # def fun_u_0(self,q):    -tf.reduce_sum( q-q,axis=1)
    #     return   tf.reshape(np.exp(-tf.reduce_sum((1/2)*q*q/10**10,axis=1))/(np.sqrt(np.pi)*10**10),(-1,1)) 

    # Define boundary condition
    def fun_u_b(self,sav): 
        return tf.zeros_like(sav.time)

    
    def potential(self,coord): 
        x, y = coord[:,0:1], coord[:,1:2]
        return  (1/6)*(4*(1-x**2-y**2)**2+2*(x**2-2)**2+((x+y)**2-1)**2+((x-y)**2-1)**2)
    
    def force(self,coord):
        x, y = coord[:,0:1], coord[:,1:2]
        fx =(1/6)*(8*x*(x**2 - 2) - 16*x*(-x**2 - y**2 + 1) + (4*x - 4*y)*((x - y)**2 - 1) + (4*x + 4*y)*((x + y)**2 - 1))
        fy = (1/6)*(-16*y*(-x**2 - y**2 + 1) + (-4*x + 4*y)*((x - y)**2 - 1) + (4*x + 4*y)*((x + y)**2 - 1))
        return tf.concat([fx,fy], axis= 1) 


    def vec_to_mat_(self,qq,xint,Ntime,Nperiod):
        qq_x = qq[0:Ntime,xint:xint+1]
        for i in range(1,Nperiod):
            qq_x = np.hstack([qq_x,qq[i*Ntime:(i+1)*Ntime,xint:xint+1]])
        return qq_x

    def box(self,theta,pm): 
        L = np.dot(help_fun().MyExp(pm.Y*theta) , pm.L0)
        Linv = np.dot( pm.L0inv, help_fun().MyExp(-pm.Y*theta))
        return L,Linv

    def remap_fix_init_dom(self,q,theta,pm): 
        L = np.matmul(help_fun().MyExp(pm.Y*theta) , pm.L0)
        Linv = np.matmul( pm.L0inv, help_fun().MyExp(-pm.Y*theta)) 
        q = np.matmul(L, help_fun().MyRound(np.matmul(pm.L0inv, q)))
        return q,L,Linv

    def remap_fix_q(self,q,theta,pm): 
        # qtmps = tf.concat([q,np.zeros_like(q[:,0:1])],axis=1)
        q_00,_,_ = self.remap_fix_init_dom(tf.transpose(q),theta,pm) 
        q_00 = tf.cast(q_00,dtype=self.DTYPE)
        q_00 = tf.matmul(tf.cast(help_fun().MyExp(-theta*pm.Y),dtype=self.DTYPE),q_00)
        return  q_00 

    def remap_fix_qq(self,q,theta,pm): 
        # qtmps = tf.concat([q,np.zeros_like(q[:,0:1])],axis=1)
        q_00,_,_ = self.remap_fix_init_dom(q,theta,pm) 
        q_00 = tf.cast(q_00,dtype=self.DTYPE)
        q_00 = tf.matmul(tf.cast(help_fun().MyExp(-theta*pm.Y),dtype=self.DTYPE),q_00)
        return  q_00 
    
    def remap_fix_q_s(self,q,thetas,pm): 
        q_return = tf.zeros_like(q)
        for i,theta_ in enumerate(thetas):
            theta = theta_.numpy()
            for j in range(pm.nPart):
                qtmps = np.array([[q[i,j]],[q[i,pm.nPart+j]]])
                q_00,_,_ = self.remap_fix_init_dom(qtmps,theta,pm) 
                q_00 = tf.matmul(tf.cast(help_fun().MyExp(-theta*pm.Y),dtype=self.DTYPE),q_00)

                q_return = tf.tensor_scatter_nd_update(q_return, [[i, j]], [q_00[0,0]])
                q_return = tf.tensor_scatter_nd_update(q_return, [[i, pm.nPart+j]], [q_00[1,0]])

        return q_return
'''
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

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(q)
            tape.watch(p)
            u = model(tf.concat([t,q,p],axis=1))
            u_p  =  tape.gradient(u,p)
        u_pp = tape.gradient(u_p,p)
        u_q = tape.gradient(u,q)
        u_t = tape.gradient(u,t)
        del tape
        
        return u,u_t,u_q,u_p,u_pp
 
    def loss_fn(self,output, target):
        return tf.reduce_mean(tf.square(output - target))
'''