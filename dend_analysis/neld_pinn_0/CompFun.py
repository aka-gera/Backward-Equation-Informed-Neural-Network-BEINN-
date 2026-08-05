import numpy as np
import time 
from neld_pinn_0.help_fun import help_fun
import copy



from neld_pinn_0.integrator import Integrator
class Fun(Integrator):
    def __init__(self):
        Integrator.__init__(self)

    def initializez(self,pm,X):
        nPart = pm.nPart
        dim = pm.dimm
        beta = pm.beta
        A = pm.A
        L0 = pm.L0
 

        if dim == 3:
            ll = 0
            for l in range(nPart):
                i = l+1
                j = i+1
                X.q[:,ll] = [(0.5 + i-0.5*nPart)/nPart,
                            (0.5 + j-0.5*nPart)/nPart,
                            (0.5 + l-0.5*nPart)/nPart]
                ll += 1
        else:
            ll = 0
            for l in range(nPart):
                j = l+1 
                X.q[:,ll] = [(0.5 + l-0.5*nPart)/nPart,
                            (0.5 + j-0.5*nPart)/nPart 
                            ]
                ll += 1
        
        #X.q = np.matmul(pm.A, X.q)
        X.q = np.matmul(pm.L0, X.q)
        X.q +=   np.random.randn(dim, nPart)
        # X.q +=  0.05 * np.random.randn(dim, nPart)
        X.p = np.matmul(A, X.q)
        X.p  += np.sqrt(1/beta) * np.random.randn(dim, nPart)
        return X

 
    def ComputeForceEulerian(self,pm,X,theta):  
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
                    X.qDist, _,_ = help_fun().remap(X.qDist, theta,pm)
                    normqD = np.linalg.norm(X.qDist)
                    ff = help_fun().fLJ(normqD,pm)
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
            X.q, _,_ = help_fun().remap(X.q, theta,pm)
            X.qDist = X.q 
            pF= help_fun().poten(X.q)
            X.f[0,0],X.f[1,0] = pF[0],pF[1] 
            # print('[[[[[[[[[99999999]]]]]]]]]',pF)
            # jk 
            X.ff = np.linalg.norm(X.f)
            X.virial+= np.dot(X.qDist[:,0],pF)
            # X.virial += -0.5 * normqD * ff
            X.fDist+= X.qDist*pF
            return X
        

    def EmEulerian(self,pm,X, theta):
        nPart = pm.nPart
        dim = pm.dim
        dt = pm.dt
        gamma = pm.gamma
        beta = pm.beta
        A = pm.A
 
        # Update position
        X.q += (X.p + np.matmul(A , X.q)) * dt

        # Compute force
        X = self.ComputeForceEulerian(pm,X, theta) 

        # Update momentum
        X.G  = np.sqrt(2 *dt * gamma / beta) * np.random.randn(dim, nPart) 
        X.p +=  X.f * dt - gamma * X.p * dt + X.G

        # Remap position
        X.q, L,Linv = help_fun().remap(X.q, theta,pm)

        return X, L,Linv
 





    def Simulation(self,pm,X,sav,X_init=None):    
        print('Pre-simulation begin \n')

        tic = time.time()

        theta = 1.0 
        fmax = 1e-16
        if X_init is None:
            X = self.initializez(pm,X) 
        else:
            X=X_init
        for i in range(pm.N):
            thetaTmp = theta + pm.Sigma * pm.dt
            theta = thetaTmp - np.floor(thetaTmp)
            pm.n = pm.n + theta - thetaTmp
            pm.theta = theta       

            # X,_,_ = self.EmEulerian(pm,X,theta)
            X,_,_ = self.SOILE_B(pm,X,theta)
            # print(f'theta {theta:.4f}     n = {pm.n} ')

            if np.abs(fmax) < np.abs(X.ff):
                    fmax = X.ff
            if np.mod(i, np.round(pm.N / 10)) == 0:
                toc = time.time()-tic
                min = np.floor(toc/60).astype(int)
                sec = np.floor(toc-60*min).astype(int)
                print(f"Time {theta:4f} executed in {min:2d} m {sec:3d} s with max f = {fmax:.3f}")
         
        print(f'force : {fmax:2f} \n\n')
        print('Main simulation begin')
 
        for j in range(pm.Nperiod):
            for i in range(pm.N):   
                jj = j*pm.N+i 
                inv_expA = help_fun().MyExp(-theta*pm.Y)
                ptmp =  np.matmul(inv_expA,X.p)
                sav.qq[jj,:] = np.matmul(inv_expA,X.q).flatten().tolist()
                sav.pp[jj,:] = ptmp.flatten().tolist()
                sav.fo[jj,:] = np.matmul(inv_expA,X.f).flatten().tolist()
                sav.Gamma[jj,:] = (np.matmul(pm.A,ptmp)+pm.gamma*ptmp).flatten().tolist() 
                if pm.flow == 'pef':  
                    sav.expA[jj,:] = np.array([[np.exp(-2.0 * theta * pm.Y[itmp]) for itmp in range(pm.dim+1)] for _ in range(pm.nPart)]).flatten().tolist()
                    
                sav.Q1[i, j] = X.q[0,0]  
                sav.Q2[i, j] = X.q[1,0]  
                sav.F[i, j] = X.ff 
                
                sav.time[jj,0] = theta
                sav.virial[jj,0] = X.virial
                sav.fDist[jj,:] = X.fDist.flatten().tolist()

                thetaTmp = theta + pm.Sigma * pm.dt
                theta = thetaTmp - np.floor(thetaTmp)
                pm.n = pm.n + theta - thetaTmp
                pm.theta = theta

                X,_,_ = self.EmEulerian(pm,X, theta)

                
                
                if np.abs(fmax) < np.abs(X.ff):
                    fmax = X.ff
            if np.mod(j, np.round(pm.Nperiod / 10)) == 0:
                toc = time.time()-tic
                min =  np.floor(toc/60).astype(int)
                sec = np.floor(toc-60*min).astype(int)
                print(f"Period {j:4d} executed in {min:2d} m {sec:3d} s with max f = {fmax:.3f}")
         
        print('force : ',fmax)
        return sav



 













    def Simulation_init(self, pm, XX,sav, dt_r=1,Fun_integ=None):

        print("Pre-simulation begin\n")

        tic = time.time()

        dt = pm.dt 
        fmax = 1e-16 

        dim = pm.dim
        nPart = pm.nPart

        # fluct  = np.sqrt(2.0 * pm.gamma * dt / pm.beta)*0.0
        # fluct2 = np.sqrt(2.0 * pm.gamma * dt**3 / pm.beta)*0.0
 
        jj,j=0,0
        X = XX[0] 
        for i in range(pm.N):

            # Brownian increments for finest trajectory
            xx = np.random.randn(dim, nPart)
            yy = np.random.randn(dim, nPart)

            X.RF  = xx * pm.fluct
            X.RF2 = 0.5 * pm.fluct2 * (xx + yy / np.sqrt(3.0))

            X.RF3 = pm.fluct * xx
            X.RF4 = 0.5 * pm.fluct2 * (xx + yy / np.sqrt(3.0)) 

            X = Fun_integ['fine'](pm, X,dt)
            self.remap_coord(X,pm)  
            if abs(X.ff) > abs(fmax):
                fmax = X.ff



            jj=i

            # jj = j * (pm.N // step) + save_count[ii] 
            # inv_expA = help_fun().MyExp(-pm.theta * pm.Y)
            # ptmp = inv_expA @ X.prel

            # sav.qq[jj, :] = (inv_expA @ X.q).ravel()
            # sav.pp[jj, :] = ptmp.ravel()
            # sav.fo[jj, :] = (inv_expA @ X.f).ravel()

            ptmp = X.p 
            sav.qq[jj, :] = X.q.ravel()
            sav.pp[jj, :] = ptmp.ravel()
            sav.fo[jj, :] = X.f.ravel()
            sav.fDist[jj,:] = X.fDist.ravel()

            sav.time[jj,0] = pm.theta





        self.remap_box(pm,dt=dt) 

            # self.remap_box(pm,dt=dt) 

 
        XX[0] = X



        for r in range(1, dt_r):
            XX[r] = copy.deepcopy(XX[0])
        return sav,XX


    def Simulation_prof(self, pm, XX, sav, dt_r=1, Tinit=0,Fun_integ=None,disp_info=False,):
        
        
        tic = time.time()

        dt = pm.dt 
        fmax = 1e-16

        dim = pm.dimm
        nPart = pm.nPart
        if disp_info:
            print("Main simulation begin")
 
        # --------------------------------------------------
        # Main simulation
        # --------------------------------------------------
 
        save_count = [Tinit['count'] for _ in range(dt_r)]
        save_count = [0 for _ in range(dt_r)]
        ki=0
        pdim=(pm.dimm,pm.nPart)
        for j in range(Tinit['range_0'],Tinit['range_N']):

            # reset accumulated noises
            for r in range(dt_r):
                # XX[r].RF3.fill(0.0)
                # XX[r].RF4.fill(0.0)
                # XX[r].RF.fill(0.0)
                # XX[r].RF2.fill(0.0)

                XX[r].RF3=np.zeros(pdim)
                XX[r].RF4 = np.zeros(pdim)
                XX[r].RF = np.zeros(pdim)
                XX[r].RF2 = np.zeros(pdim)
            for i in range(pm.N):
                iii = j*pm.N+i  

                xx = np.random.randn(dim, nPart)
                yy = np.random.randn(dim, nPart)


                RF3 = pm.fluct * xx
                RF4 = 0.5 * pm.fluct2 * (xx + yy / np.sqrt(3.0))

                XX[0].RF  = xx * pm.fluct
                XX[0].RF2 = 0.5 * pm.fluct2 * (xx + yy / np.sqrt(3.0))
                XX[0].RF3 = RF3
                XX[0].RF4 = RF4
 

                for r in range(1, dt_r):
                    # print('[[[[[]]]]]',XX[r].RF4.shape,XX[r].RF3.shape,RF4.shape,dim)
                    XX[r].RF4 += RF4 + XX[r].RF3 * dt
                    XX[r].RF3 += RF3
                    XX[r].RF2 += RF4 + XX[r].RF * dt
                    XX[r].RF += RF3
 
                XX[0] = Fun_integ['fine'](pm, XX[0],dt)
                self.remap_coord(XX[0],pm)  

                for r in range(1, dt_r):
                    step = 2 ** r
                    if (iii + 1) % step == 0: 
                        self.remap_coord(XX[r],pm) 
                        XX[r] = Fun_integ['coarse'](pm,XX[r],dt*step ) 
                        XX[r].RF3.fill(0.0)
                        XX[r].RF4.fill(0.0)
                        XX[r].RF.fill(0.0)
                        XX[r].RF2.fill(0.0)
                if (iii + 1) % 2**dt_r == 0:
                    sav.error_pp.append([np.linalg.norm(XX[kk].prel - XX[0].prel) for kk in range(1,dt_r)])
                    sav.error_qq.append([help_fun().BCnorm(XX[kk].q - XX[0].q,L=pm.L,Linv=pm.Linv) for kk in range(1,dt_r)]) 
                dq = XX[dt_r-1].q - XX[0].q 
                                
                ####################################################
                # Save trajectories
                ####################################################

                for ii in range(1):
                    step = 2**ii 
                    X = XX[ii]

                    # jj = j * (pm.N // step) + save_count[ii]
                    jj = save_count[ii]
                    # inv_expA = help_fun().MyExp(-pm.theta * pm.Y)
                    # ptmp = inv_expA @ X.prel

                    # sav.qq[jj, :] = (inv_expA @ X.q).ravel()
                    # sav.pp[jj, :] = ptmp.ravel()
                    # sav.fo[jj, :] = (inv_expA @ X.f).ravel()
 
                    ptmp = X.p 
                    sav.qq[jj, :] = X.q.ravel()
                    sav.pp[jj, :] = ptmp.ravel()
                    sav.fo[jj, :] = X.f.ravel()
                    sav.fDist[jj,:] = X.fDist.ravel()

                    sav.time[jj,0] = pm.theta
                    '''
                    sav.virial[jj,0] = X.virial
                    sav.Gamma[jj, :] = (
                        pm.A @ ptmp + pm.gamma * ptmp
                    ).ravel()

                    sav.Q1[i, j] = X.q[0,0]  
                    sav.Q2[i, j] = X.q[1,0]  
                    sav.F[i, j] = X.ff 
                    '''
                    save_count[ii] += 1

                    if abs(X.ff) > abs(fmax):
                        fmax = X.ff
                
                self.remap_box(pm,dt=dt) 
            
            if j % max(1, pm.Nperiod // 10) == 0:

                toc = time.time() - tic
                if disp_info:
                    print(
                        f"inc {iii:5d} "
                        f"Period {j:5d} "
                        f"elapsed {toc/60:.1f} min "
                        f"max force = {fmax:.4e} "
                        f"BC error: { help_fun().BCnorm(dq, pm.L, pm.Linv):.3e} "
                        f"Raw error: { np.linalg.norm(dq):.3e} "
                    )
                 

        ii=0

        # sav.error_pp=np.array(sav.error_pp)
        # sav.error_qq=np.array(sav.error_qq)
        if disp_info:
            print("Maximum force:", fmax) 
        return sav,XX










































