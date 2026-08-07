import numpy as np
import time 
from help_fun import help_fun
import copy
  
class Fun:
    def __init__(self):  
        pass
  

    def initializez(self,pm,X):
        nPart = pm.nPart
        dim = pm.dim
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

        X.q = np.matmul(pm.L0, X.q)
        X.q +=  0.05 * np.random.randn(dim, nPart)
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







    '''

    def Simulation_prof(self,pm,XX,sav,dt_r=1,X_init=None):    
        print('Pre-simulation begin \n')

        tic = time.time()
        dt=dt_r*pm.dt
        theta = 1.0 
        fmax = 1e-16
        if X_init is None:
            XX ={ii: self.initializez(pm,XX) for ii in range(dt_r)}
        else:
            XX=X_init


        dim,nPart=pm.dim,pm.nPart
        ii=0
        xx = np.random.randn(dim+1, nPart)
        yy = np.random.randn(dim+1, nPart)
        fluct  = np.sqrt(2 * pm.gamma * dt / pm.beta)
        fluct2 = np.sqrt(2 * pm.gamma * dt**3 / pm.beta)
        XX[ii].RF  = xx * fluct
        XX[ii].RF2 = 0.5 * fluct2 * (xx + yy / np.sqrt(3.0))
        XX[ii].RF3 = fluct * xx
        XX[ii].RF4 = 0.5 * fluct2 * (xx + yy / np.sqrt(3.0))
        for ii in range(1,dt_r-1):
            XX[ii].RF4 += XX[0].RF4+XX[ii].RF3*dt
            XX[ii].RF3 += XX[0].FR3





        for ii in range(dt_r):
            for i in range(pm.N):
                thetaTmp = theta + pm.Sigma * dt
                theta = thetaTmp - np.floor(thetaTmp)
                pm.n = pm.n + theta - thetaTmp
                pm.theta = theta       

                # X,_,_ = self.EmEulerian(pm,X,theta)
                X=XX[ii]
                X,_,_ = self.SOILE_B(pm,X,theta,dt=dt)
                XX[ii]=X
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
                    sav[ii].qq[jj,:] = np.matmul(inv_expA,X.q).flatten().tolist()
                    sav[r].pp[jj,:] = ptmp.flatten().tolist()
                    sav[r].fo[jj,:] = np.matmul(inv_expA,X.f).flatten().tolist()
                    sav[r].Gamma[jj,:] = (np.matmul(pm.A,ptmp)+pm.gamma*ptmp).flatten().tolist() 
                    if pm.flow == 'pef':  
                        sav[r].expA[jj,:] = np.array([[np.exp(-2.0 * theta * pm.Y[itmp]) for itmp in range(pm.dim+1)] for _ in range(pm.nPart)]).flatten().tolist()
                        
                    sav[r].Q1[i, j] = X.q[0,0]  
                    sav[r].Q2[i, j] = X.q[1,0]  
                    sav[r].F[i, j] = X.ff 
                    
                    sav[r].time[jj,0] = theta
                    sav[r].virial[jj,0] = X.virial
                    sav[r].fDist[jj,:] = X.fDist.flatten().tolist()

                    thetaTmp = theta + pm.Sigma * dt
                    theta = thetaTmp - np.floor(thetaTmp)
                    pm.n = pm.n + theta - thetaTmp
                    pm.theta = theta

                    # X,_,_ = self.EmEulerian(pm,X, theta)
                    X,_,_ = self.SOILE_B(pm,X, theta,dt=dt)

                    
                    
                    if np.abs(fmax) < np.abs(X.ff):
                        fmax = X.ff
                if np.mod(j, np.round(pm.Nperiod / 10)) == 0:
                    toc = time.time()-tic
                    min =  np.floor(toc/60).astype(int)
                    sec = np.floor(toc-60*min).astype(int)
                    print(f"Period {j:4d} executed in {min:2d} m {sec:3d} s with max f = {fmax:.3f}")
            
            print('force : ',fmax)
        return sav



'''




    def EM(self, pm, X): 

        # dt = pm.dt
        xi = pm.gamma          # friction coefficient (check this)
        A = pm.A
        theta =X.theta
        dt=X.dt


        nPart = pm.nPart
        dim = pm.dim 
        gamma = pm.gamma
        beta = pm.beta
        A = pm.A
 
        # Update position
        X.q += (X.p + np.matmul(A , X.q)) * dt

        # Compute force
        # X = self.ComputeForceEulerian(pm,X, theta) 

        # Update momentum  X.f * dt 
        # X.G  = np.sqrt(2 *dt * gamma / beta) * np.random.randn(dim, nPart) 
        X.p +=- gamma * X.p * dt #+ X.RF

        # Remap position
        X.q, L,Linv = help_fun().remap(X.q, theta,pm)

        return X, L,Linv
 





    def SOILE_B(self, pm, X): 

        # dt = pm.dt
        xi = pm.gamma          # friction coefficient (check this)
        A = pm.A
        theta =X.theta
        dt=X.dt
        # Gamma = xi*I - A
        Gamma = xi * np.eye(pm.dim) - A
 

        rhs = X.f + xi * (A @ X.q) - Gamma @ X.p

        X.p += (
            0.5 * dt * rhs
            - 0.25 * (Gamma @ (0.5 * dt**2 * rhs
                            + 2.0 * (X.RF4 - 0.25 * dt * X.RF3)))
            + 0.5 * X.RF3
        )

        # ----- Position update -----

        X.q += dt * X.p + X.RF4 - 0.5 * dt * X.RF3

        # Remap if required
        X.q, L, Linv = help_fun().remap(X.q, theta, pm)

        # Relative momentum
        X.prel = X.p - A @ X.q

        # ----- Compute new force -----

        # X = self.ComputeForceEulerian(pm, X, theta)

        # ----- Second half-step momentum -----

        rhs = X.f + xi * (A @ X.q) - Gamma @ X.p

        X.p += (
            0.5 * dt * rhs
            - 0.25 * (Gamma @ (0.5 * dt**2 * rhs
                            + 2.0 * (X.RF4 - 0.25 * dt * X.RF3)))
            + 0.5 * X.RF3
        )

        X.prel = X.p - A @ X.q

        return X, L, Linv












    def Simulation_prof(self, pm, XX, sav, dt_r=1, X_init=None,Fun_integ=None):

        print("Pre-simulation begin\n")

        tic = time.time()

        dt = pm.dt
        theta = [1.0 for _ in range(dt_r)]
        nn = [0.0 for _ in range(dt_r)]
        theta=1
        fmax = 1e-16

        if X_init is None:
            XX = {k: self.initializez(pm, XX) for k in range(dt_r)}
        else:
            XX = X_init

        dim = pm.dim
        nPart = pm.nPart

        fluct  = np.sqrt(2.0 * pm.gamma * dt / pm.beta)*0.0
        fluct2 = np.sqrt(2.0 * pm.gamma * dt**3 / pm.beta)*0.0

        # --------------------------------------------------
        # Thermalization
        # --------------------------------------------------

        # for r in range(dt_r):
        r=0
        X = XX[r]
        step = 2 ** r
        dtt=step*dt

        for _ in range(pm.N):

            # Brownian increments for finest trajectory
            xx = np.random.randn(dim, nPart)
            yy = np.random.randn(dim, nPart)

            X.RF  = xx * fluct
            X.RF2 = 0.5 * fluct2 * (xx + yy / np.sqrt(3.0))

            X.RF3 = fluct * xx
            X.RF4 = 0.5 * fluct2 * (xx + yy / np.sqrt(3.0))

            thetaTmp = theta[r] + pm.Sigma * dtt
            theta[r] = thetaTmp - np.floor(thetaTmp)

            nn[r] += theta[r] - thetaTmp
            XX[r].n=pm.n=nn[r]
            
            XX[r].theta=pm.theta = theta[r]
            XX[r].dt=dt

            X, _, _ = Fun_integ['fine'](pm, X)
            # X, _, _ = self.SOILE_B(pm, X)

            if abs(X.ff) > abs(fmax):
                fmax = X.ff

        XX[r] = X



        for r in range(1, dt_r):
            XX[r] = copy.deepcopy(XX[0])


        print("Main simulation begin")

        err_file = open("strong_error.txt", "w")
        err_file.write("# step e_h_2h e_2h_4h\n")
        # --------------------------------------------------
        # Main simulation
        # --------------------------------------------------

                ####################################################
                # Save finest trajectory
                ####################################################
        save_count = [0 for _ in range(dt_r)]
        ki=0
        for j in range(pm.Nperiod):

            # reset accumulated noises
            for r in range(dt_r):
                XX[r].RF3.fill(0.0)
                XX[r].RF4.fill(0.0)
                XX[r].RF.fill(0.0)
                XX[r].RF2.fill(0.0)

            for i in range(pm.N):
                iii = j*pm.N+i 

                ####################################################
                # Fine Brownian increment
                ####################################################

                xx = np.random.randn(dim, nPart)
                yy = np.random.randn(dim, nPart)


                RF3 = fluct * xx
                RF4 = 0.5 * fluct2 * (xx + yy / np.sqrt(3.0))

                XX[0].RF  = xx * fluct
                XX[0].RF2 = 0.5 * fluct2 * (xx + yy / np.sqrt(3.0))
                XX[0].RF3 = RF3
                XX[0].RF4 = RF4

                ####################################################
                # Accumulate coarse Brownian increments
                ####################################################

                for r in range(1, dt_r):

                    XX[r].RF4 += RF4 + XX[r].RF3 * dt
                    XX[r].RF3 += RF3

                    XX[r].RF2 += RF4 + XX[r].RF * dt
                    XX[r].RF += RF3

                ####################################################
                # Finest trajectory
                ####################################################
                ii=0
                thetaTmp = theta[ii] + pm.Sigma * dt
                theta[ii] = thetaTmp - np.floor(thetaTmp)

                nn[ii] += theta[ii] - thetaTmp
                XX[ii].n=pm.n=nn[ii] 

                XX[ii].theta=pm.theta = theta[ii]
                XX[ii].dt=dt
                
                # if integr=='SOILE_B':
                XX[ii], _, _ = Fun_integ['fine'](pm, XX[ii])
                # elif integr =='EM':
                    # XX[ii], _, _ = self.EM(pm, XX[ii])

                ####################################################
                # Coarse trajectories
                ####################################################

                for r in range(1, dt_r):

                    step = 2 ** r
                    dtt=step*dt 

                    if (iii + 1) % step == 0:
                        thetaTmp = theta[r] + pm.Sigma*dtt
                        theta[r] = thetaTmp - np.floor(thetaTmp)
                        theta[r]=theta[0]

                        nn[r] += theta[r] - thetaTmp
                        XX[r].n=pm.n=nn[r]
 
                        XX[r].theta =pm.theta =  theta[r]
                        XX[r].dt = dtt


                        # XX[r], _, _ = self.SOILE_B(pm,XX[r] )
                        XX[r], _, _ = Fun_integ['coare'](pm,XX[r] )

                        # print(' rr',r,'--',iii,np.linalg.norm(XX[r].RF),XX[0].dt,XX[r].dt)

                        # print(' rr',r,'--',iii,np.linalg.norm(XX[r].RF2),[XX[ss].theta for ss in [0,r]])

                        print(' rr',r,'--',iii,theta)

                        XX[r].RF3.fill(0.0)
                        XX[r].RF4.fill(0.0)
                        XX[r].RF.fill(0.0)
                        XX[r].RF2.fill(0.0)
                if (iii + 1) % 4 == 0:
                    # for r in range(1,dt_r):
                    # e12p = np.linalg.norm(XX[0].p - XX[1].p)
                    # e24p = np.linalg.norm(XX[0].p - XX[2].p)
                    prel0 = XX[0].p - pm.A @ XX[0].q
                    prel1 = XX[1].p - pm.A @ XX[1].q
                    prel2 = XX[2].p - pm.A @ XX[2].q

                    e12p = np.linalg.norm(prel0 - prel1)
                    e24p = np.linalg.norm(prel1 - prel2)
                    e12 = help_fun().BCnorm(pm,XX[1].q - XX[0].q,theta=XX[0].theta)
                    e24 = help_fun().BCnorm(pm,XX[2].q - XX[0].q,theta=XX[0].theta)
                    # print('[[[[]]]]',sav[ii].error_pp)
                    sav[ii].error_pp.append([e12p,e24p])
                    sav[ii].error_qq.append([e12,e24])
                    ki+=1

                    err_file.write(f"{iii+1:d} {e12:.16e} {e24:.16e}\n")
                    # print(' rr---',r,'--',iii,np.linalg.norm(XX[r].RF2),[XX[ss].theta for ss in [0,1,2]],[(iii+1)%2**ss for ss in [0,1,2]])

                ####################################################
                # Save trajectories
                ####################################################

                for ii in range(dt_r):

                    step = 2**ii

                    # if (iii + 1) % step == 0:

                    X = XX[ii]

                    # jj = j * (pm.N // step) + save_count[ii]
                    jj = save_count[ii]
                    inv_expA = help_fun().MyExp(-X.theta * pm.Y)
                    ptmp = inv_expA @ X.p

                    sav[ii].qq[jj, :] = (inv_expA @ X.q).ravel()
                    sav[ii].pp[jj, :] = ptmp.ravel()
                    sav[ii].fo[jj, :] = (inv_expA @ X.f).ravel()

                    sav[ii].Gamma[jj, :] = (
                        pm.A @ ptmp + pm.gamma * ptmp
                    ).ravel()

                    sav[ii].time[jj,0] = X.theta
                    sav[ii].virial[jj,0] = X.virial
                    sav[ii].fDist[jj,:] = X.fDist.ravel()

                    save_count[ii] += 1

                    if abs(X.ff) > abs(fmax):
                        fmax = X.ff
                '''
                if j % max(1, pm.Nperiod // 10) == 0:

                    toc = time.time() - tic

                    print(
                        f"inc {iii:5d} "
                        f"Period {j:5d} "
                        f"elapsed {toc/60:.1f} min "
                        f"max force = {fmax:.4e}"
                    )'''

        ii=0

        sav[ii].error_pp=np.array(sav[ii].error_pp)
        sav[ii].error_qq=np.array(sav[ii].error_qq)
        print("Maximum force:", fmax)
        err_file.close()
        return sav










































