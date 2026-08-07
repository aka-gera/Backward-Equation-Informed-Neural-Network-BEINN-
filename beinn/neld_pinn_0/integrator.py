

import numpy as np
import time 
from beinn.neld_pinn_0.help_fun import help_fun
import copy

class Integrator(help_fun):
    def __init__(self):
        help_fun.__init__(self)


    def EM(self, pm, X,dt): 

        # dt = pm.dt
        xi = pm.gamma          # friction coefficient (check this)
        A = pm.A
        # theta =X.theta
        # dt=pm.dt


        nPart = pm.nPart
        dim = pm.dim 
        gamma = pm.gamma
        beta = pm.beta
        A = pm.A
    
        qtmp = X.q + X.p * dt
        # # Compute force
        X = self.ComputeForce(pm,X) 

        # Momentum update
        X.p += (
            X.f * dt
            - X.p * (dt* xi)
            + A @ X.q * dt * xi
            + A @ X.p * dt
            + X.RF3
        )

        # Update position 
        X.q = qtmp

        # Peculiar momentum
        X.prel = X.p - A @ X.q 

        return X


 



    def SOILE_A(self, pm, X, dt):

        xi = pm.gamma
        A = pm.A

        # Gamma = xi*I - A
        Gamma = xi * np.eye(pm.dimm) - A 

        rhs = X.f + xi * (A @ X.q) - Gamma @ X.p

        C2 = 0.5 * dt**2 * rhs + X.RF2

        Ftmp = X.f + xi * (A @ X.q)
 

        X.q = X.q + dt * X.p + C2

        X.prel = X.p - A @ X.q

        #--------------------------------------------------
        # Force evaluation on a wrapped copy
        #--------------------------------------------------
        Xtmp = copy.deepcopy(X)

        # Wrap ONLY the temporary copy
        self.remap_coord(Xtmp, pm)

        # Compute forces using wrapped coordinates
        Xtmp = self.ComputeForce(pm, Xtmp)

        # Copy force back
        X.f = Xtmp.f
        X.ff = Xtmp.ff
        X.fDist = Xtmp.fDist
        X.virial = Xtmp.virial
        '''
'''
        #--------------------------------------------------
        # Correct momentum
        #--------------------------------------------------

        X.p = (
            X.p
            + 0.5 * dt * (Ftmp + X.f + xi * (A @ X.q))
            - Gamma @ (dt * X.p + C2)
            + X.RF
        )

        X.prel = X.p - A @ X.q

        return X















    def SOILE_B(self, pm, X,dt): 

        # dt = pm.dt
        xi = pm.gamma          # friction coefficient (check this)
        A = pm.A
        theta =pm.theta
        # dt=pm.dt
        # Gamma = xi*I - A
        Gamma = xi * np.eye(pm.dimm) - A
 

        rhs = X.f + xi * (A @ X.q) - Gamma @ X.p

        X.p += (
            0.5 * dt * rhs
            - 0.25 * (Gamma @ (0.5 * dt**2 * rhs
                            + 2.0 * (X.RF4 - 0.25 * dt * X.RF3)))
            + 0.5 * X.RF3
        )
 

        X.q += dt * X.p + X.RF4 - 0.5 * dt * X.RF3

        X.prel = X.p - A @ X.q 

        Xtmp = copy.deepcopy(X) 
        self.remap_coord(Xtmp, pm) 

        Xtmp = self.ComputeForce(pm, Xtmp)
 
        X.f = Xtmp.f
        X.ff = Xtmp.ff
        X.fDist = Xtmp.fDist
        X.virial = Xtmp.virial
        '''
''' 

        rhs = X.f + xi * (A @ X.q) - Gamma @ X.p

        X.p += (
            0.5 * dt * rhs
            - 0.25 * (Gamma @ (0.5 * dt**2 * rhs
                            + 2.0 * (X.RF4 - 0.25 * dt * X.RF3)))
            + 0.5 * X.RF3
        )

        X.prel = X.p - A @ X.q
 

        return X

 
