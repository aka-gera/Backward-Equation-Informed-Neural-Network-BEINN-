import numpy as np





class PBCs:
    def __init__(self, flow, epsilon, nPart, rcut, N, Nperiod,
                 force='zero',
                 beta=1,
                 gamma=1)-> None: 
        if nPart < 1:
            raise ValueError("The number of particles is invalid")


        if flow == 'eld':
            if nPart <= 2:
                a = 10
            else:
                a = 2*nPart
            dim = 2
            A = np.array([[0,0],[0,0]])
            invL0 = np.eye(dim)/a
            Y = A
            Yoff = np.zeros((dim,dim))
            Sigma = 1
            
        elif flow == 'shear':
            # Shear flow case with LE
            if nPart <= 2:
                a = 10
            else:
                a = 6*nPart
            dim = 2
            A = epsilon * np.array([[0, 1], [0, 0]])
            invL0 = np.eye(dim) / a
            Y = A
            Yoff = np.zeros((dim, dim))
            Sigma = epsilon
        elif flow == 'pef':
            # PEF case with KR
            if nPart <= 4:
                a = 20
            else:
                a = 6*nPart
            dim = 2
            A = epsilon * np.array([[-1, 0], [0, 1] ])
            # M = np.array([[2, -1, 0], [-1, 1, 0], [0, 0, 1]])
            V = np.array([  [-0.52573111, -0.85065081],
                            [-0.85065081,  0.52573111] 
                            ])


            Y = np.array([-0.96242365,  0.96242365])
            Yoff = np.zeros((dim, dim))
            invL0 = V / np.abs(np.linalg.det(V)) ** (1/2) / a 
            Sigma = epsilon/Y[1]
     
        self.epsilon = epsilon
        self.flow = flow
        self.L0inv = invL0
        self.L0 = np.linalg.inv(invL0)
        self.Linv = self.L0inv
        self.L = self.L0
        self.A = A
        self.Y = Y
        self.Yoff = Yoff
        self.Sigma = Sigma
        self.T = 1/abs(Sigma)  
        self.n = 0
        self.dt = self.T / N
        self.theta=self.theta1 = np.ceil(self.Sigma*self.dt)-self.Sigma*self.dt
        self.N = N
        self.Nperiod = Nperiod


        dim = A.shape[0]-1 

        self.sigm = 4
        self.eps = 1
        self.rcut = rcut
        self.dim = dim
        self.dimm = A.shape[0]
        self.gamma = gamma
        self.beta = beta
        self.a = a
        self.nPart = nPart
        self.Mmax = int(np.ceil(a/rcut*nPart))
        self.vol = a**self.dimm * nPart

        self.force=force

        self.fluct  =np.sqrt(2.0 * self.gamma * self.dt / self.beta)
        self.fluct2 = np.sqrt(2.0 * self.gamma * self.dt**3 /self.beta)
        self.k=1
        self.r0=0





class particles():
    def __init__(self,pm ): 
        dim,nPart=pm.dimm,pm.nPart
        pdim=(dim,nPart) 
        
        self.q = np.zeros(pdim)
        self.qDist = np.zeros(pdim)
        self.fDist = np.zeros(pdim)
        self.p = np.zeros(pdim) 
        self.prel = np.zeros(pdim) 
        self.f = np.zeros(pdim) 
        self.ff = 0.0
        # self.theta = 1.0
        # self.n = 0
        # self.L= np.zeros((dim+1,dim+1))
        # self.Linv= np.zeros((dim+1,dim+1))
        # self.dt=None
        self.virial =0.
        # self.G = np.zeros(pdim)
        self.error_qq = np.zeros((1,2))
        self.error_pp = np.zeros((1,2))

        self.RF  = np.zeros(pdim)
        self.RF2 = np.zeros(pdim)
        self.RF3 = np.zeros(pdim)
        self.RF4 = np.zeros(pdim)


        # xx = np.random.randn(dim+1, nPart)
        # yy = np.random.randn(dim+1, nPart)

        # fluct  = np.sqrt(2 * pm.gamma * pm.dt / pm.beta)
        # fluct2 = np.sqrt(2 * pm.gamma * pm.dt**3 / pm.beta)
 

        # self.RF  = xx * fluct

        # self.RF2 = 0.5 * fluct2 * (xx + yy / np.sqrt(3.0))

        # self.RF3 = fluct * xx

        # self.RF4 = 0.5 * fluct2 * (xx + yy / np.sqrt(3.0))






class saves():
    def __init__(self,pm): 
        # self.Q1 = np.zeros((pm.N,pm.Nperiod))
        # self.Q2 = np.zeros((pm.N,pm.Nperiod))
        # self.F = np.zeros((pm.N,pm.Nperiod))
        tdim = pm.dimm*pm.nPart
        ppdim=(pm.N*pm.Nperiod,tdim)
        self.qq = np.zeros(ppdim)
        self.pp = np.zeros(ppdim)
        self.fo = np.zeros(ppdim)
        self.fDist = np.zeros(ppdim)
        self.time = np.zeros((pm.N*pm.Nperiod,1))
        self.error_qq =[]# np.zeros((1,2))
        self.error_pp =[]# np.zeros((1,2))


        # self.Gamma = np.zeros((pm.N*pm.Nperiod,tdim))
        # self.expA = np.zeros((pm.N*pm.Nperiod,tdim))
        # self.virial = np.zeros((pm.N*pm.Nperiod,1))
        '''
        self.RF  = np.zeros((pm.N*pm.Nperiod,tdim))

        self.RF2 = np.zeros((pm.N*pm.Nperiod,tdim))
        self.RF3 = np.zeros((pm.N*pm.Nperiod,tdim))

        self.RF4 = np.zeros((pm.N*pm.Nperiod,tdim))
'''


class paramFig():
    def __init__(self,flow , sbox):
        mm = 100
        lSpace = np.linspace(1, np.exp(1), mm)
        II = np.ones(mm)
        I0 = np.zeros(mm)
        III = II - np.log(lSpace)

        dat = {
            'mapp': [II, III, III],
            'MainBoxColor': [1, 0, 0],
            'MainBoxEdge': ':',
            'MainBoxOpaque': 0.05,
            'MainBoxMarkerWidth': 1,
            'Color': 'r',
            'GridEdge': 'o',
            'GridColor': [0, 0, 0],
            'GridMarkerWidth': 2,
            'ft': 20,
            'AxisWidth': 3,
            'AxisColor': 'b',
            'aa': 15,
            'bb': 1
        }

        if flow == 'eld':
            dat['mapp'] = [II, I0, I0]
            dat['Angle'] = [0, 90]
            dat['posTextX'] = -2
            dat['posTextY'] = 4
            dat['posTextZ'] = 7
            dat['aa'] = 0
            dat['bb'] = 20
            dat['xmin'] = -1.1 * sbox
            dat['xmax'] = 2.1 * sbox
            dat['ymin'] = -1.01 * sbox
            dat['ymax'] = 2.01 * sbox
            dat['zmin'] = -1.9 * sbox
            dat['zmax'] = 1.17 * sbox
            dat['center'] = [0, 0, 1]
            dat['radius'] = 1.5
            dat['centerOff'] = 0
        elif flow == 'shear':
            dat['mapp'] = [II, I0, I0]
            dat['Angle'] = [0, 90]
            dat['posTextX'] = -2
            dat['posTextY'] = 4
            dat['posTextZ'] = 7
            dat['aa'] = 0
            dat['bb'] = 20
            dat['xmin'] = -1.1 * sbox
            dat['xmax'] = 2.1 * sbox
            dat['ymin'] = -1.01 * sbox
            dat['ymax'] = 2.01 * sbox
            dat['zmin'] = -1.9 * sbox
            dat['zmax'] = 1.17 * sbox
            dat['center'] = [0, 0, 1]
            dat['radius'] = 1.5
            dat['centerOff'] = 0
        elif flow == 'pef':
            dat['mapp'] = [II, I0, I0]
            dat['Angle'] = [180, 90]
            dat['posTextX'] = 0
            dat['posTextY'] = -5
            dat['posTextZ'] = 7
            dat['aa'] = 0
            dat['bb'] = 20
            dat['xmin'] = -2.7 * sbox
            dat['xmax'] = 2.3 * sbox
            dat['ymin'] = -1.7 * sbox
            dat['ymax'] = 1.7 * sbox
            dat['zmin'] = -1.9 * sbox
            dat['zmax'] = 1.17 * sbox
            dat['center'] = [-0.75, -0.3, 1]
            dat['radius'] = -1.75
            dat['centerOff'] = 0.4

        dat['Axi'] = [dat['xmin'], dat['xmax'], dat['ymin'], dat['ymax'], dat['zmin'], dat['zmax']]

        xlength = 2.5
        ylength = 2  # Assuming there was an error in the MATLAB code where "2.5" shouldn't be here

        dat['xmax'] = sbox# xlength * sbox  # size of x dimension of graph
        dat['ymax'] = sbox# ylength * sbox  # size of y dimension of graph
        dat['zmax'] = 1  # size of z dimension of graph

        aa = dat['xmax']+xlength * sbox
        bb = dat['ymax']+ylength * sbox
        cc = dat['zmax']

        x = np.arange(-aa, aa + 1)
        y = np.arange(-bb, bb + 1)
        z = np.arange(-cc, cc + 1)

        xx, yy, zz = np.meshgrid(x, y, z)

        dat['PP'] = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))  
        

        self.dat = dat
    
    def data_replicas(self,L, q, dat):
        # Input
        # L : simulation box in 2 dimensions
        # q : position of the particles
        # dat : unit Lattice grid

        # Output
        # qq : position of the particles in the simulation box and its replicas
        # Lk : simulation box in 3 dimensions
        # LB : simulation box with its replicas

        LL = np.dot(L, dat['PP'].T)
        # inds =  (LL[0, :] < dat['xmax']) & (LL[0, :] > -dat['xmax']) \
        #       & (LL[1, :] < dat['ymax']) & (LL[1, :] > -dat['ymax']) \
        #       & (LL[2, :] < dat['zmax']) & (LL[2, :] > -dat['zmax'])

        mm = q.shape[1]
        LB = LL#[:, inds] 
        nn = LB.shape[1] 
        qL = LB[0:3, :] 

        qTemp = qL + np.tile(q[:, 0], (1, nn)).reshape(nn,3).T 
        indsTemp = (qTemp[0, :] < dat['xmax']) & (qTemp[0, :] > -dat['xmax']) \
                & (qTemp[1, :] < dat['ymax']) & (qTemp[1, :] > -dat['ymax'])  \
                & (qTemp[2, :] < dat['zmax']) & (qTemp[2, :] > -dat['zmax']) 
        
        qq =qTemp[:,indsTemp]

        for i in range(mm-1): 
            qTemp = qL + np.tile(q[:, i+1], (1, nn)).reshape(nn,3).T 
            indsTemp = (qTemp[0, :] < dat['xmax']) & (qTemp[0, :] > -dat['xmax']) \
                & (qTemp[1, :] < dat['ymax']) & (qTemp[1, :] > -dat['ymax'])  \
                & (qTemp[2, :] < dat['zmax']) & (qTemp[2, :] > -dat['zmax']) 
            qq =np.hstack((qq, qTemp[:,indsTemp]))
            
        return qq
    
    
# class pm:
#     def __init__(self,param,pbc):
#         self.param = param
#         self.pbc = pbc 
#         self.flow = pbc['flow']
#         self.L0inv = pbc['L0inv']
#         self.L0 =  pbc['L0']
#         self.Linv = pbc['Linv']
#         self.L = pbc['L']
#         self.A = pbc['A']
#         self.Y = pbc['Y']
#         self.Yoff = pbc['Yoff']
#         self.Sigma = pbc['Sigma']
#         self.T =  pbc['T']
#         self.theta = pbc['theta']
#         self.theta1 = pbc['theta1']
#         self.n = pbc['n']
#         self.dt = pbc['dt'] 
#         self.Nperiod = pbc['Nperiod']
#         self.N = pbc['N'] 

#         self.nPart = param['nPart'] 
#         self.dim = param['dim']
#         self.gamma = param['gamma']
#         self.beta = param['beta']
#         self.eps = param['eps']
#         self.sigm = param['sigm']
#         self.rcut = param['rcut']

