'''
Created on 2020/2/27/

@author: zwq
'''
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy 

from SysSetting import SysSetting 


def x2xerror(x0,x1):
    d = x0-x1 
    return np.sqrt(d.T.dot(d))

class Lam():
    lam = 0
    a = 0
    b = 0
    def __init__(self,lam=0,a=0,b=0):
        self.lam = lam
        self.a = a 
        self.b = b 
class Gaus():
    g = 0
    m = 0
    s = 0
    def __init__(self,g =0,m=0,s=0):
        self.g = g
        self.m=m 
        self.s=s 
        
        

class DbnTracking():
    SysSet = SysSetting()
    Niter = 4
    Thr = 0.001
    F = np.array([(0,0,1,0),
                  (0,0,0,1),
                  (0,0,0,0),
                  (0,0,0,0)])
    Fk = np.eye(4) + F*SysSet.dt
    ParticalNum = 10000
    def __init__(self,sigma=80):
        self.InitParameters()
        self.InitState(sigma)
        self.ex_AHA = {}
        self.ex_A = {}
        pass
    def input(self,data):
        '''data is PXL matrix'''
        zk = np.fft.fft(data.T,axis=0)
        fr = self.SysSet.fr 
        fs = self.SysSet.fs 
        fin=self.SysSet.SnapshotLen- np.around(fr/fs*self.SysSet.SnapshotLen)
        fin = np.int16(fin)
        self.zk = zk[fin,:] 
        self.zk = self.zk/self.SysSet.SnapshotLen
        return 0
        
    def InitParameters(self):
        M = self.SysSet.M 
        N = self.SysSet.N 
        P = self.SysSet.P 
        fr = self.SysSet.fr 
        '''Init lambda_0'''
        self.InitLambda_0 = []
        for fi, f in enumerate(fr):
            mt = self.SysSet.lambda_0[fi]
            b = 1 
            a = mt*b 
            
            self.InitLambda_0.append(Lam(mt,a,b))
        self.lambda_0 = deepcopy(self.InitLambda_0)
        '''Init s_k Lambda'''
        self.InitSk = []
        for fi, f in enumerate(fr):
            skt = []
            for m in range(M):
                mt = complex(self.SysSet.Amp[fi][0])
                
                st = self.SysSet.Amp[fi][1]
                b = 1
                a = st*b 
                st_0 = 0.1
                skt.append(Gaus(mt,Gaus(mt,mt,st_0),Lam(st,a,b)))
            self.InitSk.append(skt)
        self.sk = deepcopy(self.InitSk)
        return 0
    def InitState(self,sigma):
        self.x0 = deepcopy(self.SysSet.x0) 
        self.xk = deepcopy(self.x0)
        M = self.SysSet.M 
        N = self.SysSet.N 
        P = self.SysSet.P 
        
        self.Rxk0 = []
        for m in range(M):
            Rxkt = np.diag([0.1,0.1,0.1,0.1])
            self.Rxk0.append(Rxkt)
        self.Rxk = deepcopy(self.Rxk0)
        
        dt = self.SysSet.dt
        ''' Motion Noise'''
        Gk = np.array([[dt**2/2,0],
                       [0,dt**2/2],
                       [dt,0],
                       [0,dt]])
        Wx0 = np.array([[sigma,0],
                        [0,sigma]])
        
       
        self.Qx = Gk.dot(Wx0).dot(Gk.T) 
        return 0
    def Predictxk(self):
        M = self.SysSet.M 
        self.predictedxk =[]
        self.predictedRxk =[]
        for m in range(M):
            self.predictedRxk.append(self.Fk.dot(self.Rxk[m]).dot(self.Fk.T) + self.Qx)
            self.predictedxk.append(self.Fk.dot(self.xk[m]))
        return 0
    
    def g(self,x):
        '''Acoustic propagation function '''
        
        dx2n = x.reshape([-1,1,2]) -  self.SysSet.ArrayShape
        d = np.linalg.norm(dx2n,axis=-1,ord=2)
        
        g = 1./d
        return g,d
    def A(self,f,x):
        ''' Steering vector A'''
        g,d = self.g(x)
        C = self.SysSet.C
        tau = d/C 
        
        B = np.exp(-1j*2*np.pi*f*tau)
        A = g*B 
        return A,B,g
    def A_inv(self,f,x):
        ''' Steering vector A'''
        g,d = self.g(x)
        g = g**(-1)
        C = self.SysSet.C
        tau = d/C 
        
        B = np.exp(-1j*2*np.pi*f*tau)
        A = g*B 
        return A,B,g
    def UpdateLambda_0(self):
        M = self.SysSet.M 
        N = self.SysSet.N 
        P = self.SysSet.P 
        fr = self.SysSet.fr 
        
        for fi,f in enumerate(fr):
            self.lambda_0[fi].a = self.lambda_0[fi].a + P/2 
            self.lambda_0[fi].b = self.lambda_0[fi].b + 0.5*self.EX_zAs(fi,f)
    def EX_zAs(self,fi,f):
        M = self.SysSet.M 
        
        ex_AHA,ex_A= self.EX_Ax(f)
        self.ex_AHA.update({fi:ex_AHA})
        self.ex_A.update({fi:ex_A})
        uf = []
        lambda_f = []
        for m in range(M):
            uf.append(self.sk[fi][m].m.g)
            lambda_f.append(self.sk[fi][m].s.lam)
        uf = np.array(uf)
        uf=uf.reshape([M,-1])
        laha = 0
        for m in range(M):
            laha += 1./lambda_f[m]*ex_AHA[m,m] 
        zk = self.zk[fi].copy()
        zk = zk.reshape([-1,1])
        zas = laha + zk.conj().T.dot(self.zk[fi]) - 2*np.real(zk.conj().T.dot(ex_A)).dot(uf) + uf.conj().T.dot(ex_AHA).dot(uf)
        
        zas = np.real(zas)
        
        return zas[0,0] 
    def EX_Ax(self,f):
        M = self.SysSet.M 
        A = []
        
        for m in range(M):
            mean = self.xk[m][0:2,0]
            cov = self.Rxk[m][0:2,0:2]
#             print('cov[%d] = '%(m+1))
#             print(cov)
            xk = np.random.multivariate_normal(mean,cov,self.ParticalNum)
            At,B,g = self.A(f, xk)
            A.append(At)
        A = np.array(A)
        ex_A = A.mean(axis=1)
        ex_A = ex_A.T
        
        ex_AHA = np.zeros([M,M])
        AHA = np.linalg.norm(A,axis=-1,ord=2)**2
        AHA = AHA.mean(axis=-1)
        for m in range(M):
            ex_AHA[m,m] = AHA[m]
        return ex_AHA,ex_A
    
    def UpdateLambda(self):
        M = self.SysSet.M 
        fr = self.SysSet.fr 
        
        for fi,f in enumerate(fr):
            
            for m in range(M):
                self.sk[fi][m].s.a  += 0.5
                self.sk[fi][m].s.b  += 0.5*((self.sk[fi][m].m.g-self.sk[fi][m].m.m)**2+self.sk[fi][m].s.lam**(-1)+self.sk[fi][m].m.s)
        return 0
    
    def UpdateMu(self):
        M = self.SysSet.M 
        fr = self.SysSet.fr 
        
        for fi,f in enumerate(fr):
            for m in range(M):
                La = self.sk[fi][m].s.a/self.sk[fi][m].s.b
                self.sk[fi][m].m.m = (self.sk[fi][m].m.m*self.sk[fi][m].m.s**(-1)+La*self.sk[fi][m].m.g)/(self.sk[fi][m].m.s**(-1)+La)
                self.sk[fi][m].m.s = 1./(self.sk[fi][m].m.s**(-1)+La)
        
        return 0
    
    def UpdateSk(self):
        M = self.SysSet.M 
        fr = self.SysSet.fr 
        
        for fi,f in enumerate(fr):
            ex_AHA = self.ex_AHA[fi]
            ex_A = self.ex_A[fi]
            lam_0 = self.lambda_0[fi].a/self.lambda_0[fi].b
            lam0 = []
            mu0 = []
            for m in range(M):
                lam0.append(self.sk[fi][m].s.a/self.sk[fi][m].s.b)
                mu0.append(self.sk[fi][m].m.m)
            mu0 = np.array(mu0)
            mu0 = mu0.reshape([M,-1])
            LAM0 = np.diag(lam0)
            Lamf = lam_0*ex_AHA + LAM0
            Lamf_inv = np.linalg.inv(Lamf)
            zk = self.zk[fi].copy()
            zk = zk.reshape([-1,1])
            muf = Lamf_inv.dot(lam_0*ex_A.conj().T.dot(zk) + LAM0.dot(mu0))
            muf = muf.reshape(M)
            for m in range(M):
                self.sk[fi][m].m.g = muf[m]
                self.sk[fi][m].s.lam = Lamf[m,m]
        
        return 0 
    def Updatexk(self):
        M = self.SysSet.M 
        for m in range(M):
            xk, Rxk = self.update_xk_m(m)
            self.xk[m] = xk 
            self.Rxk[m] = Rxk
        return 0 
    def update_xk_m(self,m):
        
#         fr = self.SysSet.fr 
        xk = deepcopy(self.xk[m])
#         Rxk = deepcopy(self.Rxk[m])
        for i in range(self.Niter):
            Gradient_xk,Hessian_xk = self.GradHess_xk(xk,m)
            Hessian_xk_inv = np.linalg.inv(Hessian_xk)
            xk_n = xk - Hessian_xk_inv.dot(Gradient_xk)
            dxt = xk_n - xk 
            if(dxt[0:2,0].dot(dxt[0:2,0].T)<self.Thr):
                break
            else:
                xk = deepcopy(xk_n)
        
        return xk_n, -Hessian_xk_inv 
    
    def GradHess_xk(self,xk,m):
        fr = self.SysSet.fr
#         Rxk_0_inv = np.linalg.inv(self.Rxk[m])
#         grad_0 = -Rxk_0_inv.dot(xk-self.xk[m]) 
        Rxk_0_inv = np.linalg.inv(self.predictedRxk[m])
        grad_0 = -Rxk_0_inv.dot(xk-self.predictedxk[m])
#         A,B,g = self.A(f, xk)
#         g,d = self.g(xk)
        grad_f = np.zeros(grad_0.shape)
        Hessian_f = np.zeros(Rxk_0_inv.shape)
        for fi,f in enumerate(fr):
            grad_f_t, Hessian_f_t = self.GradHess_xk_f(xk,m,fi,f)
            grad_f+= grad_f_t
            Hessian_f+=Hessian_f_t
        grad = grad_0+grad_f
        Hess = -Rxk_0_inv+ Hessian_f 
        return grad,Hess 
    def GradHess_xk_f(self,xk,m,fi,f):
        
        C = self.SysSet.C
        A,B,g = self.A(f, xk[0:2,0])
        B = B.T
        g,d = self.g(xk[0:2,0])
        g = g.T
        dx2n = xk[0:2,0] -  self.SysSet.ArrayShape
        grad_g = np.zeros(xk.shape)
        grad_az = np.zeros(xk.shape)
        gt = dx2n.T.dot(g**4)
        gt = gt.reshape(len(gt))
        grad_g[0:2,0] = gt*np.real(self.sk[fi][m].m.g.conj()*self.sk[fi][m].m.g+self.sk[fi][m].s.lam**(-1)) 
        
        der_g_u = -g**(3)*dx2n[:,0:1]
        der_g_v = -g**(3)*dx2n[:,1:2]
        
        der_B_u = B*(-1j*2*np.pi*f/C)*g*dx2n[:,0:1]
        der_B_v = B*(-1j*2*np.pi*f/C)*g*dx2n[:,1:2]
        
        derAx_u = der_g_u*B + g*der_B_u
        derAx_v = der_g_v*B + g*der_B_v
        zk = self.zk[fi].copy()
        zk = zk.reshape([-1,1])
        grad_az[0:1,:] = np.real(self.sk[fi][m].m.g.conj()*derAx_u.conj().T.dot(zk))
        grad_az[1:2,:] = np.real(self.sk[fi][m].m.g.conj()*derAx_v.conj().T.dot(zk))
        
        Grad = grad_g + np.real(grad_az)
        
        Hessian_g = np.zeros([4,4])
        Hg = 8*g**6*dx2n**2-2*g**4
        Hg = Hg.sum(axis=0)
        Hguv = 8*g**6*dx2n[:,0:1]*dx2n[:,1:2]
        Hguv = np.sum(Hguv)
        Hessian_g[0,0] = Hg[0]
        Hessian_g[1,1] = Hg[1]
        Hessian_g[0,1] = Hguv
        Hessian_g[1,0] = Hguv
        
        Hessian_g = -Hessian_g/2*(self.sk[fi][m].m.g.conj()*self.sk[fi][m].m.g+self.sk[fi][m].s.lam**(-1)) 
        
        Hessian_mAz = np.zeros([4,4])
        ddg_u = 3*g**(5)*dx2n[:,0:1]**2-g**3
        ddg_v = 3*g**(5)*dx2n[:,1:2]**2-g**3
        ddg_uv =  3*g**(5)*dx2n[:,0:1]*dx2n[:,1:2]
        ddB_u = der_B_u*(-1j*2*np.pi*f/C)*g*dx2n[:,0:1] + B*(-1j*2*np.pi*f/C)*(-g**(3)*dx2n[:,0:1]**2+g)
        ddB_v = der_B_v*(-1j*2*np.pi*f/C)*g*dx2n[:,1:2] + B*(-1j*2*np.pi*f/C)*(-g**(3)*dx2n[:,1:2]**2+g)
        ddB_uv = der_B_v*(-1j*2*np.pi*f/C)*g*dx2n[:,0:1] + B*(-1j*2*np.pi*f/C)*(-g**(3)*dx2n[:,0:1]*dx2n[:,1:2])
        dder_A_u = ddg_u*B+2*der_g_u*der_B_u+g*ddB_u 
        dder_A_v = ddg_v*B+2*der_g_v*der_B_v+g*ddB_v 
        dder_A_uv = ddg_uv*B+der_g_u*der_B_v+der_g_v*der_B_u+g*ddB_uv 
        Hessian_mAz[0:1,0:1] = np.real(self.sk[fi][m].m.g.conj()*dder_A_u.conj().T.dot(zk))
        Hessian_mAz[1:2,1:2] = np.real(self.sk[fi][m].m.g.conj()*dder_A_v.conj().T.dot(zk))
        Hessian_mAz[0:1,1:2] = np.real(self.sk[fi][m].m.g.conj()*dder_A_uv.conj().T.dot(zk))
        Hessian_mAz[1:2,0:1] = Hessian_mAz[0:1,1:2] 
        
        Hessian_mAz = np.real(Hessian_mAz)
        Hessian =np.real( Hessian_g + Hessian_mAz)
        return Grad, Hessian
    def SpacialSpectrum(self,xrang,yrang,grid):
        
        xr = np.linspace(xrang[0],xrang[1],(xrang[1]-xrang[0])//grid[0])
        yr = np.linspace(yrang[0],yrang[1],(yrang[1]-yrang[0])//grid[1])
        
        X,Y = np.meshgrid(xr,yr)
        
        Z = np.zeros(X.shape)
        N = len(self.SysSet.fr)-1
        for xi,x in enumerate(xr):
            for yi,y in enumerate(yr):
                xk = np.zeros(2)
                xk[0] = X[yi,xi]
                xk[1] = Y[yi,xi]
                
                for fi,f in enumerate(self.SysSet.fr):
                    
                    A_inv,B,g = self.A_inv(f, xk)
                    A ,B,g = self.A(f, xk)
                    A = A.T
                    A_inv = A_inv.T
                    B = B.T
                    zk = self.zk[fi].copy()
                    zk = zk.reshape([-1,1])
#                     z = zk - A.dot(self.SysSet.Amp[fi][0])
#                     Z[yi,xi] += np.linalg.norm(z,ord=2)**2
                    Z[yi,xi] += np.abs(A_inv.conj().T.dot(zk))
#         Z -=Z.min()
        Z /=Z.max()
        
        return X,Y,Z
        
        
        
        
        
        
        
        
        
        
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
            
                
        
        
        
        
        
                