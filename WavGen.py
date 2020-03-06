'''
Created on 2020/2/26

@author: zwq
'''

import numpy as np
import matplotlib.pyplot as plt


from SysSetting import SysSetting 



class WavGen():
    SysSet = SysSetting()
    SNR = []
    Fk = []
    F = np.array([(0,0,1,0),
                  (0,0,0,1),
                  (0,0,0,0),
                  (0,0,0,0)])
    for i in range(SysSet.BinNum):
        Fk.append(np.eye(4) + F*SysSet.dt)
    
    def __init__(self):
        x0 = self.SysSet.x0 
        r = np.sqrt(x0[0][0:2,0].dot(x0[0][0:2,0].T))
        w = np.sqrt(x0[0][2:4,0].dot(x0[0][2:4,0].T))/r
        theta0 = np.arctan2(x0[0][1,0],x0[0][0,0])
        dt = self.SysSet.dt
        self.xk0 = []
        for i in range(self.SysSet.BinNum):
            xt = np.array([[r*np.cos(-w*dt*i+theta0)],
                            [ r*np.sin(-w*dt*i+theta0)],
                            [-r*w*np.sin(-w*dt*i+theta0)],
                            [r*w*np.cos(-w*dt*i+theta0)]])
            self.xk0.append(xt)

    def g(self,x):
        '''Acoustic propagation function '''
        dx2n = x[0:2,0] -  self.SysSet.ArrayShape
        d = np.linalg.norm(dx2n,axis=-1,ord=2)
#         d = np.diag(dx2n.dot(dx2n.T))
#         d = np.sqrt(d)
        g = 1./d
        return g.reshape([len(g),-1]),d.reshape([len(d),-1])
    def A(self,f,x):
        ''' Steering vector A'''
        g,d = self.g(x)
        C = self.SysSet.C
        tau = d/C 
        
        B = np.exp(-1j*2*np.pi*f*tau)
        A = g*B 
        return A,B,g
    def WGenSingleFreq(self,fi,amp,f,x):
        P = self.SysSet.P 
        L = self.SysSet.SnapshotLen
        fs = self.SysSet.fs
        n = 1.0/2.0**0.5*(np.random.normal(0, 1, (P,L))+np.random.normal(0, 1, (P,L))*1j)*self.SysSet.lambda_0[fi]**(-0.5)
        t = np.linspace(0, 1./fs*(L-1),L)
        Amp = amp[0]+1.0/2.0**0.5*(np.random.normal(0, 1,[len(amp[0]),1])+np.random.normal(0,1,[len(amp[0]),1])*1j)*amp[1]**(-0.5)
        ss = np.exp(-1j*2*np.pi*f*t)*Amp
#         ss = np.exp(-1j*2*np.pi*f*t)*amp[0]
#         print(ss.shape)
        A = []
        g=[]
        for xi in x:
            At,B,gt = self.A(f, xi)
            A.append(At)
            g.append(gt)
#             print(xi,g)
        A = np.hstack(A)
        g = np.hstack(g)
#         print(A.shape)
#         s = A.dot(ss)+np.exp(-1j*2*np.pi*f*t)*n
#         Ax = np.zeros(A.shape)
#         Ax[:,1] = A[:,1]
#         A = Ax
        s = A.dot(ss)+np.exp(-1j*2*np.pi*f*t)*n
#         Amp = np.ones(Amp.shape)*10
        return s,g**2*np.abs(Amp.T)**2
    def WGenWBFreq(self,x):
        P = self.SysSet.P 
        L = self.SysSet.SnapshotLen
        M = self.SysSet.M 
        s = np.zeros([P,L],dtype=complex)
        FreqDist = self.SysSet.FreqDist
        
        realAmp = []
        for fi,f in enumerate(self.SysSet.fr):
            amp = []
            '''FreqDist.append([Amp]) 
                FreqDist.append([Amp])'''
            mu = []
            lambda_f = []
            for m in range(M):
                mu.append(FreqDist[m][fi][0])
                lambda_f.append(FreqDist[m][fi][1])
            mu = np.array(mu).reshape([M,-1])
            lambda_f = np.array(lambda_f).reshape([M,-1])
            amp.append(mu)
            amp.append(lambda_f)
            sf,realAmp_f = self.WGenSingleFreq(fi,amp,f,x)
            realAmp.append(realAmp_f) #.sum(axis = 1)
            s+= sf
        realAmp = np.array(realAmp)
        
#       
#         snr = realAmp#/(len(self.SysSet.fr)*self.SysSet.lambda_0[fi]**(-1))
#             print('f=%.2f'%(f))
        return s,realAmp
    def TraGen(self):
        #print(xk.shape)
     
        xk1 = []
        xt = self.SysSet.x0[1].copy()
        xk1.append(xt)
        for k in range(self.SysSet.BinNum-1):
            xt = self.Fk[k].dot(xt)
            xk1.append(xt)
        self.xk = []
        self.xk.append(self.xk0)
        self.xk.append(xk1)
        return 0
    def WavTraGen(self):
        self.TraGen()
        self.WanSignals = []
        
        for k in range(self.SysSet.BinNum):
            xt = []
            for m in range(self.SysSet.M):
                xt.append(self.xk[m][k])
#             print(xt)
            s,e = self.WGenWBFreq(xt)
            self.WanSignals.append(s)
            self.SNR.append(e)
            print('Bin No.%d'%(k+1))
        self.SNR = np.array(self.SNR)
        lambda_0 = np.array(self.SysSet.lambda_0)
        self.SNR = np.log10(self.SNR.sum(axis=1)/(np.sum(1./lambda_0)))*10
        return 0
        






















