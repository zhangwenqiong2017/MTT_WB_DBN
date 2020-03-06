'''
Created on 2020/2/27

@author: zwq
'''
from WavGen import WavGen 
import numpy as np
import matplotlib.pyplot as plt
from Tracking import DbnTracking,x2xerror
from copy import deepcopy 
import matplotlib.gridspec as gridspec

mywav = WavGen()
mywav.WavTraGen()
M = mywav.SysSet.M 
N = mywav.SysSet.N
BinNum = mywav.SysSet.BinNum

color_list = ['#1f77b4', '#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
marker_list = ['+','x','*','^']

trajectories = np.array(mywav.xk).reshape([M,BinNum,4])
print(trajectories.shape) # 2,40,4
snr = mywav.SNR
print('snr shape is ', snr.shape)
plt.figure('The true received snr')

plt.grid()
print()
for m in range(M):
    for n in range(N):
        plt.plot(snr[:,n*4,m],label='N%d-T%d'%(n+1,m))
plt.legend()

plt.figure('Deployment & Trajectoris')
plt.grid()
plt.plot(mywav.SysSet.ArrayShape[:,0],mywav.SysSet.ArrayShape[:,1],'r.')
for m in range(M):
    plt.plot(trajectories[m,:,0],trajectories[m,:,1],linestyle = '-',color=color_list[m],marker = marker_list[m],label='T%d'%(m+1))
plt.legend()
plt.figure('Waveform')
plt.subplot(211)
t = np.linspace(0,(mywav.SysSet.SnapshotLen-1)/mywav.SysSet.fs,mywav.SysSet.SnapshotLen)
plt.plot(t,mywav.WanSignals[0][0].real)
plt.xlabel('time (s)')
plt.subplot(212)

fr1 = np.linspace(0,mywav.SysSet.fs/2,mywav.SysSet.SnapshotLen/2)
fr2 = np.linspace(mywav.SysSet.fs/2,0,mywav.SysSet.SnapshotLen/2)
fr = np.hstack([fr1,fr2])
ffts = np.fft.fft(mywav.WanSignals[0].T,axis=0)
plt.plot(fr2,np.abs(ffts[int(mywav.SysSet.SnapshotLen/2):mywav.SysSet.SnapshotLen,0]))
plt.xlabel('f (Hz)')
# plt.plot(np.abs(ffts[:,0]))
fin=np.int16(mywav.SysSet.SnapshotLen- np.around(mywav.SysSet.fr/mywav.SysSet.fs*mywav.SysSet.SnapshotLen) )
print(mywav.SysSet.fr)
plt.grid()


Niter = 5
Thr = 0.001
myDbnTrack = DbnTracking(sigma=10)
fig = plt.figure('All variables',figsize=(10,10),dpi=120)

outer_grid = gridspec.GridSpec(2, 2)
ax_s = fig.add_subplot(outer_grid[0,0])
ax_s.grid()
ax_s.set_xlabel(r"$s_k$")
ax_l = fig.add_subplot(outer_grid[0,1])
ax_l.set_xlabel(r"$\lambda_{0,k}$")
ax_l.grid()
ax_sl = fig.add_subplot(outer_grid[1,0])
ax_sl.set_xlabel(r"$\Lambda_{k}$")
ax_sl.grid()

ax_xk = fig.add_subplot(outer_grid[1,1])
ax_xk.set_xlabel(r"$x_{k}$")
ax_xk.grid()

# plt.figure('Tracking:',dpi=100)
# 
# plt.grid()
ax_xk.plot(mywav.SysSet.NodePosition[:,0],mywav.SysSet.NodePosition[:,1],'r^')
for m in range(M):
    ax_xk.plot(trajectories[m,:,0],trajectories[m,:,1],marker = marker_list[m],color=color_list[m])
xk_buf = []
for k, data in enumerate(mywav.WanSignals):
    myDbnTrack.input(data)
    
    
    myDbnTrack.Predictxk()
    
    xk_0 = deepcopy(myDbnTrack.xk)
    for i in range(Niter):
        myDbnTrack.UpdateLambda_0()
        
        myDbnTrack.UpdateLambda()
        myDbnTrack.UpdateMu()
        myDbnTrack.UpdateSk()
        myDbnTrack.Updatexk()
        for fi,f in enumerate(myDbnTrack.SysSet.fr):
            for m in range(M):
                ax_s.plot([f],[myDbnTrack.sk[fi][m].m.m],marker = '.')
                ax_sl.plot([f],[myDbnTrack.sk[fi][m].s.a/myDbnTrack.sk[fi][m].s.b],marker = '.')
#             print(myDbnTrack.lambda_0[fi].a/myDbnTrack.lambda_0[fi].b)
            ax_l.plot([f],[myDbnTrack.lambda_0[fi].a/myDbnTrack.lambda_0[fi].b],marker = '.')
        xk = myDbnTrack.xk
        print('k=%d i=%d'%(k+1,i+1))
        ei = np.zeros(M)
        for m in range(M):
            e = x2xerror(xk[m],xk_0[m])
            xk_0 = deepcopy(myDbnTrack.xk)
            if(e<Thr):
                ei[m]=1
        if(e.sum()==M):
            break
        
           
    xk_buf.append(myDbnTrack.xk)
    for m in range(M):
        ax_xk.plot([myDbnTrack.xk[m][0,0]],[myDbnTrack.xk[m][1,0]],color=color_list[m+M],marker = marker_list[m])
    
    plt.figure('SS')
    
    X,Y,Z = myDbnTrack.SpacialSpectrum([-30,30], [-30,30], [2,2])
    plt.clf()
    plt.contourf(X,Y,Z)
#     plt.grid()
    plt.plot(mywav.SysSet.NodePosition[:,0],mywav.SysSet.NodePosition[:,1],'r^')
    plt.show(block=False)
    plt.pause(0.1)
#     plt.show()
plt.show()












    


