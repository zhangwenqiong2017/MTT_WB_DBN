'''
Created on 2020/2/26/

@author: zwq
'''
import sys
import os

import numpy as np 
from numpy import dtype
from time import sleep, strftime

class SysSetting():
    '''Acoustic array aperture:'''
    R = 15
    
    NodeArrayShape = np.array([[R,0.],
                           [0.,R],
                           [-R,0.],
                           [0.,-R]])
    Pn = len(NodeArrayShape)
    '''Node deployment'''
    Dist = 32. 
    NodePosition = np.array([[Dist,-Dist],
                           [Dist,Dist],
                           [-Dist,Dist],
                           [-Dist,-Dist]])
    N = len(NodePosition)
    '''Acoustic Propagation Speed '''
    C = 340.0
    '''Target initial states'''
    BinNum = 80 # The number of the whole trajectory. 
    fs = 4000  
    SnapshotLen = 4096 
    dt = SnapshotLen / fs 
    x0 = []
    x0.append(np.array([[0],
                        [20],
                        [2.0*np.pi/BinNum*20.0/dt],
                        [0]]))
    x0.append(np.array([[-32],
                        [0],
                        [Dist*2/BinNum/dt],
                        [0]]))
    M = len(x0) # The number of all the targets
    '''
    FreqRange =  [Beg End Stride]
    '''
    FreqDist = []
    fr = np.linspace(10,20,4)
    '''Target acoustic signal amplitude parameters''' 
    Amp = []
    lambda_0 = []
    for i,f in enumerate(fr):
        mu = 1
        lambda_f = 10
        Amp.append([mu,lambda_f])
        lambda_0.append(100) 
    FreqDist.append(Amp) 
    FreqDist.append(Amp)
    
    
    '''
    Merge all node sensors in one shape:
    '''
    
    ArrayShape = np.zeros([Pn*N,2])
    
    for ni,n in enumerate(NodePosition):
        for ai, a in enumerate(NodeArrayShape):
            ArrayShape[ni*Pn+ai,:] = n+a 
            
    P = len(ArrayShape) # The number of all the sensors in ASAN
    
    
    
    
    
    def __init__(self):
        pass


# MySysSetting = SysSetting()
# print(MySysSetting.ArrayShape)