# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 10:07:41 2023

@author: Thomas
"""
import numpy as np 
import os 
    
def Zfunc(N,R0,epsilonZ,alphaZ,phi):
    return R0*epsilonZ*np.sin(N*phi+alphaZ*np.sin(N*phi))

def Radfunc(N,R0,epsilonR,alphaR,phi):
    return R0*epsilonR*np.cos(N*phi+alphaR*np.sin(N*phi))+R0

def etos(npos=200,R_1=.31,current_ratio=-4.5,epsilon_R1=0.58,alpha_R1=0.65,epsilon_Z1=0.65,alpha_Z1=0.65,R_2=0.34):

    coil_pos=np.zeros((2,3,npos))
    currents=np.zeros(2)
    phi = np.linspace(0,2*np.pi,npos)

    N=5
    I0=10.e3
    
    rr=Radfunc(N,R_1,epsilon_R1,alpha_R1,phi)
    coil_pos[0,2,:]=Zfunc(N,R_1,epsilon_Z1,alpha_Z1,phi)
    coil_pos[0,0,:]=rr*np.cos(phi)
    coil_pos[0,1,:]=rr*np.sin(phi)
    currents[0]=I0

    coil_pos[1,0,:]=R_2*np.cos(phi)
    coil_pos[1,1,:]=R_2*np.sin(phi)
    coil_pos[1,2,:]=0*phi
    currents[1]=I0/current_ratio

    return(coil_pos,currents)


def make_coils(config, npos, R_1, R_2, epsilon_R, alpha_R, epsilon_Z, alpha_Z, current_ratio, exec_path):
    with open(exec_path+"/coils.etos_"+str(config), "w") as f:
        f.write('periods 5\n' +
                'begin filament\n' +
                'mirror NUL\n')
        coil_pos, currents = etos(npos=npos, R_1=R_1, current_ratio=current_ratio, epsilon_R1=epsilon_R, \
                                     alpha_R1=alpha_R, epsilon_Z1=epsilon_Z, alpha_Z1=alpha_Z, R_2=R_2)
        for i in range(2):
            for j in range(npos):
                
                f.write('\t {0:0.8f} \t {1:0.8f} \t {2:0.8f} \t -1\n'.format(
                    coil_pos[i, 0, j], coil_pos[i, 1, j], coil_pos[i, 2, j]))
            f.write('\t {0:0.8f} \t {1:0.8f} \t {2:0.8f} \t 0 \t {3} MainCoil{3}\n'.format(
                coil_pos[i, 0, j], coil_pos[i, 1, j], coil_pos[i, 2, j], i+1))
    
        f.write('end')
        f.close() 
    
        
    return(coil_pos, currents)

if __name__ == '__main__':  
    epath=os.getcwd()
    make_coils('231', 1800, R_1=.2, R_2 =.368, epsilon_R=.491, alpha_R=.25, epsilon_Z=.441,\
               alpha_Z=.7, current_ratio=-2.5, exec_path=epath)
    
    
    