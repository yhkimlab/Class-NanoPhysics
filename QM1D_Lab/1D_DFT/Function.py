#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 12:54:22 2020

@author: ramanujan
"""

import numpy as np
import matplotlib.pyplot as plt


#Functions
def integral(x,y,axis=0):
    dx=x[1]-x[0]
    return np.sum(y*dx, axis=axis)

def print_log(i,log):
    print(f"step: {i:<5} energy: {log['energy'][-1]:<10.4f} energy_diff: {log['energy_diff'][-1]:.10f}")
    
def get_nx(num_electron, psi, x):
    # normalization
    I=integral(x,psi**2,axis=0)
    normed_psi=psi/np.sqrt(I)[0]    #CHECK NORMAL FORM WAS np.sqrt(I)[None,:]
    # occupation num
    fn=[2 for _ in range(num_electron//2)]
    if num_electron % 2:
        fn.append(1)
    res=np.zeros_like(normed_psi[:,0])
    for ne, psi  in zip(fn,normed_psi.T):
        res += ne*(psi**2)
    return res

def get_exchange(nx,x):
    energy=-3./4.*(3./np.pi)**(1./3.)*integral(x,nx**(4./3.))
    potential=-(3./np.pi)**(1./3.)*nx**(1./3.)
    return energy, potential

def get_hatree(nx,x, eps=1e-1):
    h=x[1]-x[0]
    energy=np.sum(nx[None,:]*nx[:,None]*h**2/np.sqrt((x[None,:]-x[:,None])**2+eps)/2)
    potential=np.sum(nx[None,:]*h/np.sqrt((x[None,:]-x[:,None])**2+eps),axis=-1)
    return energy, potential

def geteigenenergy(nx, x, num_electron, eig):  
    # occupation num
    fn=[2 for _ in range(num_electron//2)]
    if num_electron % 2:
        fn.append(1)
    eigenergy = sum(fn*eig[:len(fn)])
    return eigenergy

def ex_integral(nx, x, num_electron, ex_potential):
    int_ex = integral(x, nx*ex_potential)
    return int_ex

def cal_total_energy(ha_energy, ex_energy, eigenergy, int_ex):
    totalE = eigenergy - ha_energy + ex_energy - int_ex
    return totalE

def Fermi(e,ef,T):
    if T==0:
        if e>ef: fermi=0
        if e<ef: fermi=1
        if e==ef: fermi=0.5
    else:
        tmp = (e-ef)/(T*8.617343*0.00001)
        if abs(tmp) < 40:
            fermi = 1/(1+np.exp(tmp))
        else:
            if tmp < -40: fermi=1
            if tmp > 40: fermi=0
    return fermi