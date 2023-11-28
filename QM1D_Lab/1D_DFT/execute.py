#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 13:32:01 2020

@author: ramanujan
"""
import DFTcal as DFTcal
import Inputs as inputs
import matplotlib.pyplot as plt
import numpy as np

#Declare inputs
n_grid = inputs.n_grid; xi = inputs.xi; xf = inputs.xf; NE = inputs.NE
epsil = inputs.epsil; effective_mass = inputs.effective_mass; pw = inputs.pw
ph = inputs.ph; pot_type = inputs.pot_type; bias = inputs.bias
max_iter = inputs.max_iter; num_electron = inputs.num_electron; 
itercrit = inputs.itercrit; scaled = inputs.scaled; mix = inputs.mix
energy_tolerance = inputs.energy_tolerance
occupation_scheme = inputs.occupation_scheme
efl = inputs.efl; efr = inputs.efr; T = inputs.T
Emin = inputs.Emin; Emax = inputs.Emax; zplus = inputs.zplus
D = inputs.D


#unit conversion
au2ang = 0.5291772109; au2eV = 27.211386245
ang2au = 1 / au2ang; eV2au = 1 / au2eV
eu2ang = au2ang * epsil/effective_mass 
eu2eV = au2eV * effective_mass /epsil
ang2eu = 1 / eu2ang
eV2eu = 1 / eu2eV
au_length2eu = effective_mass/epsil; au_energy2eu = epsil/effective_mass
eu2au_length = 1 / au_length2eu; eu2au_energy = 1 / au_energy2eu


if __name__ == '__main__':
    #Excute DFT calculation
    DFT_cal = DFTcal.DFT(n_grid, xi, xf, epsil, effective_mass, pw, ph, pot_type, bias, \
                 max_iter, num_electron, itercrit, scaled, \
                 mix, energy_tolerance, occupation_scheme)
    
    nx, H, psi, energy, ex, ha, pot = DFT_cal.DFT_calculation()
    
    #Transmission cal
    trans = DFTcal.transmission(D, xi, xf, n_grid, H, NE, Emin, Emax, zplus, pot, ex, ha, epsil, effective_mass, efl, efr, T)
    
    x, E, t_butti_L, t_balli, t_relax, F, I_balli, I_butti = trans.Buttiker_transmission_cal()
    
    #unit conversion
    E = E * eu2eV
    x = x * eu2ang
    pot = pot * eu2eV
    energy = energy * eu2eV
    
    #plotting potential
    plt.figure(1)
    plt.title('Potential barrier', fontsize = 15)
    plt.plot(x, pot, label = 'Potential')
    plt.xlabel('$x$ [$\AA$]', fontsize = 12)
    plt.ylabel('Potential [eV]', fontsize = 12)
    
    #plotting transmission
    plt.figure(2)
    plt.title('Electron transmission', fontsize = 15)
    
    plt.plot(E, t_balli, color = 'r', label = "ballistic transport")
    plt.plot(E, t_relax, color ='b', label = "scattering")  
    plt.legend(loc=1)
    #plt.ylim(0, 1e-5)
    plt.xlabel('Energy [eV]', fontsize = 12)
    plt.ylabel('Transmission', fontsize = 12)
    
    #plotting occupation
    plt.figure(3)
    plt.title('Electrochemical potential profile', fontsize = 15)
    plt.plot(x, F)
    plt.xlabel('$x$ [$\AA$]', fontsize = 12)
    plt.ylabel('$f$', fontsize = 12)
    
    #plot eigenvalue
    plt.figure(4)
    plt.title('Eigen energy', fontsize = 15)
    plt.xlabel(r"x [$\AA$]", fontsize = 12)
    plt.ylabel("wavefunction", fontsize = 12)
    for i in range(5):
        plt.plot(x,psi[:,i], label=f"{energy[i]:.4f} eV")
        plt.legend(loc=1)
  
    #plot density
    plt.figure(5)
    plt.title('Density', fontsize = 15)
    plt.plot(x, nx)
    plt.xlabel("$x$ [$\AA$]", fontsize = 12)
    plt.ylabel("$rho$", fontsize = 12)