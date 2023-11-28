    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:22:40 2020

@author: ramanujan
"""


import numpy as np
import matplotlib.pyplot as plt
import Function as func
import Inputs as inputs

au2ang = 0.5291772109; au2eV = 27.211386245
ang2au = 1 / au2ang; eV2au = 1 / au2eV


#set Hamiltonican class
class Hamiltonian:
    def __init__(self, n_grid, xi, xf, epsil, effective_mass, pw, ph, pot_type, bias):
        self.n_grid = n_grid
        self.ep = epsil; self.em = effective_mass
        self.potential_shape = pot_type        #potential height(ph), potential width(pw)
        
        #effectiv mass atomic units
        self.eu2ang = au2ang * self.ep / self.em
        self.eu2eV = au2eV * self.em / self.ep
        self.ang2eu = 1 / self.eu2ang
        self.eV2eu = 1 / self.eu2eV
        self.au_length2eu = self.em/self.ep; self.au_energy2eu = self.ep/self.em
        self.eu2au_length = 1 / self.au_length2eu; self.eu2au_energy = 1 / self.au_energy2eu
        
        #potential
        self.ph = ph * self.eV2eu
        self.bias = bias * self.eV2eu
        
        #scaled length
        self.xi = xi * self.ang2eu; self.xf = xf * self.ang2eu
        self.x = np.linspace(self.xi, self.xf, n_grid)
        self.pw = pw * self.ang2eu
        self.h = self.x[1] - self.x[0]
        
        #construct Laplacian
    def Laplacian(self):
        D=-np.eye(self.n_grid)+np.diagflat(np.ones(self.n_grid-1),1)
        #print (D)
        D = D / self.h
        D2=D.dot(-D.T)              #-D*D.T
        D2[-1,-1]=D2[0,0]
        self.D2 = D2 
        
        #external potential
    def Potential(self):
        biasL = self.bias*np.ones(10)
        biascenter = np.linspace(self.bias,0, self.n_grid-20)
        biasR = np.zeros(10)
        bias_cal = np.concatenate((biasL,biascenter, biasR))
            
        #no-well potential
        if self.potential_shape == 0:                   
            self.H = -self.D2/2 + np.diagflat(bias_cal) 
            pot = np.full_like(self.x,0)
            self.pot = (pot + bias_cal)
            
        #Harmonic potential
        elif self.potential_shape == 1:
            center = (self.x[-1]+self.x[0]) / 2
            pot =((self.x-center)**2)*self.ph
            self.pot = (pot + bias_cal)
            self.H = -self.D2/2 + np.diagflat(self.pot) 


        #well potential    
        elif self.potential_shape == 2:
            pot = np.full_like(self.x, self.ph)
            pot[np.logical_and(self.x > self.x[int(len(self.x)/2)]-(self.pw)/2, \
                               self.x < self.x[int(len(self.x)/2)]+(self.pw)/2)]=0.
            self.pot = (pot + bias_cal)
            self.H = -self.D2/2 + np.diagflat(self.pot)
                
        #double wall
        elif self.potential_shape == 3:
            pot = np.full_like(self.x, self.ph)
            pot[np.logical_or(self.x < self.x[int(len(self.x)/4)], \
                              self.x > self.x[int(len(self.x)*3/4)])] = 0.
            pot[np.logical_and(self.x > self.x[int(len(self.x)/4)]+self.pw, \
                               self.x < self.x[int(len(self.x)*3/4)]-self.pw)] = 0.
            self.pot = (pot + bias_cal)
            self.H = -self.D2/2 + np.diagflat(self.pot)
#           startboundary = np.where(pot==ph/hartree2eV)[0]

        #single wall potential
        elif self.potential_shape == 4:
            pot = np.full_like(self.x, 0)
            pot[np.logical_and(self.x > self.x[int(len(self.x)/2)]-(self.pw)/2, \
                               self.x < self.x[int(len(self.x)/2)]+(self.pw)/2)]=self.ph 
            self.pot = (pot + bias_cal)
            self.H = -self.D2/2 + np.diagflat(self.pot)

        return self.H 



#DFT calculation class
class DFT(Hamiltonian):
    def __init__(self, n_grid, xi, xf, epsil, effective_mass, pw, ph, pot_type, bias, \
                 max_iter, num_electron, itercrit, scaled, \
                 mix, energy_tolerance, occupation_scheme):
        #Define initial
        Hamiltonian.__init__(self, n_grid, xi, xf, epsil, effective_mass, pw, ph, pot_type, bias)
        Hamiltonian.Laplacian(self)
        Hamiltonian.Potential(self)
        self.num_electron = num_electron; self.max_iter = max_iter; self.itercrit = itercrit
        self.scaled = scaled; self.mix = mix; self.energy_tolerance = energy_tolerance 
        self.occupation_scheme = occupation_scheme
        
        #effectiv mass atomic units
        self.eu2ang = au2ang * self.ep / self.em
        self.eu2eV = au2eV * self.em / self.ep
        self.ang2eu = 1 / self.eu2ang
        self.eV2eu = 1 / self.eu2eV
        self.au_length2eu = self.em/self.ep; self.au_energy2eu = self.ep/self.em
        
        self.eu2au_length = 1 / self.au_length2eu; self.eu2au_energy = 1 / self.au_energy2eu

    def DFT_calculation(self):
        x = self.x
        H = self.H
        
        log={"energy":[float("inf")], "energy_diff":[float("inf")]}
        self.nx=np.zeros(self.n_grid) 
        #Start DFT calculation
        if self.occupation_scheme == 0: 
            for i in range(self.max_iter):
                #exchange & Hartree terms
                self.ex_energy, self.ex_potential = func.get_exchange(self.nx, x)
                self.ha_energy, self.ha_potential = func.get_hatree(self.nx, x)
                self.ex_energy = self.ex_energy * self.au_energy2eu ; self.ex_potential = self.ex_potential * self.au_energy2eu
                self.ha_energy = self.ha_energy * self.au_energy2eu ; self.ha_potential = self.ha_potential * self.au_energy2eu 
                
                if self.num_electron == 1:
                    self.ex_energy = 0; self.ex_potential = 0
                    self.ha_energy = 0; self.ha_potential = 0
                
                #get wavefunction and eigenenergy
                self.Hnew = H + np.diagflat(self.ex_potential + self.ha_potential)
                self.energy, self.psi= np.linalg.eigh(self.Hnew)

                #calculate total energy component
                eigenergy = func.geteigenenergy(self.nx, self.x, self.num_electron, self.energy)
                int_ex = func.ex_integral(self.nx, self.x, self.num_electron, self.ex_potential)
                totalE = func.cal_total_energy(self.ha_energy, self.ex_energy, eigenergy, int_ex)
                
                #converge criteria Total energy or ground eigenvalue
                if self.itercrit == 0:             #diff from eigenvalue
                    # log
                    log["energy"].append(self.energy[0])
                    energy_diff=self.energy[0]-log["energy"][-2]
                    energy_diff = energy_diff
                    log["energy_diff"].append(energy_diff)
                    func.print_log(i,log)
            
                elif self.itercrit == 1:           #diff from total energy
                    log["energy"].append(totalE)
                    energy_diff=totalE-log["energy"][-2]
                    energy_diff = energy_diff
                    log["energy_diff"].append(energy_diff)
                    func.print_log(i,log)
    
                # convergence check
                if abs(energy_diff) < self.energy_tolerance:
                    print("converged!")
                    break
            
                # update density
                if self.scaled == 0:
                    nx_before = self.nx
                    self.nx = func.get_nx(self.num_electron, self.psi,x)
                    self.nx = (self.mix)*self.nx+(1-self.mix)*nx_before
                elif self.scaled == 1:
                    nx_before = self.nx
                    self.nx = func.get_nx(self.num_electron, self.psi,x)
                    self.nx = (self.mix)*self.nx+(1-self.mix)*nx_before
                    self.nx = ((self.num_electron-1)/(self.num_electron)) * self.nx
            else:
                print("not converged")
                 
        #find bounded wavefunctions
        elif self.occupation_scheme == 1: 
            try: 
                for i in range(self.max_iter):
                    #exchange & Hartree terms
                    self.ex_energy, self.ex_potential = func.get_exchange(self.nx, x)
                    self.ha_energy, self.ha_potential = func.get_hatree(self.nx, x)
                    self.ex_energy = self.ex_energy * self.au_energy2eu ; self.ex_potential = self.ex_potential * self.au_energy2eu
                    self.ha_energy = self.ha_energy * self.au_energy2eu ; self.ha_potential = self.ha_potential * self.au_energy2eu 
                    
                    
                    if self.num_electron == 1:
                        self.ex_energy = 0; self.ex_potential = 0
                        self.ha_energy = 0; self.ha_potential = 0
            
                    #get wavefunction and eigenenergy
                    self.Hnew = H + np.diagflat(self.ex_potential + self.ha_potential)
                    self.energy, self.psi= np.linalg.eigh(self.Hnew)

                    #calculate total energy component
                    eigenergy = func.geteigenenergy(self.nx, self.x, self.num_electron, self.energy)
                    int_ex = func.ex_integral(self.nx, self.x, self.num_electron, self.ex_potential)
                    totalE = func.cal_total_energy(self.ha_energy, self.ex_energy, eigenergy, int_ex)
                    halfbound = int(len(np.where(self.pot == self.ph)[0])/2)
                    bound_left = np.where(self.pot == self.ph)[0][0]
                   # print ('ss', np.where(self.pot == self.ph)[0])
                    bound_right = np.where(self.pot == self.ph)[0][-1]
                    weight_c = np.zeros(len(x))
                    
                    for j in range(len(x)):
                        wc = np.trapz(self.psi[bound_left:bound_right,j]**2, x=x[bound_left:bound_right])
                        norm = np.trapz(self.psi[:,j]**2, x=x)
                            #print ("wc =", wc)
                        weight_c[j] = wc/norm
                        #print ("weigth_c=", weight_c)
           
                        ind_inner = np.where(weight_c > 0.5)[0]
                    
                        self.psi_inner = np.zeros((len(x),len(x)), dtype=float)
                    for k in range(len(ind_inner)):
                        a = ind_inner[k]
                        self.psi_inner[:,k] = self.psi[:,a]
                  #  plt.plot(self.psi_inner[:,1])
                    if self.itercrit == 0:             #diff from eigenvalue
                   # log
                        log["energy"].append(self.energy[0])
                        energy_diff=self.energy[0]-log["energy"][-2]
                        energy_diff = energy_diff
                        log["energy_diff"].append(energy_diff)
                        func.print_log(i,log)
            
                    elif self.itercrit == 1:           #diff from total energy
                        log["energy"].append(totalE)
                        energy_diff=totalE-log["energy"][-2]
                        energy_diff = energy_diff
                        log["energy_diff"].append(energy_diff)
                        func.print_log(i,log)
    
                    # convergence check
                    if abs(energy_diff) < self.energy_tolerance:
                        self.psi = self.psi_inner
                        print("converged!")
                        break
                    nx_before = self.nx
                    self.nx = func.get_nx(self.num_electron, self.psi_inner,x)
                    self.nx = (self.mix)*self.nx+(1-self.mix)*nx_before  
                else:
                    print("not converged")
            except:
                print ("error: change your potential or occupation shcheme!!!!")
        return self.nx, self.Hnew, self.psi, self.energy, self.ex_potential, self.ha_potential, self.pot

class transmission():
    def __init__(self, D, xi, xf, n_grid, H, NE, Emin, Emax, zplus,\
                 pot, ex, ha, epsil, effective_mass, efl, efr, T):
        self.NE = NE;  
        self.ep = epsil; self.em = effective_mass; self.H = H;  self.T = T
        self.n_grid = n_grid; 
        self.pot = pot ; self.ex = ex ; self.ha = ha 
        
        #effectiv mass atomic units
        self.eu2ang = au2ang * self.ep / self.em
        self.eu2eV = au2eV * self.em / self.ep
        self.ang2eu = 1 / self.eu2ang
        self.eV2eu = 1 / self.eu2eV
        self.au_length2eu = self.em/self.ep; self.au_energy2eu = self.ep/self.em
        self.eu2au_length = 1 / self.au_length2eu; self.eu2au_energy = 1 / self.au_energy2eu
        
        #scaled length
        self.xi = xi*self.ang2eu; self.xf = xf*self.ang2eu
        self.x = np.linspace(self.xi, self.xf, n_grid)

        self.xii = xi*ang2au; self.xff = xf*ang2au
        self.x_au = np.linspace(self.xii, self.xff, n_grid)
                
        self.h = self.x[1] - self.x[0]
        
        
        #scaled energy
        self.Emin = Emin * self.eV2eu; self.Emax = Emax * self.eV2eu
        #self.Emin = Emin ; self.Emax = Emax 
        self.efl = efl * self.eV2eu ; self.efr = efr * self.eV2eu;
        self.bias = (efl-efr)    
        self.D = D * self.eV2eu**2
        self.zplus = zplus * self.eV2eu;
        
        #current
        hbar = 1.06e-34; q=1.6e-19;self.IE = (q*q)/(2*np.pi*hbar)
        
    def Buttiker_transmission_cal(self):
        t0 = 1/(2*(self.h**2)) 
        E = np.linspace(self.Emin, self.Emax, self.NE); dE = E[1] - E[0]
        n_grid = self.n_grid; D = self.D
        #D = 1e-4*(t0**2)
        I_balli=0; I_buttiker=0
        transmatrix = []
        transmatrixB = []
        t_ballistic = np.zeros(self.NE)
        t_Butti_left = np.zeros(self.NE)
        t_Butti_right = np.zeros(self.NE)
        t_phase_relaxation = np.zeros(self.NE)
        for i in range(self.NE):
            sig1 = np.zeros((n_grid,n_grid), dtype = complex)
            sig2 = np.zeros((n_grid,n_grid), dtype = complex)
            sigB = np.zeros((n_grid,n_grid), dtype = complex)
            siginB1 = np.zeros((n_grid,n_grid), dtype = complex)
            
            ckl = 1 - (E[i] + self.zplus - self.pot[0])/(2*t0)
            kal = np.arccos(ckl)
            sig1[0][0] = -t0*np.exp(1j*kal) ; gam1 = 1j*(sig1-sig1.T.conj())
            ckr = 1 - (E[i] + self.zplus - self.pot[-1])/(2*t0)
            kar = np.arccos(ckr) 
            sig2[-1][-1] = -t0*np.exp(1j*kar); gam2 = 1j*(sig2-sig2.T.conj())
            G = np.linalg.inv((E[i] + self.zplus) * np.eye(n_grid) - self.H - sig1 - sig2)
            T_m = np.dot(gam2,G)
            T_m = np.dot(G.T.conj(), T_m)
            T_m = np.dot(gam1,T_m)
            T_mat = np.matrix(T_m)
            transmatrix.append(T_mat) 
            tt = float(T_mat.trace().real) ;  t_ballistic[i] = tt
            I_balli = I_balli + \
                self.IE*(dE * (self.eu2eV) * tt)*(func.Fermi(E[i] * self.eu2eV, self.efl * self.eu2eV, self.T) - func.Fermi(E[i] * self.eu2eV, self.efr * self.eu2eV, self.T))

            maxiter = 100; change = 100
            for j in range(maxiter):
                G_scatter = np.linalg.inv((E[i] + self.zplus)*np.eye(n_grid) - self.H - sig1 - sig2 - sigB)
                #sigBnew = np.diag(np.diag(np.dot(G, D)))  #for momentum relax
                sigBnew = np.dot(G_scatter, D)                     #for phase relax
                change = sum(sum(abs(sigBnew-sigB)))
                sigB = sigB + 0.25*(sigBnew-sigB)
                if abs(change) < 1e-6:
                    print("Self energy converged!")
                    break
            A = np.diag(np.real(1j*(G_scatter-G_scatter.T.conj())))
         #   M = D *(G * np.conj(G_scatter))
      
            # calculationg the inscattering function from the contacts F1, F2
            gam1 = 1j*(sig1-sig1.T.conj())
            gam2 = 1j*(sig2-sig2.T.conj())
            gamB = 1j*(sigBnew-sigBnew.T.conj())
          #  gam1 = 1j*(sig1-np.matrix(sig1).getH())
          #  gam2 = 1j*(sig2-np.matrix(sig2).getH())
          #  gamB = 1j*(sigB-np.matrix(sigB).getH())
          #  gamma = gam1 + gam2 + gamB
            sigin1 = func.Fermi(E[i] * self.eu2eV, self.efl * self.eu2eV, self.T) * gam1 
            sigin2 = func.Fermi(E[i] * self.eu2eV, self.efr * self.eu2eV, self.T) * gam2
          #  n = np.dot((np.linalg.inv( np.eye(n_grid)-M)) , np.diag(G_scatter*(sigin1 + sigin2)*np.conjugate(G_scatter)))

          #  siginB = D * np.diag(n)
            Gn = G_scatter * ( sigin1 + sigin2 + sigBnew ) * np.matrix(G_scatter).getH()
            
            if self.bias != 0:
                try:
            # calculating the current 
                #t_Butti_left[i] = (np.trace(np.dot(gam1,Gn))-np.trace(np.dot(sigin1,A))).real
                #t_Butti_right[i] = (np.trace(np.dot(gam2,Gn))-np.trace(np.dot(sigin2,A))).real
                    t_Butti_left[i] = -(1/(func.Fermi(E[i] * self.eu2eV, self.efl * self.eu2eV, self.T)-func.Fermi(E[i] * self.eu2eV, self.efr * self.eu2eV, self.T)))\
                        *(np.trace(np.dot(gam1,Gn))-np.trace(np.dot(sigin1,A))).real
                    t_Butti_right[i] = (1/(func.Fermi(E[i] * self.eu2eV, self.efl * self.eu2eV, self.T)-func.Fermi(E[i] * self.eu2eV, self.efr * self.eu2eV, self.T)))\
                        *(np.trace(np.dot(gam2,Gn))-np.trace(np.dot(sigin2,A))).real
                except:
                    t_Butti_left[i] = 0
                    t_Butti_right[i] = 0
  
            change1 = 100
            for k in range(maxiter):
                Gn_sigB_gamma1 = np.dot(gam1+siginB1, G_scatter.T.conj())
                Gn_sigB_gamma1 = np.dot(G_scatter,Gn_sigB_gamma1)
            
               # siginBnew = np.diag(np.diag(Gn_sigB_gamma1*D))
                siginBnew = np.dot(Gn_sigB_gamma1,D)
                change1 = sum(sum(abs(siginBnew-siginB1)))
                #print ("change1= ", change1)
                siginB1 = siginB1 + 0.25*(siginBnew-siginB1)
                if abs(change1).all() < 1e-6:
                    print("self energy converged!")
                    break
            T_mB = np.dot(gam2,G_scatter)
            T_mB = np.dot(G_scatter.T.conj(), T_mB)
            T_mB = np.dot(gam1,T_mB)
            
            T_matB = np.matrix(T_mB)
            transmatrixB.append(T_matB) 
            t_phase_relaxation[i] = float(T_matB.trace().real)
            ttt = float(T_matB.trace().real) ;  t_phase_relaxation[i] = ttt
            I_buttiker = I_buttiker + self.IE*(dE* (self.eu2eV) *ttt)*(func.Fermi(E[i] * self.eu2eV, self.efl * self.eu2eV, self.T) -\
                                                func.Fermi(E[i] * self.eu2eV, self.efr * self.eu2eV, self.T))
            F = np.diag(np.real(Gn_sigB_gamma1))/A
       
        return self.x, E, t_Butti_left, t_ballistic, t_phase_relaxation, F, I_balli, I_buttiker    
        