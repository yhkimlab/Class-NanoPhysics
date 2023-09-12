#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 12:59:02 2020

@author: ramanujan
"""

""""""""""""""""""""""""""""""""""""
"            Input                 "
""""""""""""""""""""""""""""""""""""
n_grid = 200                    # #of x grid
xi = 0                          #initial x coordinate
xf = 20                         #final x coordinate
num_electron = 2
pot_type = 3                    # 0 : infinite well
                                # 1 : Harmonic
                                # 2 : well potential
                                # 3 : double-wall potential
                                # 4 : single-wall potential
bias = 0
ph = 10                           #potential height
pw = 1                           #potential width
epsil =  1
effective_mass = 1
max_iter = 1000
energy_tolerance = 1e-5
NE = 200                         #energy range
efl = 0
efr = 0
T = 300
Emin = 0
Emax = 12
zplus = 1e-7j
D = 1e-1                       #scattering sctrength, eV**2


""""""""""""""""""""""""""""""""""""
"              Options             "
""""""""""""""""""""""""""""""""""""                    
itercrit = 1                    #0 eigenvalue, 1 total energy
scaled = 0                      #0 LDA, 1 scaled potential
mix = 0.01                      #mixing weight
occupation_scheme = 0          #0 normal, 1 only bound states



