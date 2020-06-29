import numpy as np
Lx = 100                # Box Length (Angstrom = 10^-10 m)
ngx = 1001              # Number of grid points (spacing = L/(n-1))
nstep = 100             # Number of time steps * 0.1fs
ewave = 80              # Energy of packet (eV)
pot_shape = 5           # Shape of potential
                        # 0 = No potential
                        # 1 = Step potential
                        # 2 = Single wall
                        # 3 = Double wall
                        # 4 = Finite well (Packet starts at middle)
                        # 5 = Harmonic well

pot_height_eV = 78.02   # eV
barrier_thickness = 10  # Thickness of the barrier(Angstrom = 10^-10 m)
                        # Only for Potential_shape = 2 or 3!

dispersion_gaussian = 2 # Spatial dispersion(sigma) of gaussian wave packet (Angstrom = 10^-10 m)

lpacket = 2             # 0: For wavepacket tunneling,
                        # 1: For oscillating bound state with linear combination of eigenstates
                        # 2: For coherent state
ncombistates = 7        # number of eigenstates in linear combination 
