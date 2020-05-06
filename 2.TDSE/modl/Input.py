L = 2000              # Box Length (Angstrom = 10^-10 m)
n = 3001                 # Number of grid points (spacing = L/(n-1))
nstep = 50              # Number of time steps * 0.1fs
E = 1                   # Energy of packet (eV)
Potential_Shape = 2     # Shape of potential
                        # 0 = No potential
                        # 1 = Step potential
                        # 2 = Single wall
                        # 3 = Double wall
                        # 4 = Finite well (Packet starts at middle)
Potential_Height = 10   # eV
Barrier_Thickness = 10  # Thickness of the barrier(Angstrom = 10^-10 m)
                        # This is the input for only Potential_shape= 2 or 3!
