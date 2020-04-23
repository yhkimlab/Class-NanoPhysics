L = 100                 # Box Length (Angstrom = 10^-10 m)
n = 501                 # Number of grid points (spacing = L/(n-1))
nstep = 50              # Number of time steps * 0.1fs
E = 20                  # Energy of packet (eV)
Potential_Shape = 1     # Shape of potential
                        # 0 = No potential
                        # 1 = Step potential
                        # 2 = Single wall
                        # 3 = Double wall
                        # 4 = Finite well (Packet starts at middle)
Potential_Height = 50   # eV
