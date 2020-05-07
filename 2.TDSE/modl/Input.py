L = 100                 # Box Length (Angstrom = 10^-10 m)
n = 1001                # Number of grid points (spacing = L/(n-1))
nstep = 30              # Number of time steps * 0.1fs
E = 10                  # Energy of packet (eV)
Potential_Shape = 2     # Shape of potential
                        # 0 = No potential
                        # 1 = Step potential
                        # 2 = Single wall
                        # 3 = Double wall
                        # 4 = Finite well (Packet starts at middle)
Potential_Height = 20   # eV
Barrier_Thickness = 10  # Thickness of the barrier(Angstrom = 10^-10 m)
                        # Only for Potential_shape = 2 or 3!
