# Input parameter
Lx = 100                # Box Length (Angstrom = 10^-10 m)
ngx = 1001              # Number of grid points (spacing = L/(n-1))
nstep = 30              # Number of time steps * 0.1fs
ewave = 10              # Energy of packet (eV)
pot_shape = 6           # Shape of potential
                        # 0 = No potential
                        # 1 = Step potential
                        # 2 = Single wall
                        # 3 = Double wall
                        # 4 = Finite well (Packet starts at middle)
                        # 5 = Asymmetric potential barrier
                        # 6 = Triangular potential barrier (E-field)
pot_height_eV = 10      # eV
barrier_thickness = 10  # Thickness of the barrier(Angstrom = 10^-10 m)
                        # Only for Potential_shape = 2 or 3!

dispersion_gaussian = 10 # Spatial dispersion(sigma) of gaussian wave packet (Angstrom = 10^-10 m)
