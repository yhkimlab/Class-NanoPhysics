import numpy as np
import numpy.linalg as lin
import inp
import operator as oprt

#parameter  
pot_height = inp.Potential_Height
# Grid class is real space,, and mold for inheritance.
class grid:
    def __init__(self, L, n):
        self.n   = n
        self.dx  = L/n
        self.grd = np.linspace(-L/2., L/2., num = n, endpoint=False) # Make grid
        self.grd = np.array(self.grd, dtype = np.float64)

# Return string grid data (probability density for waveFunction)
    def plot_grid(self):
        self.strline = ""
        if "float" in str(self.grd.dtype):
            self.tmp_grd = self.grd
        if "complex" in str(self.grd.dtype):
            self.tmp_grd = self.grd.real**2 + self.grd.imag**2
        for i in range(self.n):
            self.strline += "  %.5e " % self.tmp_grd[i]
        self.strline += "  %.6e  \n" % self.tmp_grd[0]
        return self.strline

# Set the potential
class Potential(grid):
    def __init__(self, L, n):
        grid.__init__(self, L, n)
        self.grd[0] = 100000000
        self.grd[1] = 100000000
        self.grd[n-1]= 100000000
        for i in range(2, n-2):
            self.grd[i] = 0                           # Make potential
        for i in range((100*n)//200,(105*n)//200):
            self.grd[i] = pot_height                   # eV unit
            self.grd[i] = self.grd[i]/27.211          # eV -> Har 
        for i in range((110*n)//200,(115*n)//200):
            self.grd[i] = pot_height                   # eV unit
            self.grd[i] = self.grd[i]/27.211          # eV -> Har 

