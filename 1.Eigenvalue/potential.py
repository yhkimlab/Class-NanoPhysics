import numpy as np
import numpy.linalg as lin
import inp
import operator as oprt

#parameter  
pot_height = inp.Potential_Height/27.211
pot = inp.Potential_Shape


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
ost=[]

# Set the potential
class Potential(grid):
    def __init__(self, L, n):
        grid.__init__(self, L, n)
        self.grd[:]=0
        
        self.grd[0] = 1000000000
        self.grd[n-1]=1000000000
        

        if pot == 0:                               #Double barrier
           for i in range(100, 201):
               self.grd[i] = pot_height            
           for i in range(300, 1000):
               self.grd[i] = pot_height            # eV unit
        
        if pot == 1:                               # Harmonic
           for i in range(0, 1001):
               x=L/(n-1)*i
               self.grd[i] = 1/2*(x-L/(n-1)*500)**2/2500*5.73436/1.88973**2*pot_height/10*27.211
        
        if pot == 2:                               # Square well 
           for i in range(400,600):
               self.grd[i]=-pot_height
        
        
        if pot == 3:                               #Triangular
           self.grd[:]=10**6
           for i in range(500,1001):
               self.grd[i]=pot_height*abs(i-500)/200

