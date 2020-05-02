import numpy as np
import numpy.linalg as lin
from modl import Input

#parameter  
pot_height = Input.Potential_Height/27.211
pot = Input.Potential_Shape


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

# Define FDM points & coefficients (Here, 7-points FDM)
def fdmcoefficient(l):
    A = np.zeros((l,l))                # for coefficient of h^2m -> 0, m = [0, 2~l)
    C = np.zeros((l+1))                # df/dx^2 = sum C[m] f(a+mh) / h^2

    for j in range(l):
        for i in range(l):
            A[i][j] = (j+1)**(2*(i+1)) # A_i,j = j^2i (for i,j = [1,l])

    A = lin.inv(A)

    for i in range(l):
        C[i+1] = A[i,0]                # C = A^-1 [1 0 ... 0]^T

    for i in range(1,l+1):
        C[0] += -2.*C[i]               # C[0] = -2 * sum[ C[1~l] ]

    return C

# Mold for inheritance.
class Operator:
    def __init__(self, n):
        self.oprt = np.zeros((n, n))
        self.oprt = np.array(self.oprt, dtype = np.float64)

    def oprtprint(self):
        print (self)                       # print Name
        print (self.oprt)                  # print Matrix
        print (self.oprt.dtype)            # print Type

# Define Laplacian for Hamiltonian
class Laplacian(Operator):
    def __init__(self, n, l, dx):
        Operator.__init__(self, n)
        self.C = fdmcoefficient(l)    # FDM coefficient
        for i in range(n):
            for j in range(-l, l + 1, 1):
                k = i + j
                if (k >= n):
                    k -= n
                    self.oprt[i][k] = self.C[abs(j)] / (dx**2)
                elif (k<0):
                    k += n
                    self.oprt[i][k] = self.C[abs(j)] / (dx**2)
                else:
                    self.oprt[i][k] = self.C[abs(j)] / (dx**2)


# Set the potential
class Potential(grid):
    def __init__(self, L, n):
        grid.__init__(self, L, n)
        self.grd[:]=0
        
        self.grd[0] = 1000000000
        self.grd[n-1]=1000000000
        

        if pot == 1:                               # Harmonic
           for i in range(1, 1000):
               x=L/(n-1)*i
               self.grd[i]=(i-n//2)**2/(n//2-1)**2*pot_height       
        if pot == 2:                               # Square well 
           for i in range(0,400):
               self.grd[i] = pot_height
           for i in range(400,601):
               self.grd[i] = 0
           for i in range(601,n-2):
               self.grd[i] = pot_height
        
        if pot == 3:                               #Triangular
           self.grd[:]=10**6
           for i in range(500,1001):
               self.grd[i] = pot_height*abs(i-500)/200

        if pot == 4:                                   #Double barrier
           self.left = 0.5*n
           self.right = n*20//40
           for i in range(2, n-2):
               self.grd[i] = 0                           # Make potential
           for i in range(400,412):
               self.grd[i] = pot_height                   # eV unit
               self.grd[i] = self.grd[i]/27.211          # eV -> Har
           for i in range(427,439):
               self.grd[i] = pot_height                   # eV unit
               self.grd[i] = self.grd[i]/27.211          # eV -> Har

        self.oprt=np.zeros((n,n))
        for i in range(0, n):
            self.oprt[i, i]=self.grd[i]
