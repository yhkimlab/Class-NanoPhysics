import numpy as np
import numpy.linalg as lin
from modl import Input

# Inputs & Unit conversion
k = (Input.E*2/27.211)**0.5
pot = Input.Potential_Shape
pot_height = Input.Potential_Height
thickness = Input.Barrier_Thickness

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

# wave function : complex grid, construct a time operator
class wave(grid):
    def __init__(self, L, n, l):
        grid.__init__(self, L, n)
        self.grd = np.array(self.grd, dtype = np.complex64)
        self.integral = 0.
        self.dxx=np.float64(L/n)

        if pot == 4:
            for i in range(n):                                   # wave packet initialize (gaussian)
                if (i > n*4/10 and i < n*6/10):
                    self.grd[i] = np.exp(-(i*self.dxx-0.5*n*self.dxx)**2/10)
                else:
                    self.grd[i] = 0. + 0.j
            self.grd /= lin.norm(self.grd)                       # Fix to normalize
            for i in range(n):
                self.grd[i] = self.grd[i]*np.exp(1j*k*(i*self.dxx-0.5*n*self.dxx))  #Wave packet

        else:
            for i in range(n):                                   # wave packet initialize (gaussian)
                if (i > n*0/10 and i < n*4/10):
                    self.grd[i] = np.exp(-(i*self.dxx-0.3*n*self.dxx)**2/10)
                else:
                    self.grd[i] = 0. + 0.j
            self.grd /= lin.norm(self.grd)                       # Fix to normalize
            for i in range(n):
                self.grd[i] = self.grd[i]*np.exp(1j*k*(i*self.dxx-0.3*n*self.dxx))  #Wave packet

# Set the potential
class Potential(grid):
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

    def __init__(self, L, n):
        grid.__init__(self, L, n)
        self.grd[0] = 100000000
        self.grd[1] = 100000000
        self.grd[n-1]= 100000000
        if pot == 0:                                   #no potential
           self.left = 0.5*n
           self.right = 0.5*n
           for i in range(1, n):
               self.grd[i] = 0                           # Make potential

        if pot == 1:                                   #Step potential
           self.left = 0.5*n
           self.right = 0.5*n
           for i in range(2, n-2):
               self.grd[i] = 0                           # Make potential
           for i in range((n//2),(n-2)):
               self.grd[i] = pot_height                   # eV unit
               self.grd[i] = self.grd[i]/27.211          # eV -> Har

        if pot == 2:                                   #Potential barrier
           self.left = 0.5*n
           self.right = 0.5*n
           for i in range(2, n-2):
               self.grd[i] = 0                           # Make potential
           for i in range((5*n)//10,(5*n)//10+thickness*10):
               self.grd[i] = pot_height                   # eV unit
               self.grd[i] = self.grd[i]/27.211          # eV -> Har

        if pot == 3:                                   #Double barrier
            self.left = (45*n)//100-thickness*10
            self.right = (50*n)//100+thickness*10
            for i in range(2, n-2):
                self.grd[i] = 0                           # Make potential
            for i in range((45*n)//100-thickness*10,(45*n)//100):
                self.grd[i] = pot_height                   # eV unit
                self.grd[i] = self.grd[i]/27.211          # eV -> Har
            for i in range((50*n)//100,(50*n)//100+thickness*10):
                self.grd[i] = pot_height                   # eV unit
                self.grd[i] = self.grd[i]/27.211          # eV -> Har

        if pot == 4:                                   # Square well
            self.left = (n*4)//10
            self.right = (n*6)//10
            for i in range(2, n-2):
                self.grd[i] = 0                           # Make potential
            for i in range(2,(n*4)//10):
                self.grd[i]= pot_height
                self.grd[i] = self.grd[i]/27.211          # eV -> Har
            for i in range((n*6)//10,n-2):
                self.grd[i]= pot_height
                self.grd[i] = self.grd[i]/27.211          # eV -> Har

        if pot == 5:                                   #Double barrier(Resonant)
            self.left = 0.6*n
            self.right = 1213
            for i in range(2, n-2):
                self.grd[i] = 0                           # Make potential
            for i in range(1200, 1204):
                self.grd[i] = pot_height                   # eV unit
                self.grd[i] = self.grd[i]/27.211          # eV -> Har
            for i in range(1209,1213):
                self.grd[i] = pot_height                   # eV unit
                self.grd[i] = self.grd[i]/27.211          # eV -> Har

        self.oprt = np.zeros((n,n))
        for i in range(0, n):
            self.oprt[i, i]=self.grd[i]



# Construct FDM coefficient

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

# Construct Laplacian using FDM Coefficient
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

# Construct Potential Operator using Potential grid
# Potential considering inp.py input

class Poten(Operator):
    def __init__(self, L, n):
        Operator.__init__(self, n)
        self.vectorV = Potential(L, n)
        for i in range(n):
            self.oprt[i][i] = self.vectorV.grd[i]


# Construct Hamiltonian using Potential and Laplacian

def Hamiltonian(L,n,l,dx):
    oprt=np.zeros((n,n))
    V = Poten(L, n)
    L = Laplacian(n, l, dx)                   # h- = 1, m_e = 1
    oprt = -L.oprt / 2. + V.oprt    # H = - (h-^2/2m) L + V
    return oprt


