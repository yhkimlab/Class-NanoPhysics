import numpy as np
import numpy.linalg as lin
from modl import Input

#parameter  
pot_height_har = Input.pot_height_eV/27.211
pot_shape = Input.pot_shape


# Grid class is real space,, and mold for inheritance.
class grid:
    def __init__(self, L, n):
        self.n   = n
        self.dx  = L/n
        self.grd = np.linspace(-L/2., L/2., num = n, endpoint=False) # Make grid
        self.grd = np.array(self.grd, dtype = np.float64)

# Define FDM points & coefficients (Here, 7-points FDM)
def fdmcoefficient(l):
    a=np.zeros((2*l+1,2*l+1))
    b=np.zeros(2*l+1)
    c=np.zeros(2*l+1)

    for i in range(0, 2*l+1):
        for j in range(0, 2*l+1):
            a[i,j]= (j-l)**i
    c[2]=2
    a = lin.inv(a)
    b= np.matmul(a, c)

    C=np.zeros((l+1))

    for i in range(0, l+1):
        C[i]=b[i+l]
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
        
        if pot_shape == 1:                               # Harmonic
           for i in range(1, 1000):
               x=L/(n-1)*i
               self.grd[i]=(i-n//2)**2/(n//2-1)**2*pot_height_har       

        if pot_shape == 2:                               # Square well 
           for i in range(1,(4*n)//10):
               self.grd[i] = pot_height_har
           for i in range((4*n)//10,(6*n)//10+1):
               self.grd[i] = 0
           for i in range((6*n)//10+1,n-1):
               self.grd[i] = pot_height_har
        
        if pot_shape == 3:                               #Triangular
           self.grd[:]=10**6
           for i in range((5*n)//10,n):
               self.grd[i] = pot_height_har*abs(i-500)/500

        if pot_shape == 4:                                   #Asymmetric infinite Well
           self.left = 0.5*n
           self.right = n*20//40
           for i in range((5*n//10), n-2):
               self.grd[i] = pot_height_har                 # eV unit

        if pot_shape == 5:                                   #Double barrier
           self.left = 0.5*n
           self.right = n*20//40
           for i in range(2, n-2):
               self.grd[i] = 0                           # Make potential
           for i in range(400,412):
               self.grd[i] = pot_height_har                   # eV unit
           for i in range(427,439):
               self.grd[i] = pot_height_har                   # eV unit

        self.oprt=np.zeros((n,n))
        for i in range(0, n):
            self.oprt[i, i]=self.grd[i]
