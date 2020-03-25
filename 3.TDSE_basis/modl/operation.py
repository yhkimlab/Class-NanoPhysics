import numpy as np
import numpy.linalg as lin
from modl import grd
from modl import inp

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
        self.vectorV = grd.Potential(L, n)
        for i in range(n):
            self.oprt[i][i] = self.vectorV.grd[i]


# Construct Hamiltonian using Potential and Laplacian

def Hamiltonian(L,n,l,dx):
    oprt=np.zeros((n,n))
    V = Poten(L, n)
    L = Laplacian(n, l, dx)                   # h- = 1, m_e = 1
    oprt = -L.oprt / 2. + V.oprt    # H = - (h-^2/2m) L + V
    return oprt


