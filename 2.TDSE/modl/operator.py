import numpy as np
import numpy.linalg as lin
import grid


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
        self.oprt = np.array(self.oprt, dtype = np.float128)

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

class Potential(Operator):
    def __init__(self, L, n):
        Operator.__init__(self, n)
        self.vectorV = grid.Potential(L, n)
        for i in range(n):
            self.oprt[i][i] = self.vectorV.grd[i]


# Construct Hamiltonian using Potential and Laplacian
class Hamiltonian(Operator):
    def __init__(self, L, n, l, dx):
        Operator.__init__(self, n)
        self.V = Potential(L, n)
        self.L = Laplacian(n, l, dx)                   # h- = 1, m_e = 1
        self.oprt = -self.L.oprt / 2. + self.V.oprt    # H = - (h-^2/2m) L + V

# Construct Time Operator using Hamiltonian
class time_operator(Operator):
    def __init__(self, L, n, l, dt, dx):
        Operator.__init__(self, n)
        self.oprt        = np.array(self.oprt, dtype = np.complex256)
        self.H           = Hamiltonian(L, n, l, dx)
        self.exp_iHdt_2  = np.eye(n) + 1.j * self.H.oprt * dt / 2.
        self.exp_iHdt_2  = np.array(self.exp_iHdt_2, dtype = np.complex128)
        self.exp_iHdt_2  = lin.inv(self.exp_iHdt_2)
        self.exp_miHdt_2 = np.zeros_like(self.oprt)
        self.exp_miHdt_2 = np.eye(n) - 1.j * self.H.oprt * dt / 2.
        self.oprt        = np.dot(self.exp_iHdt_2, self.exp_miHdt_2)
#               PSI(x,t) = e^-iHt PSI(x,0)
#   e^iHdt/2 PSI(x,t+dt) = e^-iHdt/2 PSI(x,t)
# (1+iHdt/2) PSI(x,t+dt) = (1-iHdt/2) PSI(x,t)
#            PSI(x,t+dt) = (1+iHdt/2)^-1 (a-iHdt/2) PSI(x,t)
