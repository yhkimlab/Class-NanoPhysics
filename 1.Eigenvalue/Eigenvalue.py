import numpy as np
import numpy.linalg as lin
import operator as oprt
import inp
from matplotlib import pyplot as plt
import potential

#Inputs & Unit conversion
pot_height = inp.Potential_Height
pot = inp.Potential_Shape
n=inp.n
l=3
L=inp.L*1.88973
dx=L/(n)
numstate = inp.num

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

# Define Potential for Hamiltonian 
class Potential(Operator):
    def __init__(self, L, n):
        Operator.__init__(self, n)
        self.vectorV = potential.Potential(L, n)
        for i in range(n):
            self.oprt[i][i] = self.vectorV.grd[i]

# Define Hamiltonian using Potential and Laplacian
def Hamiltonian(L,n,l,dx):
    oprt=np.zeros((n,n))
    V = Potential(L, n)
    L = Laplacian(n, l, dx)                   # h- = 1, m_e = 1
    oprt = -L.oprt / 2. + V.oprt    # H = - (h-^2/2m) L + V
    return oprt

# Construct Hamiltonian using Potential and Laplacian
H=np.zeros((n, n))
dx=L/(n-1)
dn=np.ones(n)/dx**2
dno=np.ones(n-1)*(-1/2)/dx**2

V = potential.Potential(L,n)
H = Hamiltonian(L, n, l, dx)

#Diagonalize Hamiltonian to get Eigenvalues
(w,v) = np.linalg.eigh(H)
num = np.argsort(w)

#Make box & Visualize
a = np.linspace(-L/2, L/2, n)


for i in range(0,numstate):
    print ('%dth state eigenvalue is %f eV' %(i,w[num[i]]*27.211))
    plt.clf()
    if pot == 0: 
        plt.plot(a,v[:,num[i]])
    else:
        plt.plot( a, V.grd*np.max(np.abs(v[:,num[i]])*0.9/np.max(np.abs(V.grd[2:n-2]))),a,v[:,num[i]])
    plt.ylim((-np.max(np.abs(v[:,num[i]])),np.max(np.abs(v[:,num[i]]))))
    plt.savefig('%04d' %i)
