import numpy as np
import numpy.linalg as lin
from matplotlib import pyplot as plt
from modl import operator, Input

#Inputs & Unit conversion
pot_height = Input.Potential_Height/27.211
n=Input.n
l=3
L=Input.L*1.88973
dx=L/(n)
numstate = Input.num

# Define Hamiltonian using Potential and Laplacian
def Hamiltonian(L,n,l,dx):
    Hamiltonian = np.zeros((n,n))
    V = operator.Potential(L, n)
    L = operator.Laplacian(n, l, dx)                   # h- = 1, m_e = 1
    Hamiltonian = -L.oprt / 2. + V.oprt    # H = - (h-^2/2m) L + V
    return Hamiltonian

# Construct Hamiltonian using Potential and Laplacian
H=np.zeros((n, n))
dx=L/(n-1)
dn=np.ones(n)/dx**2
dno=np.ones(n-1)*(-1/2)/dx**2

V = operator.Potential(L,n)
H = Hamiltonian(L, n, l, dx)

#Diagonalize Hamiltonian to get Eigenvalues
(w,v) = np.linalg.eigh(H)
num = np.argsort(w)

#Make box & Visualize
a = np.linspace(-L/2, L/2, n)

#Making Total figure
for i in range(0,numstate):
    bb=pot_height*27.211/20
    bbb=np.max(v[:,num[i]])-np.min(v[:,num[i]])
    
    plt.plot(a,v[:,num[i]]/bbb*bb+np.ones(n)*w[i]*27.211)

plt.plot(a,V.grd*27.211)
plt.ylim(0,Input.Potential_Height)
plt.ylabel('Energy [eV]')
plt.xlabel('Box [Angstrom]')
plt.savefig('Total.png')
plt.show()

#Making figure for individual eigenstate
for i in range(0,numstate):
    print ('%dth state eigenvalue is %f eV' %(i+1,w[num[i]]*27.211))
    plt.clf()
    plt.plot( a,v[:,num[i]], label='%dth eigenvalue = %f eV' %(i,w[num[i]]*27.211))
    plt.legend()
    plt.ylim((-np.max(np.abs(v[:,num[i]])),np.max(np.abs(v[:,num[i]]))))
    plt.savefig('%04d' %(i+1))
