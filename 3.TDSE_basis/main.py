import numpy as np
from modl import grd
from modl import inp
import numpy.linalg as lin
from modl import operation
from matplotlib import pyplot as plt

#Inputs & Unit conversion

atomic_unit_of_time_to_fs = 2.418884326509E-2
L=inp.L * 1.88973
l = 3   # Number of FDM points (2l+1) Here, 7-points   
nt=inp.nstep
dt = 0.1/atomic_unit_of_time_to_fs
pot = inp.Potential_Shape
pot_height = inp.Potential_Height
n = inp.n
dx=L/n

# Construct Hamiltonian using Potential and Laplacian
# Get Eigenvalue & vectors of the hamiltonian

H = operation.Hamiltonian(L,n,l,dx)
E,phi = np.linalg.eigh(H)

# Make wave fucntion with basis (also construct it's operator)

wave = grd.waveF(L, inp.n, l)

c_n =(wave.grd).dot(np.conjugate(phi))
Psi = c_n.dot(phi)
tt = np.linspace(0, (nt-1)*dt,nt)
z=np.zeros((n,n), dtype=complex)
Psi_t = np.zeros((nt,n), dtype =complex)
xx=np.linspace(0, L, n)


f= open("wave.txt", 'w')
f.write("# t(fs) " )
for i in range(0, n):
    f.write('  %.6f  '%xx[i])
f.write('\n')

poten = grd.Potential(L, inp.n)

saveprob=[]
for i in range(nt):
    for j in range(n):
        z[:,j] = c_n[j]*(phi[:,j]*np.exp(-1j*E[j]*(tt[i])))
    Psi_t[i, :] = np.sum(z, 1)
    reflec = np.float64(0)
    trans = np.float64(0)

#counting
    prob = np.abs(Psi_t[i,:])**2
    for k in range(0, n):
            if k <= poten.left :
                reflec += prob[k]
            if k > poten.right :
                trans += prob[k]

# Save wave function as txt
    t=i*dt*atomic_unit_of_time_to_fs
    f.write('%.6f' %t)
    for k in range(0, n):
        f.write('  %.8f  ' %prob[k])
    f.write('\n')
    saveprob.append('%.6f    %.6f     %.6f\n' %(t,reflec,trans))

f2 = open("Potential.txt",'w')
f2.write(grd.Potential(L, inp.n).plot_grid())
f2.close()

f3 = open("Probablity.txt",'w')
f3.write("# t(fs)    Reflection  Transmission\n")
f3.writelines(saveprob)
f3.close()
