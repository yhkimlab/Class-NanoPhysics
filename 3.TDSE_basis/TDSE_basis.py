import numpy as np
import numpy.linalg as lin
from modl import operator, Input
from matplotlib import pyplot as plt
import math

#Inputs & Unit conversion

atomic_unit_of_time_to_fs = 2.418884326509E-2
Lx = Input.Lx * 1.88973
l = 3   # Number of FDM points (2l+1) Here, 7-points   
nt=Input.nstep
dt = 0.1/atomic_unit_of_time_to_fs
pot_shape = Input.pot_shape      
pot_height_eV = Input.pot_height_eV   
ngx = Input.ngx
dx=Lx/ngx

# Construct Hamiltonian using Potential and Laplacian
# Get Eigenvalue & vectors of the hamiltonian

H = operator.Hamiltonian(Lx,ngx,l,dx)
E,phi = np.linalg.eigh(H)

# Make wave fucntion with basis (also construct it's operator)

wave = operator.wave(Lx, ngx, l)

c_n =(wave.grd).dot(np.conjugate(phi))
Psi = c_n.dot(phi)
tt = np.linspace(0, (nt-1)*dt,nt)
z=np.zeros((ngx,ngx), dtype=complex)
Psi_t = np.zeros((nt,ngx), dtype =complex)
xx=np.linspace(0, Lx, ngx)


f= open("wave.txt", 'w')
f.write("# t(fs) " )
for i in range(0, ngx):
    f.write('  %.6f  '%xx[i])
f.write('\n')

poten = operator.Potential(Lx, ngx)

saveprob=[]



####Linear combination for Bound state###

bounce=np.zeros(ngx, complex)

for i in range(nt):
    if(Input.lpacket == 0):
        for j in range(ngx):
            z[:,j] = c_n[j]*(phi[:,j]*np.exp(-1j*E[j]*(tt[i])))
        Psi_t[i,:] = np.sum(z,1)
    if(Input.lpacket == 1):
        bounce=0
        for j in range(0, Input.ncombistates):
            bounce += phi[:,j]*np.exp(-1j*E[j]*(tt[i]))
        a=np.sqrt(np.sum(bounce*np.conjugate(bounce)))*dx
        Psi_t[i,:]=bounce/a/5

    if(Input.lpacket == 2):
        bounce=0
        for j in range(0, Input.ncombistates):
            z[:,j] = (1/np.sqrt(math.factorial(j+1)))*(phi[:,j]*np.exp(-1j*E[j]*(tt[i])))            
        Psi_t[i,:] = np.sum(z,1)


    reflec = np.float64(0)
    trans = np.float64(0)

#counting
    prob = np.abs(Psi_t[i,:])**2
    for k in range(0, ngx):
            if k <= poten.left :
                reflec += prob[k]
            if k > poten.right :
                trans += prob[k]

# Save wave function as txt
    t=i*dt*atomic_unit_of_time_to_fs
    f.write('%.6f' %t)
    for k in range(0, ngx):
        f.write('  %.8f  ' %prob[k])
    f.write('\n')
    saveprob.append('%.6f    %.6f     %.6f\n' %(t,reflec,trans))

f2 = open("Potential.txt",'w')
f2.write(operator.Potential(Lx,ngx).plot_grid())
f2.close()

f3 = open("Probablity.txt",'w')
f3.write("# t(fs)    Reflection  Transmission\n")
f3.writelines(saveprob)
f3.close()
