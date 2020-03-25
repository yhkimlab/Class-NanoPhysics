import numpy as np
from modl import grid, inp


#Inputs & Unit conversion

atomic_unit_of_time_to_fs = 2.418884326509E-2
angsL = inp.L * 1.88973
l = 3
nstep = inp.nstep


# Make wave fucntion (also construct it's operator)
wave = grid.waveF(angsL, inp.n, l, inp.dt)
# Save wave function as txt
f = open("wave.txt",'w')
f.write("# t(fs) " + grid.grid(angsL, inp.n).plot_grid())
f.write("  0.0000 " + wave.plot_grid())
reflec = np.float64(0)
trans = np.float64(0)
poten = grid.Potential(angsL, inp.n)


inten=np.zeros(inp.n)
saveprob = []
# Time propagation
for i in range(nstep):
    for j in range(nstep):
        wave.next_step()
        reflec = 0
        trans = 0 
        #counting
        prob = np.abs(wave.grd)**2
        for k in range(0, inp.n):
            if k <= poten.left :
                reflec += prob[k]
            if k > poten.right :
                trans += prob[k]
    t = (i+1) * j * inp.dt * atomic_unit_of_time_to_fs
    f.write("%.6f" % t + wave.plot_grid())
    saveprob.append('%.6f    %.6f     %.6f\n' %(t,reflec,trans))
f.close()
# Print Potential Shape
f2 = open("Potential.txt",'w')
f2.write(grid.Potential(angsL, inp.n).plot_grid())
f2.close()

f3 = open("Probablity.txt",'w')
f3.write("# t(fs)    Reflection  Transmission\n")
f3.writelines(saveprob)
f3.close()

