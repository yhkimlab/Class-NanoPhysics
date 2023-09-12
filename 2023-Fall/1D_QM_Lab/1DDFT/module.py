import numpy as np
import Input

m0 = 9.110e-31
hbar = 1.055e-34
q = 1.602e-19
epsil_0 = 8.854e-12
j2ev = 6.242e18

epsil_rel = Input.epsil_rel
m_eff_c = Input.m_eff_c     #effective mass constant (m = m_eff_c * m0)

barrier = Input.barrier

class Conversion:

    def au2ang(self, x):
        a0_eff = (epsil_rel/m_eff_c) * ((4 * np.pi * epsil_0 * hbar**2)/(m0 * q**2))
        conv_x = a0_eff * x * 1e10
        return conv_x

    def ang2au(self, x):
        a0_eff = (epsil_rel / m_eff_c) * ((4 * np.pi * epsil_0 * hbar ** 2) / (m0 * q ** 2))
        conv_x = (1/a0_eff) * x / 1e10
        return conv_x

    def au2ev(self, x):
        hartree_eff = (m_eff_c/epsil_rel**2) * ((m0 * q**4)/(16*(np.pi)**2*(epsil_0)**2*(hbar)**2))
        conv_x = hartree_eff * x * j2ev
        return conv_x

    def ev2au(self, x):
        hartree_eff = (m_eff_c / epsil_rel ** 2) * ((m0 * q ** 4) / (16 * (np.pi) ** 2 * (epsil_0) ** 2 * (hbar) ** 2))
        conv_x = (1/hartree_eff) * x / j2ev
        return conv_x

def potential(n_source,n_channel,n_drain,n_tot):
    if barrier == 0:
        UBa = 0 * np.ones(n_tot)

    elif barrier == 1:
        UBa = np.hstack([np.zeros(n_source), Input.barrier_height * np.ones(n_channel), np.zeros(n_drain)])

    elif barrier == 2:
        UBa = np.hstack([np.zeros(n_source), Input.barrier_height * np.ones(4), np.zeros(n_channel - 8), Input.barrier_height * np.ones(4),
                        np.zeros(n_drain)])
    conversion = Conversion()
    return conversion.ev2au(UBa)

def hamiltonian(n_source, n_channel, n_drain, n_tot, t0):
    UB = potential(n_source, n_channel, n_drain, n_tot)
    D = 2 * t0 * np.ones((1, n_tot)) + UB # diagonal
    TL = -t0 * np.ones((1, n_tot - 1))  # subdiagonal
    TU = -t0 * np.ones((1, n_tot - 1))  # superdiagoanl

    T = np.diagflat(D) + np.diagflat(TU, +1) + np.diagflat(TL, -1)

    return T




