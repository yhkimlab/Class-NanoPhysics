# %%
#!/usr/bin/env python3
"""
Simplified semiconductor device simulator using NumPy arrays and SciPy constants.
• Duplicated grid-size constants have been eliminated.
• The unit conversions use literal values.
• Only essential widgets are provided (a few parameter inputs and a Run button).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import display, clear_output
import scipy.constants as const

# Constants
EPSILON_0 = const.epsilon_0       # F/m
K_B       = const.k               # J/K
E_CHARGE  = const.e               # C
HP        = const.h               # J s

# -----------------------
# Simulator Class
# -----------------------
class DeviceSimulator:
    def __init__(self, params, Nx=5000, Temp=300):
        self.params = params
        self.Nx = Nx
        self.N_points = Nx + 2  # interior nodes + 2 boundaries
        self.Temp = Temp
        self.setup_mesh()
        self.ep = 11.7 * EPSILON_0
        self.Ut = K_B * self.Temp / E_CHARGE

        # Allocate arrays
        self.Voltage = np.zeros(self.N_points)
        self.E_field = np.zeros(self.N_points)
        self.p = np.zeros(self.N_points)
        self.n = np.zeros(self.N_points)
        self.Dop   = np.zeros(self.N_points)
        self.Dop_x = np.zeros(self.N_points)
        self.Dop_y = np.zeros(self.N_points)
        self.p_eq  = np.zeros(self.N_points)
        self.n_eq  = np.zeros(self.N_points)
        
        self.initialize_doping()
        self.initialize_equilibrium()

    def setup_mesh(self):
        xi = 0.0
        xf = self.params['semi_length']
        self.dx = (xf - xi) / (self.Nx + 1)
        self.x = np.array([xi + i * self.dx for i in range(self.N_points)])

    def initialize_doping(self):
        Na = self.params['Na']
        Nd = self.params['Nd']
        half_length = 0.5 * self.params['semi_length']
        self.Dop   = np.where(self.x < half_length, Na, -Nd)
        self.Dop_x = np.where(self.x < half_length, Na, 0.0)
        self.Dop_y = np.where(self.x < half_length, 0.0, Nd)

    def initialize_equilibrium(self):
        ni = self.params['ni']
        self.p_eq = 0.5 * (np.sqrt(self.Dop**2 + 4*ni**2) + self.Dop)
        self.n_eq = 0.5 * (np.sqrt(self.Dop**2 + 4*ni**2) - self.Dop)

    def initialize_boundary_conditions(self, v_applied):
        ni = self.params['ni']
        Ut = self.Ut
        Na = self.params['Na']
        Nd = self.params['Nd']
        Vi = -Ut * math.log(Na/ni) + v_applied
        Vf = Ut * math.log(Nd/ni)
        mid = self.N_points // 2
        self.Voltage[:mid] = Vi
        self.Voltage[mid:] = Vf
        self.E_field[1:self.N_points-1] = -(self.Voltage[2:self.N_points] - self.Voltage[0:self.N_points-2])/(2*self.dx)
        self.p = self.p_eq.copy()
        self.n = self.n_eq.copy()

    @staticmethod
    def bernoulli(x, tol=1e-7):
        if abs(x) < tol:
            return 1 - x/2 + x*x/12 - x**4/720
        else:
            return x / (math.exp(x) - 1)

    def tdm_solve(self, a, b, c, r, n_internal):
        beta = np.zeros(n_internal+2)
        rho  = np.zeros(n_internal+2)
        x_sol = np.zeros(n_internal+2)
        beta[1] = b[1]
        rho[1]  = r[1]
        for j in range(2, n_internal+1):
            beta[j] = b[j] - a[j]*c[j-1]/beta[j-1]
            rho[j]  = r[j] - a[j]*rho[j-1]/beta[j-1]
        x_sol[n_internal] = rho[n_internal]/beta[n_internal]
        for j in range(n_internal-1, 0, -1):
            x_sol[j] = (rho[j]-c[j]*x_sol[j+1])/beta[j]
        return x_sol

    def solve_poisson(self, tol=1e-7):
        n_int = self.N_points - 2
        dx, Ut = self.dx, self.Ut
        ec, ep = E_CHARGE, self.ep
        a = np.zeros(self.N_points)
        b = np.zeros(self.N_points)
        c = np.zeros(self.N_points)
        r = np.zeros(self.N_points)
        delta_V = np.zeros(self.N_points)

        niter = 0
        oldV = self.Voltage.copy()
        while True:
            for i in range(1, self.N_points-1):
                a[i] = 1.0
                b[i] = -2 - (ec/ep)*dx*dx/Ut * (self.p[i]*math.exp((oldV[i]-self.Voltage[i])/Ut) +
                                                 self.n[i]*math.exp((self.Voltage[i]-oldV[i])/Ut))
                c[i] = 1.0
                r[i] = -(self.Voltage[i+1] + self.Voltage[i-1] - 2*self.Voltage[i]) - \
                       (ec/ep)*dx*dx*(self.p[i]*math.exp((oldV[i]-self.Voltage[i])/Ut) -
                                      self.n[i]*math.exp((self.Voltage[i]-oldV[i])/Ut) - self.Dop[i])
            delta_sol = self.tdm_solve(a, b, c, r, n_int)
            diff = 0.0
            for i in range(1, self.N_points-1):
                delta_V[i] = delta_sol[i]
                self.Voltage[i] += delta_V[i]
                diff += abs(delta_V[i])
            if diff <= tol:
                break
            niter += 1
        self.E_field[1:self.N_points-1] = -(self.Voltage[2:self.N_points]-self.Voltage[0:self.N_points-2])/(2*dx)

    def solve_continuity(self, tol_factor=1e-8):
        n_int = self.N_points - 2
        dx, Ut = self.dx, self.Ut
        # Compute mobilities and diffusion coefficients
        mu_p = np.zeros(self.N_points)
        mu_n = np.zeros(self.N_points)
        Dp = np.zeros(self.N_points)
        Dn = np.zeros(self.N_points)
        for i in range(self.N_points):
            if self.x[i] < 0.5*self.params['semi_length']:
                mu_p[i] = self.params['mu_p_P']
                mu_n[i] = self.params['mu_n_P']
            else:
                mu_p[i] = self.params['mu_p_N']
                mu_n[i] = self.params['mu_n_N']
            Dp[i] = mu_p[i] * Ut
            Dn[i] = mu_n[i] * Ut
        
        tau_n = self.params['tau_n']
        tau_p = self.params['tau_p']
        p1 = self.params['p1']
        n1 = self.params['n1']
        F_scatt = self.params['F_scatt']
        ni = self.params['ni']
        
        # Solve for holes
        a = np.zeros(self.N_points)
        b = np.zeros(self.N_points)
        c = np.zeros(self.N_points)
        r = np.zeros(self.N_points)
        dp = np.zeros(self.N_points)
        tol = ((self.params['Na']+self.params['Nd'])/2)*tol_factor
        while True:
            for i in range(1, self.N_points-1):
                RecombRate = 1.0/(tau_n*(self.p[i]+p1) + tau_p*(self.n[i]+n1))
                co0 = Dp[i]/(dx*dx)
                co1 = Dp[i+1]/(dx*dx)
                co_a = co0*self.bernoulli((self.Voltage[i]-self.Voltage[i-1])/Ut)
                co_b = co0*self.bernoulli((self.Voltage[i-1]-self.Voltage[i])/Ut) + \
                       co1*self.bernoulli((self.Voltage[i+1]-self.Voltage[i])/Ut)
                co_c = co1*self.bernoulli((self.Voltage[i]-self.Voltage[i+1])/Ut)
                a[i] = co_a
                b[i] = -co_b - RecombRate*self.n[i] + (RecombRate**2)*tau_n*(self.p[i]*self.n[i]-ni**2)
                if not F_scatt:
                    b[i] = -co_b
                c[i] = co_c
                r[i] = -(co_a*self.p[i-1] - co_b*self.p[i] + co_c*self.p[i+1] - 
                         RecombRate*(self.p[i]*self.n[i]-ni**2))
                if not F_scatt:
                    r[i] = 0.0
            if not F_scatt:
                r[1] = -a[1]*self.p[0]
                a[1] = 0.0
                r[self.N_points-2] = -c[self.N_points-2]*self.p[self.N_points-1]
                c[self.N_points-2] = 0.0
                dp_sol = self.tdm_solve(a,b,c,r,n_int)
                for i in range(1, self.N_points-1):
                    self.p[i] += dp_sol[i]
                break
            else:
                dp_sol = self.tdm_solve(a,b,c,r,n_int)
                diff = 0.0
                for i in range(1, self.N_points-1):
                    dp[i] = dp_sol[i]
                    self.p[i] += dp[i]
                    diff += abs(dp[i])
                if diff <= tol:
                    break

        # Solve for electrons (similar structure)
        a = np.zeros(self.N_points)
        b = np.zeros(self.N_points)
        c = np.zeros(self.N_points)
        r = np.zeros(self.N_points)
        dn = np.zeros(self.N_points)
        while True:
            for i in range(1, self.N_points-1):
                RecombRate = 1.0/(tau_n*(self.p[i]+p1) + tau_p*(self.n[i]+n1))
                co0 = Dn[i]/(dx*dx)
                co1 = Dn[i+1]/(dx*dx)
                co_a = co0*self.bernoulli((self.Voltage[i-1]-self.Voltage[i])/Ut)
                co_b = co1*self.bernoulli((self.Voltage[i]-self.Voltage[i+1])/Ut) + \
                       co0*self.bernoulli((self.Voltage[i]-self.Voltage[i-1])/Ut)
                co_c = co1*self.bernoulli((self.Voltage[i+1]-self.Voltage[i])/Ut)
                a[i] = co_a
                b[i] = -co_b - RecombRate*self.p[i] + (RecombRate**2)*tau_p*(self.p[i]*self.n[i]-ni**2)
                if not F_scatt:
                    b[i] = -co_b
                c[i] = co_c
                r[i] = -(co_a*self.n[i-1] - co_b*self.n[i] + co_c*self.n[i+1] - 
                         RecombRate*(self.p[i]*self.n[i]-ni**2))
                if not F_scatt:
                    r[i] = 0.0
            if not F_scatt:
                r[1] = -a[1]*self.n[0]
                a[1] = 0.0
                r[self.N_points-2] = -c[self.N_points-2]*self.n[self.N_points-1]
                c[self.N_points-2] = 0.0
                dn_sol = self.tdm_solve(a,b,c,r,n_int)
                for i in range(1, self.N_points-1):
                    self.n[i] += dn_sol[i]
                break
            else:
                dn_sol = self.tdm_solve(a,b,c,r,n_int)
                diff = 0.0
                for i in range(1, self.N_points-1):
                    dn[i] = dn_sol[i]
                    self.n[i] += dn[i]
                    diff += abs(dn[i])
                if diff <= tol:
                    break

    def calculate_current_density(self):
        n_int = self.N_points - 2
        dx, Ut = self.dx, self.Ut
        ec = E_CHARGE
        mu_p = np.zeros(self.N_points)
        mu_n = np.zeros(self.N_points)
        Dp = np.zeros(self.N_points)
        Dn = np.zeros(self.N_points)
        for i in range(self.N_points):
            if self.x[i] < 0.5*self.params['semi_length']:
                mu_p[i] = self.params['mu_p_P']
                mu_n[i] = self.params['mu_n_P']
            else:
                mu_p[i] = self.params['mu_p_N']
                mu_n[i] = self.params['mu_n_N']
            Dp[i] = mu_p[i]*Ut
            Dn[i] = mu_n[i]*Ut

        Jp = np.zeros(self.N_points)
        Jn = np.zeros(self.N_points)
        Jt = np.zeros(self.N_points)
        JV_average = 0.0
        for i in range(1, self.N_points-1):
            Jp[i] = -ec * Dp[i+1] * (self.bernoulli((self.Voltage[i]-self.Voltage[i+1])/Ut)*self.p[i+1] -
                                      self.bernoulli((self.Voltage[i+1]-self.Voltage[i])/Ut)*self.p[i]) / dx
            Jn[i] = -ec * Dn[i+1] * (self.bernoulli((self.Voltage[i]-self.Voltage[i+1])/Ut)*self.n[i] -
                                      self.bernoulli((self.Voltage[i+1]-self.Voltage[i])/Ut)*self.n[i+1]) / dx
            Jt[i] = Jp[i] + Jn[i]
            JV_average += Jt[i]
        JV_average /= n_int
        JV_plot = JV_average/10000  # unit conversion
        return {'Jp': Jp, 'Jn': Jn, 'Jt': Jt, 'JV_average': JV_average, 'JV_plot': JV_plot}

    def run(self, v_applied):
        self.initialize_boundary_conditions(v_applied)
        tol = 1e-7
        self.solve_poisson(tol)
        self.solve_continuity(tol)
        curr = self.calculate_current_density()
        Fp = -self.Voltage - self.Ut * np.log(self.p/self.params['ni'])
        Fn = -self.Voltage + self.Ut * np.log(self.n/self.params['ni'])
        Charge = (self.p - self.n - self.Dop) / 1e6
        Eg = 1.12
        E_band_con = -self.Voltage + 0.5*Eg
        E_band_int = -self.Voltage
        E_band_val = -self.Voltage - 0.5*Eg
        dV_dx = (self.Voltage[2:self.N_points] - self.Voltage[0:self.N_points-2])/(2*self.dx)
        depletion_length = 0.0
        if dV_dx.size > 0:
            threshold = np.max(np.abs(dV_dx)) * 0.1
            indices = np.where(np.abs(dV_dx) > threshold)[0]
            if indices.size > 0:
                depletion_length = (self.x[indices[-1]+1] - self.x[indices[0]+1]) / 1e-6
        return {
            'v_applied': v_applied,
            'x': self.x/1e-6,
            'Dop_x': self.Dop_x/1e6,
            'Dop_y': self.Dop_y/1e6,
            'Fp': Fp,
            'Fn': Fn,
            'Charge': Charge,
            'E_potential': self.Voltage.copy(),
            'p_plot': self.p/1e6,
            'n_plot': self.n/1e6,
            'E_field_plot': self.E_field/1e5,
            'E_band_con': E_band_con,
            'E_band_int': E_band_int,
            'E_band_val': E_band_val,
            'current_density': curr,
            'depletion_length': depletion_length
        }


def set_default_parameters():
    return {
        'V_min': 0.0,
        'V_max': 0.6,
        'V_step': 0.1,
        'Vd': 0.6,
        'ni': 1.5e10*1e6,
        'semi_length': 20.0e-6,  # 20 µm in meters
        'Na': 1.0e16*1e6,
        'Nd': 1.0e15*1e6,
        'mu_p_P': 200e-4,
        'mu_p_N': 450e-4,
        'mu_n_P': 700e-4,
        'mu_n_N': 1300e-4,
        'tau_n': 0.001e-6,       # in seconds
        'tau_p': 0.001e-6,
        'p1': 0.0,
        'n1': 0.0,
        'F_scatt': True
    }

def update_parameters_from_widgets(param_widgets):
    return {
        'V_min': param_widgets[0].value,
        'V_max': param_widgets[1].value,
        'V_step': param_widgets[2].value,
        'Vd': param_widgets[3].value,
        'ni': param_widgets[4].value,
        'semi_length': param_widgets[5].value,
        'Na': param_widgets[6].value,
        'Nd': param_widgets[7].value,
        'mu_p_P': param_widgets[8].value,
        'mu_p_N': param_widgets[9].value,
        'mu_n_P': param_widgets[10].value,
        'mu_n_N': param_widgets[11].value,
        'tau_n': param_widgets[12].value,
        'tau_p': param_widgets[13].value,
        'p1': param_widgets[14].value,
        'n1': param_widgets[15].value,
        'F_scatt': param_widgets[16].value
    }

def run_simulation(params):
    clear_output(wait=True)
    results_list = []
    v_values = np.around(np.arange(params['V_min'], 
                                    params['V_max'] + params['V_step']/2, 
                                    params['V_step']), decimals=1)
    for v in v_values:
        sim = DeviceSimulator(params)
        result = sim.run(v)
        results_list.append(result)
    plot_simulation_results(results_list)

def plot_simulation_results(results_list):
    fig, axes = plt.subplots(3, 3, figsize=(12,10))
    axes = axes.flatten()
    result = results_list[-1]
    x = result['x']
    axes[0].plot(x, result['Dop_x'], label="Dop_x")
    axes[0].plot(x, result['Dop_y'], label="Dop_y")
    axes[0].set_title('Doping Density')
    axes[0].set_xlabel('x (µm)')
    axes[0].set_ylabel('Doping (/cm³)')
    axes[0].legend()

    axes[1].set_title('Carrier Density')
    axes[1].set_xlabel('x (µm)')
    axes[1].set_ylabel('Carrier Density (/cm³)')
    axes[1].set_yscale('log')
    axes[1].plot(x, result['p_plot'], 'r', label='p')
    axes[1].plot(x, result['n_plot'], 'b', label='n')
    axes[1].legend()

    axes[2].plot(x, result['Charge'])
    axes[2].set_title('Net Charge Density')
    axes[2].set_xlabel('x (µm)')
    axes[2].set_ylabel('Charge (/cm³)')

    axes[3].plot(x, result['E_field_plot'])
    axes[3].set_title('Electric Field')
    axes[3].set_xlabel('x (µm)')
    axes[3].set_ylabel('E Field (kV/cm)')

    axes[4].plot(x, result['E_potential'])
    axes[4].set_title('Electrostatic Potential')
    axes[4].set_xlabel('x (µm)')
    axes[4].set_ylabel('Potential (V)')

    shift = result['v_applied'] / 2
    axes[5].plot(x, result['E_band_con'] + shift, label='Conduction Band')
    axes[5].plot(x, result['E_band_int'] + shift, '--', label='Intrinsic Level')
    axes[5].plot(x, result['E_band_val'] + shift, label='Valence Band')
    axes[5].plot(x, result['Fp'] + shift, label='Fp')
    axes[5].plot(x, result['Fn'] + shift, label='Fn')
    axes[5].set_title('Energy Bands')
    axes[5].set_xlabel('x (µm)')
    axes[5].set_ylabel('Energy (eV)')
    axes[5].legend()

    v_vals = [r['v_applied'] for r in results_list]
    JV_vals = [r['current_density']['JV_plot'] for r in results_list]
    axes[6].plot(v_vals, JV_vals, 'bo-')
    axes[6].set_title('J-V Curve')
    axes[6].set_xlabel('Bias (V)')
    axes[6].set_ylabel('Current Density (A/cm²)')

    axes[7].set_yscale('log')
    axes[7].plot(x, result['current_density']['Jt'], label='J_total')
    axes[7].set_title('Current Density Distribution')
    axes[7].set_xlabel('x (µm)')
    axes[7].set_ylabel('Current Density (A/cm²)')
    axes[7].legend()

    depletion_vals = [r['depletion_length'] for r in results_list]
    axes[8].plot(v_vals, depletion_vals, 'gs-')
    axes[8].set_title('Depletion Region Length')
    axes[8].set_xlabel('Bias (V)')
    axes[8].set_ylabel('Depletion Length (µm)')

    plt.tight_layout()
    plt.show()

def main():
    default_params = set_default_parameters()
    run_simulation(default_params)

if __name__ == '__main__':
    main()


