#From st-0111, zero forward speed, sumemr load line and even keel. Must check in c205 about the whole forward speed
import numpy as np
import ship_characteristics as sc

wind_bf=[0,1.5,3.4,5.4,7.9,10.7,13.8,17.1,20.7,24.4,28.4,32.6] #m/s
wave_height_bf=[0,0.1,0.4,0.8,1.3,2.1,3.1,4.2,5.7,7.4,9.5,12.1] #m
current_bf=[0,0.25,0.5,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75] #m/s
wave_period_bf=[1,3.5,4.5,5.5,6.5,7.5,8.5,9,10,10.5,11.5,12]  #s
bf=[0,1,2,3,4,5,6,7,8,9,10,11]

import numpy as np

class EnvironmentalForces:
    def __init__(self, direction_deg, ship_parameters, environmental_speeds):
        self.direction_rad = np.radians(direction_deg)
        self.ship = ship_parameters
        self.env = environmental_speeds
        self.rho_air = 1.226  # kg/m^3
        self.rho_water = 1026  # kg/m^3
        self.dir= self.direction(direction_rad)

    def WindForces(self):
        AF = self.ship['AF_wind']
        AL = self.ship['AL_wind']
        XL_air = self.ship["XL_air"]
        V = self.env['V_wind']
        Fx = 0.5 * self.rho_air * V**2 * (-0.7*AF * np.cos(self.direction_rad) )
        Fy = 0.5 * self.rho_air * V**2 * (0.9*AL * np.sin(self.direction_rad) )
        Mz = Fy* (XL_air+0.3*(1-(2*self.dir/pi)) 
        return [Fx, Fy, Mz]

    def CurrentForces(self):
        AL = self.ship['AL_current']
        XL_current = self.XL_current()
        V = self.env['V_current']
        Fx = 0.5 * self.rho_water * V**2 * AL * np.cos(self.direction_rad)
        Fy = 0.5 * self.rho_water * V**2 * AL * np.sin(self.direction_rad)
        Mz = 0.5 * self.rho_water * V**2 * XL_current * AL * np.sin(self.direction_rad)
        return Fx, Fy, Mz

    def WaveForces(self):
        Hs = self.env['Hs']
        LOS = self.ship['LOS']
        XLos = self.XLos()
        CWLaft = self.CWLaft()
        Fx = 0.5 * Hs**2 * LOS * np.cos(self.direction_rad) * CWLaft
        Fy = 0.5 * Hs**2 * LOS * np.sin(self.direction_rad) * CWLaft
        Mz = 0.5 * Hs**2 * LOS * XLos * np.sin(self.direction_rad) * CWLaft
        return Fx, Fy, Mz

    def CWLaft(self):
        Lpp = self.ship['Lpp']
        B = self.ship['B']
        AWLaft = self.ship['AWLaft']
        CWLaft = AWLaft / (Lpp / 2 * B)
        return min(max(CWLaft, 0.85), 1.15)

    def XL_air(self):
        h = self.ship['superstructure_height']
        Lpp = self.ship['Lpp']
        return 0.6 * h - Lpp / 2

    def XL_current(self):
        draft = self.ship['draft']
        Lpp = self.ship['Lpp']
        return 0.4 * draft - Lpp / 2

    def XLos(self):
        Lpp = self.ship['Lpp']
        return 0.25 * Lpp - Lpp / 2

    def bowangle(self):
        B = self.ship['B']
        xmax = self.ship['xmax']
        xb4 = self.ship['xb4']
        return np.arctan((B / 4) / (xmax - xb4))

    def t_surge(self):
        return self.WaveForces()[0]

    def t_sway(self):
        return self.WaveForces()[1]

    def f_t_prime(self):
        return np.sqrt(self.t_surge()**2 + self.t_sway()**2)

    def h1(self):
        return self.env['Hs']**2 * self.ship['LOS']

    def h2(self):
        return self.h1() * np.cos(self.direction_rad)

    def h1a(self):
        return self.h1() * np.sin(self.direction_rad)

    def h1b(self):
        return self.h1() * self.XLos()

    def h(self):
        return self.h1() * self.CWLaft()

    def dir(self):
        return self.direction_rad