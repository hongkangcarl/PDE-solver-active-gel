#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:06:14 2020

@author: carl
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
from numpy import cos, sin, exp, pi, sqrt
from scipy.special import sph_harm
from scipy.special import lpmv
from matplotlib import animation
import cartopy.crs as ccrs
real = np.real
imag = np.imag
import SHT
import SIMULATION
import PLOT

#%%

S = 85
L = 85
thetas = np.linspace(1e-5, pi-1e-5, S)
phis   = np.delete(np.linspace(0, 2*pi, 2*S+1), -1)
theta, phi = np.meshgrid(thetas, phis)

Iseed = 1
np.random.seed(Iseed)

def Gaussian(t0):
    return np.exp(-(theta - t0)**2/0.05) * 0.1

def rho_initial_generator():
    rho_initial = 1
    Iseed = 4
    np.random.seed(Iseed)
    for i in range(1):
        rho_initial += 1 + 0 * theta
    return rho_initial

# rho_initial = np.random.random(theta.shape) * 0.1 + 0.95

rho_initial = 1 - (sin(theta)**3 * cos(3*phi)/2)**2

rho0 = 1
# for i in range(6):
#     rho0 = rho0 - Gaussian(2.6, 0.4 + i * 2*pi/6)
    
rhos = np.load('local_test_2.npy')

parameter = {'dt'    : 0.03,
             'r'     : 3,
             'turn'  : 3,
             'rho0'  : rho0,
             'D'     : 0.01,
             'hydro' : 1,
             'alpha' : 5, #+ Gaussian(2.6) * np.random.random(phi.shape),
             'step'  : 5000,
             'init'  : rhos[-1],
             'name'  : 'local_test_3'
             }
    

rhos = SIMULATION.run(parameter)

#%%



plot_parameter = {'inc'      : 30,
                  'angle'    : pi,
                  'color'    : 'coolwarm',
                  'name'     : 'local_test_3',
                  'interval' : 0.01,
                  'vmin'     : 0,
                  'vmax'     : 6,
                  'size'     : parameter['r']
                  }
                  
PLOT.TIME_PLOT(plot_parameter)

# plot_parameter['inc'] = 120

# PLOT.TIME_PLOT(plot_parameter)

# plot_parameter['interval'] = 0.1

# PLOT.FINAL_PLOT(plot_parameter)







