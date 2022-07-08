#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:51:47 2020

@author: carl
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from  numpy.fft import fft2, ifft2, fftshift, ifftshift


D     = 1.
gamma = 80


L = 40      # µm

# parameters

dt   = 0.05     # sec
# steps = 3000
turn = 10       # sec
eta  = 10       # pN·s/µm
sigma_m = 500   # pN/µm
rho0 = 0.5      # rho_s


dx = 0.1
Nx = int(L/dx)

xs = np.arange(Nx) * dx
ys = np.arange(Nx) * dx

X, Y = np.meshgrid(xs, ys)

# rho_init = rho0 + 1e-2 * np.exp(-(X - L/2)**2/4e-2) * 1e-2 * np.exp(-(Y - L/2)**2/4e-2)
rho_init = np.random.random(X.shape) * 0.1 + rho0

def FFT(value):
    return fftshift(fft2(value))

def IFT(vec):
    return ifft2(ifftshift(vec))

def MUL(vec1, vec2):
    result = sp.convolve(vec1, vec2, mode = 'same')/Nx**2
    result = np.roll(result, -1, axis = 0)
    result = np.roll(result, -1, axis = 1)
    return result

kx = (np.arange(0, Nx) - Nx/2) * 2*np.pi/L
ky = (np.arange(0, Nx) - Nx/2) * 2*np.pi/L
Kx, Ky = np.meshgrid(kx, ky)
K2 = Kx**2 + Ky**2

RHO0 = FFT(rho0 + 0*X)

def VELOCITY(rho):
    sigma_act = sigma_m * rho/(rho+1)
    SIGMA = FFT(sigma_act)
    Vx = 1j * Kx * SIGMA / (2 * eta * K2 + gamma)
    Vy = 1j * Ky * SIGMA / (2 * eta * K2 + gamma)
    V = np.array([Vx, Vy])
    return V

def ADVECTION(RHO, V):
    RHO_Vx = MUL(RHO, V[0])
    RHO_Vy = MUL(RHO, V[1])
    ADV = 1j * Kx * RHO_Vx + 1j * Ky * RHO_Vy
    return ADV

def next_rho(RHO, ADV):
    NEXT_RHO = (dt * (-ADV + RHO0/turn) + RHO) / (1 + dt/turn + dt * D * K2)
    return np.real(IFT(NEXT_RHO))



# run simulation

def main(steps):
    
    rho = rho_init
    # rho = np.load('rho.npy')
    
    for i in range(steps):
        
        if i%100 == 0:
            plt.pcolormesh(X, Y, rho)
            plt.colorbar()
            plt.title(str(i))
            plt.show()
            plt.close()
            
        
        V = VELOCITY(rho)
        RHO = FFT(rho)
        ADV = ADVECTION(RHO, V)
        rho2 = next_rho(RHO, ADV)
        
        rho = rho2









