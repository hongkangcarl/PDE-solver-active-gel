#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:07:52 2020

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

####### meshgrid

S = 85
L = 85
thetas = np.linspace(1e-5, pi-1e-5, S)
phis   = np.delete(np.linspace(0, 2*pi, 2*S+1), -1)
theta, phi = np.meshgrid(thetas, phis)

####### spherical harmonic
def Y(l, m, t, p):
    return sph_harm(m, l, p, t)

####### solution to the huge factorials
def Fact_Div(x1, x2):
    prod = 1
    for k in range(x1+1, x2+1):
        prod = prod * k
    return prod

####### gradient of spherical harmonic
def P(l, m, t):  # associated legendre polynomial
    if abs(m) > l:
        return 0
    else:
        return lpmv(m, l, cos(t))

def E(m, p):     # fourier mode
    return exp(1j * m * p)
def Norm(l, m):
    return sqrt( (2*l + 1)/(4*pi) / Fact_Div(l-m, l+m)  )
def DP_DT(l, m, t):
    return 1/sin(t) * (l*cos(t)*P(l, m, t) - (l+m)*P(l-1, m, t))
def DE_DP(m, p):
    return 1j * m * E(m, p)
def PSI(l, m, t, p):
    return Norm(l, m) * DP_DT(l, m, t) * E(m, p), 1/sin(t) * Norm(l, m) * P(l, m, t) * DE_DP(m, p)

####### plot functions

def Psg(mat):
    plt.figure(figsize = (12, 4))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(real(mat))
    plt.colorbar()
    plt.title("real part")
    plt.subplot(1, 2, 2)
    plt.pcolormesh(imag(mat))
    plt.colorbar()
    plt.title("imag part")
    plt.show()

def Pvalue(values):
    plt.figure(figsize = (6, 3))
    plt.pcolormesh(phi, theta, real(values))
    plt.colorbar()
    plt.show()
    
def Plot(value):
    plt.plot(real(value))
    plt.plot(imag(value))
    plt.show()

#%%
###### Compute the legendre on (m, theta) mesh:

AssoLegendre = np.zeros((S, 2*S, S))
for l in range(0, L):
    for m in range(0, l+1):
        AssoLegendre[l, m, :] = Norm(l, m) * P(l, m, thetas) * sin(thetas)
        
###### Compute the legendre on (l, m) mesh:
P_lm = np.zeros((S, S, S))
for l in range(0, L):
    for m in range(0, l+1):
        P_lm[:, l, m] = Norm(l, m) * P(l, m, thetas)

###### Compute the legendre on (m, theta) mesh:

PSItheta = np.zeros((S, 2*S, S), dtype = np.complex_)
PSIphi   = np.zeros((S, 2*S, S), dtype = np.complex_)
for l in range(0, L):
    for m in range(0, l+1):
        PSItheta[l, m, :] = Norm(l, m) * (l*cos(thetas)*P(l, m, thetas) - (l+m)*P(l-1, m, thetas))
        PSIphi[l, m, :]   = Norm(l, m) * 1j * m * P(l, m, thetas)

###### Compute the Psi on (l, m) mesh:
PSIt = np.zeros((S, S, S))
PSIp = np.zeros((S, S, S), dtype = np.complex_)
for l in range(0, L):
    for m in range(0, l+1):
        PSIt[:, l, m] = Norm(l, m) * DP_DT(l, m, thetas)
        PSIp[:, l, m] = Norm(l, m) * 1j * m * P(l, m, thetas) / sin(thetas)

#%%
###### Define the SHT Functions:

def FSHT(value):
    value_m = fft.fft(value, axis = 0)/(2*S)
    value_lm = np.zeros((S, S), dtype = np.complex_)
    for l in range(0, S):
        coefficient = np.trapz(value_m * AssoLegendre[l], thetas) * 2*pi
        value_lm[l] = coefficient[0:S]
    return value_lm

def ISHT(coeff):
    f_theta = np.zeros((S, S), dtype = np.complex_)
    for t in range(len(thetas)):
        f_theta[:, t] += np.sum(coeff * P_lm[t], axis = 0)
    value_theta = np.vstack((f_theta, np.zeros(S), 
                             np.flip(np.conj(f_theta[1:]),
                            axis = 0)))
    value = real(fft.ifft(value_theta, axis = 0)) * 2*S
    return value

        
def FVSHT(vector):
    vector_m = fft.fft(np.array(vector), axis = 1)/(2*S)
    vector_lm = np.zeros((S, S), dtype = np.complex_)
    for l in range(1, L):
        coefficient = np.trapz(vector_m[0] * PSItheta[l] - vector_m[1] * PSIphi[l], thetas) /l/(l+1) * 2*pi
        vector_lm[l] = coefficient[0:S]
    return vector_lm

def IVSHT(coeff):
    vec_theta = np.zeros((S, S), dtype = np.complex_)
    vec_phi = np.zeros((S, S), dtype = np.complex_)
    for t in range(S):
        vec_theta[:, t] = np.sum(coeff * PSIt[t], axis = 0)
        vec_phi[:, t]   = np.sum(coeff * PSIp[t], axis = 0)
    vec_theta = np.vstack((vec_theta, np.zeros(S), np.flip(np.conj(vec_theta[1:]), axis = 0)))
    vec_phi = np.vstack((vec_phi, np.zeros(S), np.flip(np.conj(vec_phi[1:]), axis = 0)))
    vec_theta = fft.ifft(vec_theta, axis = 0) * 2*S
    vec_phi   = fft.ifft(vec_phi,   axis = 0) * 2*S
    return vec_theta, vec_phi