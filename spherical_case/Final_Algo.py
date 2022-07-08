#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:51:52 2020

@author: carl
"""

## NEW MODEL

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
    
def Pdb(mat):
    plt.figure(figsize = (12, 8))
    plt.subplot(2, 2, 1)
    plt.pcolormesh(real(mat[0]))
    plt.colorbar()
    plt.title("theta real part")
    plt.subplot(2, 2, 2)
    plt.pcolormesh(imag(mat[0]))
    plt.colorbar()
    plt.title("theta imag part")
    plt.subplot(2, 2, 3)
    plt.pcolormesh(real(mat[1]))
    plt.colorbar()
    plt.title("phi real part")
    plt.subplot(2, 2, 4)
    plt.pcolormesh(imag(mat[1]))
    plt.colorbar()
    plt.title("phi imag part")
    plt.show()
    
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
    plt.figure(figsize = (12, 3))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(phi, theta, real(values))
    plt.colorbar()
    plt.title("real part")
    plt.subplot(1, 2, 2)
    plt.pcolormesh(phi, theta, imag(values))
    plt.colorbar()
    plt.title("imag part")
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
    
#%%
###### Start the simulation
    
def SIMULATION(file_name):
    
    global r, dt, step                  # spatiotemperal system
    global turn, rho0, hydro, alpha, D  # parameters
    global rho_initial, RHO, rho        # initial condition of density distribution
    global VELOCITY, ADVECTION, LAPLACIAN, ADV, V, v, rho_v, RHO_V
    global rhos
    
    ### SHT of initial condition and preferred density rho0
    
    RHO = FSHT(rho_initial - rho0)      # SPH of excessive density
    
    RHO0 = np.zeros(RHO.shape)          # SPH of preferred density
    RHO0[0, 0] = 2 * sqrt(pi) * rho0
    
    ### VELOCITY, ADVECTION, LAPLACIAN
    
    def sigma(rho):
        return rho**2/(rho**2 + rho0**2)
    
    VELOCITY = np.zeros(RHO.shape)
    ADVECTION = np.zeros(RHO.shape)
    LAPLACIAN = np.zeros(RHO.shape)
    
    for l in range(0, L):
        for m in range(0, l+1):
            VELOCITY[l, m] = alpha * r / ( 2 * hydro**2 * l * (l+1) + r**2 )
            ADVECTION[l, m] = - l * (l+1) / r
            LAPLACIAN[l, m] = - l * (l+1) / r**2
    
    ### iteration
    
    rhos = []
    
    for i in range(step):
        
        rho = ISHT(RHO) + rho0
        
        ## store values of rho per 100 steps
        
        if i % 10 == 0:
            rhos.append(real(rho))

        if i % 50 == 0:
            Pvalue(rho)
        
        ## active stress
        
        SIGMA = FSHT(sigma(rho) - sigma(rho0))
        
        ## velocity
        
        V = VELOCITY * SIGMA
        
        ## advection
        
        v = IVSHT(V)
        
        rho_v =  rho * v
        
        RHO_V = FVSHT(rho_v)
        
        ADV = ADVECTION * RHO_V
        
        RHO = (dt * (- ADV) + RHO) / (1 - dt * D * LAPLACIAN + dt/turn)
        
        np.save(file_name + '.npy', rhos)
        
    return

###### Data Analysis - Spherical Animation

def SPHERICAL_ANIMATION(inclination, Type, phi_angle):
    
    global rhos, color, file_name, nframes
    
    fig = plt.figure(figsize=(1,1),tight_layout = {'pad': 0})
    plotcrs = ccrs.Orthographic(0, 90 - inclination)
    ax = plt.subplot(projection=plotcrs)
    ax.relim()
    ax.autoscale_view()
    
    if Type == 'final':
        nframes = 62
    
    interval = 0.1
    
    def init():
        return 
    
    def animate(i):
        ax.clear()
        phi_fake     = np.vstack((phi, np.ones(S)*2*pi))
        theta_fake   = np.vstack((theta, theta[0]))
        if Type == 'time':
            phigif = (phi_fake - phi_angle*np.pi)*180/np.pi
            thetagif = (theta_fake - np.pi/2)*180/np.pi
            rhos_fake = np.vstack((rhos[i], rhos[i][0]))
            plt.pcolormesh(phigif, thetagif, rhos_fake, transform=ccrs.PlateCarree(), cmap = color,  vmin = 0, vmax = 5)
            ax.relim()
            ax.autoscale_view()
            if i % 50 == 0:
                print(i)
        if Type == 'final':
            phigif = (phi_fake - 1*np.pi + i/10)*180/np.pi
            thetagif = (theta_fake - np.pi/2)*180/np.pi
            plt.pcolormesh(phigif, thetagif, np.vstack((rhos[-1], rhos[-1][0])), transform=ccrs.PlateCarree(), cmap = color,  vmin = 0, vmax = 5)
            ax.relim()
            ax.autoscale_view()
        return
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nframes, interval=interval, blit=False)
    fig.set_size_inches(5, 5, True)
    anim.save(file_name, fps = 1/interval, writer = "pillow", dpi = 300)
    return
    
    
#%%

###### Explore the Parameter Space

r = 3

def Gaussian(t0, p0):
    return np.exp(-(theta - t0)**2/0.2 - (phi - p0)**2/0.2)*0.001

def rho_initial_generator():
    rho_initial = 1
    for i in range(10):
        rho_initial += Gaussian(np.random.random()*pi*0.6 + 0.2 * pi, np.random.random()*2*pi)
    return rho_initial

rho_initial = rho_initial_generator()

dt    = 0.05

turn  = 1

rho0  = 1

D = 0.1

hydro = 0.1

alpha = 0.5

step = 3000



rhos = SIMULATION()

#%%


inclination = 60

nframes     = 300

phi_angle   = pi

color       = 'afmhot'

file_name   = "Test1.gif"

import PLOT

PLOT.SPHERICAL_ANIMATION(inclination, 'time', pi)

#%%

file_name = 'Test2.gif'
SPHERICAL_ANIMATION(inclination, 'final', pi)


