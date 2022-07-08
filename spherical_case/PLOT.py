#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:33:49 2020

@author: carl
"""

import numpy as np
from numpy import pi
from matplotlib import animation
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

S = 85
L = 85
thetas = np.linspace(1e-5, pi-1e-5, S)
phis   = np.delete(np.linspace(0, 2*pi, 2*S+1), -1)
theta, phi = np.meshgrid(thetas, phis)

def TIME_PLOT(p):
    
    rhos = np.load(p['name'] + '.npy')
    
    inclination = p['inc']
    phi_angle   = p['angle']
    color       = p['color']
    vmin        = p['vmin']
    vmax        = p['vmax']
    size        = p['size']
    nframes     = len(rhos)
    
    fig = plt.figure(figsize = (size, size), tight_layout = {'pad', 0})
    plotcrs = ccrs.Orthographic(0, 90 - inclination)
    ax = plt.subplot(projection = plotcrs)
    ax.relim()
    ax.autoscale_view()
    
    def init():
        return
    
    def animate(i):
        ax.clear()
        phi_fake     = np.vstack((phi, np.ones(S)*2*pi))
        theta_fake   = np.vstack((theta, theta[0]))
        phigif = (phi_fake - phi_angle * pi)*180/pi
        thetagif = (theta_fake - np.pi/2)*180/pi
        rhos_fake = np.vstack((rhos[i], rhos[i][0]))
        
        plt.pcolormesh(phigif, thetagif, rhos_fake, transform = ccrs.PlateCarree(), cmap = color,  vmin = vmin, vmax = vmax)
        ax.relim()
        ax.autoscale_view()
        return
    
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = nframes, interval = p['interval'], blit = False)
    anim.save(p['name'] + '_time_' + str(p['inc']) + '.mp4', fps = 1/p['interval'], writer = 'ffmpeg', dpi = 300)
    return

def FINAL_PLOT(p):
    
    rhos = np.load(p['name'] + '.npy')
    
    inclination = p['inc']
    color       = p['color']
    vmin        = p['vmin']
    vmax        = p['vmax']
    size        = p['size']
    nframes     = 62
    
    fig = plt.figure(figsize = (size, size), tight_layout = {'pad', 0})
    plotcrs = ccrs.Orthographic(0, 90 - inclination)
    ax = plt.subplot(projection = plotcrs)
    ax.relim()
    ax.autoscale_view()
    
    def init():
        return
    
    def animate(i):
        ax.clear()
        phi_fake     = np.vstack((phi, np.ones(S)*2*pi))
        theta_fake   = np.vstack((theta, theta[0]))
        phigif = (phi_fake - pi + i/10)*180/np.pi
        thetagif = (theta_fake - pi/2)*180/np.pi
        
        plt.pcolormesh(phigif, thetagif, np.vstack((rhos[-1], rhos[-1][0])), transform = ccrs.PlateCarree(), cmap = color,  vmin = vmin, vmax = vmax)
        ax.relim()
        ax.autoscale_view()
        return
    
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = nframes, interval = p['interval'], blit = False)
    anim.save(p['name'] + '_final.mp4', fps = 1/p['interval'], writer = 'ffmpeg', dpi = 300)
    return
    
def INIT_PLOT(p):
    
    rhos = np.load(p['name'] + '.npy')
    
    inclination = p['inc']
    color       = p['color']
    vmin        = np.min(rhos[0])
    vmax        = np.max(rhos[0])
    size        = p['size']
    nframes     = 62
    
    fig = plt.figure(figsize = (size, size), tight_layout = {'pad', 0})
    plotcrs = ccrs.Orthographic(0, 90 - inclination)
    ax = plt.subplot(projection = plotcrs)
    ax.relim()
    ax.autoscale_view()
    
    def init():
        return
    
    def animate(i):
        ax.clear()
        phi_fake     = np.vstack((phi, np.ones(S)*2*pi))
        theta_fake   = np.vstack((theta, theta[0]))
        phigif = (phi_fake - pi + i/10)*180/np.pi
        thetagif = (theta_fake - pi/2)*180/np.pi
        
        plt.pcolormesh(phigif, thetagif, np.vstack((rhos[0], rhos[0][0])), transform = ccrs.PlateCarree(), cmap = color,  vmin = vmin, vmax = vmax)
        ax.relim()
        ax.autoscale_view()
        return
    
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = nframes, interval = p['interval'], blit = False)
    anim.save(p['name'] + '_init.mp4', fps = 1/p['interval'], writer = 'ffmpeg', dpi = 300)
    return
    