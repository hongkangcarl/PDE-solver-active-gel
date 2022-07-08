#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:15:37 2020

@author: carl
"""

import numpy as np
from numpy import sqrt, pi
import SHT
real = np.real
imag = np.imag

L = 85

def run(p):
    
    RHO = SHT.FSHT(p['init'] - p['rho0'])
    
    RHO0 = np.zeros(RHO.shape)
    RHO0[0, 0] = 2 * sqrt(pi) * p['rho0']
    
    def sigma(x):
        return x**2 / (x**2 + p['rho0']**2)
    
    VELOCITY = np.zeros(RHO.shape)
    ADVECTION = np.zeros(RHO.shape)
    LAPLACIAN = np.zeros(RHO.shape)
    
    for l in range(0, L):
        for m in range(0, l+1):
            VELOCITY[l, m] = p['alpha'] * p['r'] / ( 2 * p['hydro']**2 * l * (l+1) + p['r']**2 )
            ADVECTION[l, m] = - l * (l+1) / p['r']
            LAPLACIAN[l, m] = - l * (l+1) / p['r']**2
    
    rhos = []
    
    for i in range(p['step']):
        
        rho = SHT.ISHT(RHO) + p['rho0']
        
        if i % 10 == 0:
            rhos.append(real(rho))
        
        if i % 50 == 0:
            SHT.Pvalue(rho)
            
        SIGMA = SHT.FSHT(sigma(rho) - sigma(p['rho0']))
        
        V = VELOCITY * SIGMA
        
        v = SHT.IVSHT(V)
        
        RHO_V = SHT.FVSHT(rho * v)
        
        ADV = ADVECTION * RHO_V
        
        RHO = (p['dt'] * (- ADV) + RHO) / (1 - p['dt'] * p['D'] * LAPLACIAN + p['dt']/p['turn'])
        
    np.save(p['name'] + '.npy', rhos)
    
    return
        
        
        
        