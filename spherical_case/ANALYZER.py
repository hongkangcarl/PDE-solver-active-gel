#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:32:50 2020

@author: carl
"""

import sys
import numpy as np

import PLOT

plot_parameter = {'inc'      : 60,
                  'angle'    : np.pi,
                  'color'    : 'Greens',
                  'name'     : 'test_' + sys.argv[2] + '_' +  sys.argv[1],
                  'interval' : 0.1,
                  'vmin'     : 0,
                  'vmax'     : 5,
                  'size'     : float(sys.argv[1])/2
                  }
                  
PLOT.TIME_PLOT(plot_parameter)
PLOT.FINAL_PLOT(plot_parameter)