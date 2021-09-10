#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 17:15:22 2021

@author: mrborges
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow.keras
import matplotlib.pyplot as plt
import random
from functions_mrb import diagnostic_graphs2, hist_error_rel
print('Tensorflow version == ',tf.__version__)

# read data ###################################################################
home     = './'
namep    = home + 'figuras/'
namein   = 'data/datan.mat'
z  = np.fromfile(namein , dtype=np.float32)
n  = np.int(np.size(z)/2)
z  = np.transpose(np.reshape(z,(2,n)))
namein   = 'data/datax.mat'
x  = np.fromfile(namein , dtype=np.float32)
n  = np.int(np.size(x)/2)
x  = np.transpose(np.reshape(x,(2,n)))
namein   = 'data/datay.mat'
y  = np.fromfile(namein , dtype=np.float32)
namein   = 'data/datat.mat'
t  = np.fromfile(namein , dtype=np.float32)
# Normalizing data ############################################################
xmax = np.array([np.max(x[:,0]), np.max(x[:,1])])
xmin = np.array([np.min(x[:,0]), np.min(x[:,1])])
ymax = np.max(y)
ymin = np.min(y)
x    = (x - xmin)/(xmax - xmin)
y    = (y - ymin)/(ymax - ymin)
z    = (z - xmin)/(xmax - xmin)
# read model ##################################################################
model = tf.keras.models.load_model('models/camclay_model.h5')
# Check its architecture
model.summary()
# Prediction by model #########################################################
predict = model.predict(z)
predict = np.reshape(predict,(np.size(predict),))
# Return to original data #####################################################
x      = x*(xmax - xmin) + xmin
y      = y*(ymax - ymin) + ymin
z      = z*(xmax - xmin) + xmin
predict= predict*(ymax - ymin) + ymin
# Plot data ###################################################################
fig = plt.figure(constrained_layout=True)
fig, axs = plt.subplots(nrows=1,ncols=2, constrained_layout=True)
axs[0].plot(z[:,0],predict,'tab:blue',linewidth=3)
axs[0].set_title('CamClay',fontsize=18)
axs[0].tick_params(axis="x", labelsize=16)
axs[0].tick_params(axis="y", labelsize=16)
axs[0].set_ylabel('$Void\ ratio$', fontsize=18, weight='bold', color='k')
axs[0].set_xlabel('$p_{mean}\ (MPa)$', fontsize=18, weight='bold', color='k')
#
axs[1].plot(t,z[:,0],'tab:blue',linewidth=3)
axs[1].plot(t,z[:,1],'tab:orange',linewidth=3)
axs[1].set_title('CamClay',fontsize=18)
axs[1].tick_params(axis="x", labelsize=16)
axs[1].tick_params(axis="y", labelsize=16)
axs[1].set_xlabel('$time\ (days)$', fontsize=18, weight='bold', color='k')
axs[1].set_ylabel('$p\ (MPa)$', fontsize=18, weight='bold', color='k')
axs[1].legend(['$p_{mean}$', '$p_0$'], loc='upper left')
name = 'figuras/new_camclay_data.png'
plt.savefig(name, transparent=True, dpi=300)
plt.show()
###############################################################################
fig = plt.figure(constrained_layout=True)
ax  = plt.axes(projection='3d')
#ax.plot3D(x[:,0],x[:,1],y[:],'tab:orange', linewidth=1.5)
#ax.scatter(x[:,0],x[:,1],y[:],cmap='viridis', linewidth=0.5)
ax.scatter(z[:,0],z[:,1],predict, color='tab:blue', linewidth=0.5)
ax.set_xlabel('$p_{mean}\ (MPa)$')
ax.set_ylabel('$p_0\ (MPa)$')
ax.set_zlabel('$Void\ ratio$')
ax.view_init(5, 35)
name = 'figuras/new_camclay_network3d.png'
plt.savefig(name, transparent=True, dpi=300)
plt.show()
