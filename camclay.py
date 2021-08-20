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
namein   = 'data/datax.mat'
x  = np.fromfile(namein , dtype=np.float32)
n  = np.int(np.size(x)/2)
x  = np.transpose(np.reshape(x,(2,n)))
namein   = 'data/datay.mat'
y  = np.fromfile(namein , dtype=np.float32)
namein   = 'data/datat.mat'
t  = np.fromfile(namein , dtype=np.float32)
# Plot data ###################################################################
fig = plt.figure(constrained_layout=True)
fig, axs = plt.subplots(nrows=1,ncols=2, constrained_layout=True)
axs[0].plot(x[:,0],y,'tab:blue',linewidth=3)
axs[0].set_title('CamClay',fontsize=18)
axs[0].tick_params(axis="x", labelsize=16)
axs[0].tick_params(axis="y", labelsize=16)
axs[0].set_ylabel('$Void\ ratio$', fontsize=18, weight='bold', color='k')
axs[0].set_xlabel('$p_{mean}\ (MPa)$', fontsize=18, weight='bold', color='k')
#
axs[1].plot(t,x[:,0],'tab:blue',linewidth=3)
axs[1].plot(t,x[:,1],'tab:orange',linewidth=3)
axs[1].set_title('CamClay',fontsize=18)
axs[1].tick_params(axis="x", labelsize=16)
axs[1].tick_params(axis="y", labelsize=16)
axs[1].set_xlabel('$time\ (days)$', fontsize=18, weight='bold', color='k')
axs[1].set_ylabel('$p\ (MPa)$', fontsize=18, weight='bold', color='k')
axs[1].legend(['$p_{mean}$', '$p_0$'], loc='upper left')
name = 'figuras/camclay_data.png'
plt.savefig(name, transparent=True, dpi=300)
plt.show()
# Normalizing data ############################################################
xmax = np.array([np.max(x[:,0]), np.max(x[:,1])])
xmin = np.array([np.min(x[:,0]), np.min(x[:,1])])
ymax = np.max(y)
ymin = np.min(y)
x    = (x - xmin)/(xmax - xmin)
y    = (y - ymin)/(ymax - ymin)
# Sets ########################################################################
rate = 0.5
rt   = 0.5*(1 - rate)
ntrain = np.int(n*rate)
nteste = np.int(n*rt)
nvalid = np.int(n - (ntrain + nteste))
lista = list(range(n))
random.shuffle(lista)
xvalid = x[lista[0:nvalid],:]
yvalid = y[lista[0:nvalid]]
xteste = x[lista[nvalid:nvalid+nteste],:]
yteste = y[lista[nvalid:nvalid+nteste]]
xtrain = x[lista[nvalid+nteste:-1],:]
ytrain = y[lista[nvalid+nteste:-1]]
# Neural ######################################################################
inputshape = (2,)
lrate      = 0.0001
num_class  = 1
epochs     = 250
batch_size = 64
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=inputshape),
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dense(num_class, activation='linear')
    ])
print(model.summary())
tf.keras.utils.plot_model(model, 'figuras/model_with_shape_info.png', show_shapes=True)
###############################################################################
# Compilacao
metrica=tf.keras.metrics.MeanSquaredError()

model.compile(loss='mean_squared_error',
              metrics=[metrica],
              optimizer=tf.keras.optimizers.Adam(learning_rate = lrate))

# Criterio de parada
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', 
                                      verbose=1, patience=20,
                                      min_delta=0, baseline=None,
                                      restore_best_weights=False)

# Train
history = model.fit(xtrain, ytrain, epochs=epochs,
          batch_size=batch_size,
          verbose=1,
          validation_data=(xvalid, yvalid),validation_freq=1,
          callbacks=[es])

# Plot training & validation accuracy values ##################################
lista_metric = list()
#print(history.history.keys())
for i in history.history.keys():
    lista_metric.append(i)
    
loss    = lista_metric[0]
metric  = lista_metric[1]
vloss   = lista_metric[2]
vmetric = lista_metric[3]
diagnostic_graphs2(history,namep,loss,metric,vloss,vmetric)

###############################################################################
# Evaluation of Generalisation
print('\n______________________________________________________\nValidation\n')
score = model.evaluate(x = xteste, y = yteste,
                       batch_size=None, verbose=1, sample_weight=None,
                       steps=None, callbacks=None, max_queue_size=10,
                       workers=1, use_multiprocessing=True)
print('\nTest ' + loss + '.........: {:5.3e}'.format(score[0]))
print('Test ' + metric + '.....: {:5.3e}%'.format(100*score[1]))

predict = model.predict(xteste)
predict = np.reshape(predict,(np.size(predict),))

# Return to original data #####################################################
x    = (x)*(xmax - xmin) + xmin
y    = (y)*(ymax - ymin) + ymin
xteste = (xteste)*(xmax - xmin)  + xmin
yteste = (yteste )*(ymax - ymin) + ymin
predict= (predict)*(ymax - ymin) + ymin
er    = np.abs(predict - yteste)/np.abs(yteste)
erroR = np.mean(er)

hist_error_rel(er,80,namep)

print('\n==============================================\n')
print('\nErro relativo..........: {:5.3}%'.format(erroR*100))
print('Erro relativo máximo...: {:5.3}%'.format(np.max(er)*100))
print('==============================================\n')

# Plot data ###################################################################
fig = plt.figure(constrained_layout=True)
fig, axs = plt.subplots(nrows=1,ncols=2, constrained_layout=True)
axs[0].plot(x[:,0],y,'tab:blue',linewidth=3)
axs[0].set_title('CamClay',fontsize=18)
axs[0].tick_params(axis="x", labelsize=16)
axs[0].tick_params(axis="y", labelsize=16)
axs[0].set_ylabel('$Void\ ratio$', fontsize=18, weight='bold', color='k')
axs[0].set_xlabel('$p_{mean}\ (MPa)$', fontsize=18, weight='bold', color='k')
#
axs[1].plot(xteste[:,0],predict,'bo',label='predict')
#axs[1].plot(xteste[:,0],yteste,'r+',label='true')
axs[1].plot(x[:,0],y,'tab:orange',linewidth=3,label='true data')
axs[1].set_title('CamClay',fontsize=18)
axs[1].tick_params(axis="x", labelsize=16)
axs[1].tick_params(axis="y", labelsize=16)
axs[1].set_ylabel('$Void\ ratio$', fontsize=18, weight='bold', color='k')
axs[1].set_xlabel('$p_{mean}\ (MPa)$', fontsize=18, weight='bold', color='k')
axs[1].legend(loc='upper right')    
name = 'figuras/camclay_datanorm.png'
plt.savefig(name, transparent=True, dpi=300)
plt.show()

##########################
fig = plt.figure(constrained_layout=True)
ax  = plt.axes(projection='3d')
ax.plot3D(x[:,0],x[:,1],y[:],'tab:orange', linewidth=1.5)
#ax.scatter(x[:,0],x[:,1],y[:],cmap='viridis', linewidth=0.5)
ax.scatter(xteste[:,0],xteste[:,1],predict, color='tab:blue', linewidth=0.5)
ax.set_xlabel('$p_{mean}\ (MPa)$')
ax.set_ylabel('$p_0\ (MPa)$')
ax.set_zlabel('$Void\ ratio$')
ax.view_init(5, -140)
name = 'figuras/camclay_network3d.png'
plt.savefig(name, transparent=True, dpi=300)
plt.show()
#ax.plot3D(xteste[:,0],xteste[:,1],predict,'r')
#ax.plot3D(x[:,0],x[:,1],y[:],'b')
