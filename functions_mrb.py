#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 09:47:26 2021

@author: mrborges
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker


def diagnostic_graphs(history,namep,loss,metric,vloss,vmetric):
    # Plot training & validation accuracy values
    ya = min(min(history.history[metric]),min(history.history[vmetric]))*0.95
    yb = max(max(history.history[metric]),max(history.history[vmetric]))*1.05
    dy = (yb-ya)/4
    axy = np.arange(ya,yb+1e-5,dy)
    font = font_manager.FontProperties(family='serif', 
                                       weight='bold', style='normal', size=12)
    plt.figure(constrained_layout=True)
    plt.plot(history.history[metric],'tab:blue',linewidth=3)
    plt.plot(history.history[vmetric],'tab:orange',linewidth=3)
    plt.title('Model Loss',fontsize=18)
#    plt[0].set(ylabel='Accuracy', xlabel='Epoch')
    plt.legend(['Train', 'Test'], loc='upper right',prop=font)
    plt.ylim((ya,yb))
    plt.yticks(axy)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.xlabel('Epoch', fontsize=18, weight='bold', color='k')
    plt.ylabel('Mean Squared Error', fontsize=18, weight='bold', color='k')
    name = namep + 'error.png'
    plt.savefig(name, transparent=True, dpi=300)
    
    plt.show()
    
def diagnostic_graphs2(history,namep,loss,metric,vloss,vmetric):
    # Plot training & validation accuracy values
    ya = min(min(history.history[metric]),min(history.history[vmetric]))*0.95
    yb = max(max(history.history[metric]),max(history.history[vmetric]))*1.05
    dy = (yb-ya)/4
    axy = np.arange(ya,yb+1e-5,dy)
    font = font_manager.FontProperties(family='serif', 
                                       weight='bold', style='normal', size=12)
    fig = plt.figure(constrained_layout=True)
    fig, axs = plt.subplots(nrows=1,ncols=2, constrained_layout=True)
    axs[0].plot(history.history[metric],'tab:blue',linewidth=3)
    axs[0].plot(history.history[vmetric],'tab:orange',linewidth=3)
    axs[0].set_title('Model Loss',fontsize=18)
#    axs[0].set(ylabel='Accuracy', xlabel='Epoch')
    axs[0].legend(['Train', 'Test'], loc='upper right',prop=font)
    axs[0].set_ylim((ya,yb))
    axs[0].set_yticks(axy)
    axs[0].tick_params(axis="x", labelsize=16)
    axs[0].tick_params(axis="y", labelsize=16)
    axs[0].set_xlabel('Epoch', fontsize=18, weight='bold', color='k')
    axs[0].set_ylabel('Mean Squared Error', fontsize=18, weight='bold', color='k')
    axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1e'))

    # Plot training & validation loss values
    ya = 1e-8
    yb = 0.1
    axy = np.concatenate((np.arange(yb,yb+1,1),1.0/10**np.arange(1,7,1)))
    ya = min(axy)
    axs[1].set_yscale('log')
    axs[1].plot(history.history['loss'],'tab:blue',linewidth=3)
    axs[1].plot(history.history['val_loss'],'tab:orange',linewidth=3)
    axs[1].set_yticks(axy)
    axs[1].set_title('Model loss',fontsize=18)
#    axs[1].set(ylabel='Loss', xlabel='Epoch')
    axs[1].legend(['Train', 'Valid'], loc='upper right',prop=font)
#    axs[1].set_ylim((ya,yb))
    axs[1].tick_params(axis="x", labelsize=16)
    axs[1].tick_params(axis="y", labelsize=16)
    axs[1].set_xlabel('Epoch', fontsize=18, weight='bold', color='k')
    axs[1].set_ylabel('Log Mean Squared Error' , fontsize=18, weight='bold', color='k')

    name = namep + 'error.png'
    plt.savefig(name, transparent=True, dpi=300)
    
    plt.show()
    
# Histograma
def hist_error_rel(a,nbins,namep):
    y, bins, ignored = plt.hist(a, bins=nbins, range=None, density=True,
             weights=None, cumulative=False, bottom=None, histtype='bar', 
             align='mid', orientation='vertical', rwidth=None, 
             log=False, color=None, label=None, stacked=False)

    media = np.mean(a)
    sigma = np.std(a)
    z = np.asarray(y)
    sz = z.size
    ymax = -10e30
    for i in range(sz-1):
        if ymax < np.max(z[i]):
            ymax = np.max(z[i])

    maximo = media + 4*sigma
    minimo = 0.0#media - 2*sigma

    ymax   = max(y)*1.2
    inform = '$\mu=' + format(media, '.2e') +',\ \sigma=' + format(sigma, '.2e') + '$'
    plt.text(1e-4, 0.9*ymax, inform,
             fontsize=16,color='black')
    plt.title("Histogram of relative error", fontsize=16)
    plt.ylabel('Freq.', fontsize=16, weight='bold', color='k');
    plt.xlabel('Relative error', fontsize=18, weight='bold', color='k');
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.ylim((0,ymax))
    plt.xlim((minimo,maximo))
    # show the figure
    name = namep + 'error_hist.png'
    plt.savefig(name, transparent=True, dpi=300)
    plt.show()
