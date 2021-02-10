# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


train_loss = np.load('train_loss_list_1.npy')
train_accu = np.load('train_accuracy_list_1.npy')
genuine_train_loss = np.load('genuine_train_loss_list_1.npy')
genuine_train_accu = np.load('genuine_train_accuracy_list_1.npy')
test_loss = np.load('test_loss_list_1.npy')
test_accu = np.load('generalization_list_1.npy')

sharpness = np.load('sharpness_list_1.npy')

max_values = np.load('max_value_list_1.npy')
volume = np.load('volume_list.npy')
volume = volume * np.log10(np.e)

###############################

fig = plt.figure(figsize=(7,4.8))
ax1 = fig.add_subplot(111)
ax1.plot(volume,label='volume')
ax1.legend(loc=1)

ax1.vlines(len(train_loss)-501, -160, 5, colors = "r", linestyles = "dashed")
ax1.vlines(len(train_loss)-1000, -160, 5, colors = "g", linestyles = "dashed")

ax1.set_xlabel('Epochs',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax1.set_ylabel(r'$log_{10}V(f)$',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax1.set_xlim(0,len(train_loss))
ax1.set_ylim(-160,5)

ax2 = ax1.twinx()
ax2.plot(sharpness,label='sharpness',color = 'y')
ax2.legend(loc=5)
ax2.set_ylabel(r'$log_{10}$(sharpness)',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax1.tick_params(direction='in')
ax2.tick_params(direction='in')

axins = zoomed_inset_axes(ax1, 14, loc='upper left',
                          bbox_to_anchor=(0.235, 0.975),bbox_transform=ax1.figure.transFigure)  # zoom = 6
axins.plot(volume)
axins.vlines(len(train_loss)-501, -360*np.log10(np.e), -300*np.log10(np.e),
             colors = "r", linestyles = "dashed")
axins.set_xlabel('Epochs', fontdict={'fontsize': 12, 'fontweight': 'medium'})

axins.set_xlim(635, 665)
axins.set_ylim(-353*np.log10(np.e), -345*np.log10(np.e))
axins.set_xticks([640,650,660])
axins.set_yticks([])
axins.xaxis.set_minor_locator(AutoMinorLocator())
axins.tick_params(direction='in',which='both')

mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.savefig('sharpness-volume.png', dpi=300)

