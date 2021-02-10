# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import os


generalization = []
sharpness = []

path = os.getcwd()
dir_list = []
for i in os.listdir(path):
    if os.path.isdir(i) and 'attack' in i:
        dir_list.append(i)
dir_list.sort()

for dirs in dir_list:
    current_path = os.path.join(path, dirs)
    os.chdir(current_path)
    generalization.append(np.load('generalization_sample.npy'))
    sharpness.append(np.load('sharpness_sample.npy'))
    os.chdir(path)


volume = np.load('volume_list_no_attack.npy') * np.log10(np.e)
generalization = 100. * np.array(generalization).flatten()
sharpness = -1 * np.array(sharpness).flatten()

fig, ax = plt.subplots()
ax.scatter(sharpness,generalization)
ax.set_xlabel(r'-$log_{10}$(sharpness)',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.set_ylabel('Generalization (%)',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.text(0.6,0.25,'FCN/MNIST', bbox=dict(facecolor='none'), fontsize=20)
fig.savefig('sharpness-generalization-log10.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(volume,generalization)
ax.set_xlabel(r'$log_{10}V(f)$',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.set_ylabel('Generalization (%)',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.text(0.6,0.25,'FCN/MNIST', bbox=dict(facecolor='none'), fontsize=20)
fig.savefig('volume-generalization-log10.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(sharpness,volume)
ax.set_xlabel(r'-$log_{10}$(sharpness)',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.set_ylabel(r'$log_{10}V(f)$',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.text(0.6,0.25,'FCN/MNIST', bbox=dict(facecolor='none'), fontsize=20)
fig.savefig('sharpness-volume-log10.png', dpi=300)
plt.clf()
plt.close()



