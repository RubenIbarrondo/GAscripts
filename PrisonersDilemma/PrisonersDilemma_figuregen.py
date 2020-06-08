##
## Title: Prisioners Dilemma Figures
## Author: Rubén Ibarrondo
## Description:

import numpy as np

from PrisionersDilema import *
import PD_behaviour_analysis as pda
import plotTools

import matplotlib.pyplot as plt
from matplotlib import cm

#import matplotlib.animation as animation
#import matplotlib.patches as patches
#import matplotlib.path as path

#from mayavi import mlab
#import  moviepy.editor as mpy

# ============================================
#
#               Figure generator
#
# ============================================

'''GENERAL EXAMPLE: DATA'''
pa, fa, fxa, fma, fna = ParseTestData(path='PrisonersDilemma_Tests/PrisionersDilema_Test')
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

Np = len(pa[0])
fa = np.array(fa) * Np/(Np+1)
fxa = np.array(fxa) * Np/(Np+1)
fma = np.array(fma) * Np/(Np+1)
fna = np.array(fna) * Np/(Np+1)

iscoop = np.zeros(len(pa), dtype=int)
coop_name = ['NC', 'C']

for fx, i in zip(fxa, range(len(fxa))):
    if fx[-1] > 2.99:
        iscoop[i] = 1

#### Crossed tournament data generator
##points = np.zeros((len(pa), len(pa)))
##for i in range(len(pa)):
##    for j in range(i, len(pa)):
##        pi = pa[i]
##        pj = pa[j]
##        score, corate = pda.Tournament_behaviour(pi+pj, 100)
##        si = sum(score[:len(pi)])/len(pi)
##        sj = sum(score[len(pi):])/len(pj)
##        points[i,j] = sj
##        points[j,i] = si
##        #print("{:d},{:d}   {:5.2f}, {:5.2f},  corrate: {:5.2f}".format(i,j, si, sj, corate))
points = np.array([[1.775    , 2.4676125, 2.3745   , 2.2251   , 1.847975 , 2.4301875,
        1.668075 , 3.234425 , 2.115175 , 2.480425 ],
       [2.069025 , 3.       , 3.       , 2.4397625, 2.4767875, 3.       ,
        2.3621625, 3.       , 2.106625 , 3.       ],
       [1.8421   , 3.       , 3.       , 1.9840625, 1.8433   , 3.       ,
        2.1498625, 3.       , 2.3480875, 3.       ],
       [2.01695  , 2.3957625, 2.7214375, 2.284    , 1.7248625, 2.822225 ,
        1.861425 , 2.59925  , 2.4494625, 2.145    ],
       [2.52145  , 2.8996625, 3.0943   , 2.8556125, 2.1      , 2.9265875,
        2.2002375, 3.209    , 2.637775 , 2.28625  ],
       [2.0571   , 3.       , 3.       , 2.0166   , 1.9074   , 3.       ,
        2.089725 , 3.       , 2.04555  , 3.       ],
       [2.2516   , 2.7318375, 2.793725 , 2.467975 , 2.29035  , 2.5169   ,
        2.081825 , 2.505    , 2.229675 , 2.7400625],
       [1.273025 , 3.       , 3.       , 1.6169375, 1.494    , 3.       ,
        2.069075 , 3.       , 2.119375 , 3.       ],
       [1.6385   , 2.3189125, 2.36575  , 1.8238125, 1.63925  , 2.3783375,
        1.801725 , 2.413725 , 2.12765  , 2.209625 ],
       [2.1304   , 3.       , 3.       , 2.654    , 2.515    , 3.       ,
        1.8117   , 3.       , 2.5394625, 3.       ]])

rancking = [ (sum(points[:,i])/len(pa), i) for i in range(len(pa))]
rancking.sort(reverse=True)
devs = [np.sqrt(sum((points[:,i]-r)**2)/len(pa))
        for r, i in rancking]
rancking = np.array(rancking)

'''Fitness evolution of all the GAs'''
##c = 0
##
##plt.grid(axis='y', linestyle='dashed')
##plt.ylim((1,5))
##plt.ylabel("$f_i$")
##plt.xlabel("generation")
##plt.yticks(ticks=[i/10 for i in range(0,55,5)], labels=[i/10 for i in range(0,55,5)])
##
##for fx, fm, fn in zip(fxa, fma, fna):
##    if fx[-1] >2.99:
##        labelfmt = '{:d} {:2s}'.format(c, 'C')
##    else:
##        labelfmt = '{:d} {:2s}'.format(c, 'NC')
##    plt.plot(fx, color=colors[c], label=labelfmt)
##    #plt.plot(fm, color=colors[c])
##    #plt.plot(fn, color=colors[c])
##    c+=1
##plt.legend(loc='upper right', ncol=3)
##plt.show()

'''Fitness evolution of the cooperating GAs'''
##c = 0
##
##plt.grid(axis='y', linestyle='dashed')
##plt.ylim((1,5))
##plt.ylabel("$f_i$")
##plt.xlabel("generation")
##plt.yticks(ticks=[i/10 for i in range(0,55,5)], labels=[i/10 for i in range(0,55,5)])
##
##for fx, fm, fn in zip(fxa, fma, fna):
##    if fx[-1] >2.99:
##        plt.plot(fx, color=colors[c], label='{:d} {:2s}'.format(c, 'C'))
##    else:
##        plt.plot(fx, color=colors[c], alpha=0.2, label='{:d} {:2s}'.format(c, 'NC'))
##    #plt.plot(fm, color=colors[c])
##    #plt.plot(fn, color=colors[c])
##    c+=1
##plt.legend(loc='upper right', ncol=3)
##plt.show()

'''Fitness evolution of the non cooperating GAs'''
##c = 0
##
##plt.grid(axis='y', linestyle='dashed')
##plt.ylim((1,5))
##plt.ylabel("$f_i$")
##plt.xlabel("generation")
##plt.yticks(ticks=[i/10 for i in range(0,55,5)], labels=[i/10 for i in range(0,55,5)])
##
##for fx, fm, fn in zip(fxa, fma, fna):
##    if fx[-1] <2.99:
##        plt.plot(fx, color=colors[c], label='{:d} {:2s}'.format(c, 'NC'))
##    else:
##        plt.plot(fx, color=colors[c], alpha=0.2, label='{:d} {:2s}'.format(c, 'C'))
##    #plt.plot(fm, color=colors[c])
##    #plt.plot(fn, color=colors[c])
##    c+=1
##plt.legend(loc='upper right', ncol=3)
##plt.show()

'''Final populations' tournament: Rancking'''
##values = rancking[:,0]
##names = ['{:d} {:2s}'.format(int(i), coop_name[iscoop[int(i)]]) for i in rancking[:,1]]
##error = devs
##xlabel = 'Score'
##title = None
##
##plt.rcdefaults()
##fig, ax = plt.subplots()
##
##y_pos = np.arange(len(values))
##
##ax.barh(y_pos, values, xerr=error, align='center')
##ax.set_yticks(y_pos)
##if names:
##    ax.set_yticklabels(names)
##ax.invert_yaxis()  # labels read top-to-bottom
##ax.set_xlabel(xlabel)
##ax.set_title(title)
##ax.set_zorder(1)
###ax.grid(axis='x', linestyle='dashed')
##ax.xaxis.grid(True, linestyle='--', which='major',
##                   color='grey', alpha=.5)
##plt.show()

'''Final population's tournament: Score table'''
# Score of columns vs rows
##names = ['{:d} {:2s}'.format(int(i), coop_name[iscoop[int(i)]]) for i in range(len(pa))]
##againstnames = [ name for name in names]
##
##fig, ax = plt.subplots()
##
##im, cbar = plotTools.heatmap(data=points, row_labels=againstnames, col_labels=names,
##                             ax=ax, cmap="YlGn", cbarlabel="Score")
##texts = plotTools.annotate_heatmap(im, valfmt="{x:.1f}")
##
##fig.tight_layout()
##plt.show()

'''Final population's tournament: Score table ORDERED'''
# Score of columns vs rows
new_order = [7, 2, 5, 1, 9, 8, 3, 6, 4, 0]

names = np.array(['{:d} {:2s}'.format(int(i), coop_name[iscoop[int(i)]]) for i in range(len(pa))])
againstnames = np.array([ name for name in names])

new_names = names[new_order]
new_againstnames = againstnames[new_order]
new_points = points[new_order]
new_points = new_points[:, new_order]

fig, ax = plt.subplots()

im, cbar = plotTools.heatmap(data=new_points, row_labels=new_againstnames, col_labels=new_names,
                             ax=ax, cmap="YlGn", cbarlabel="Score")
texts = plotTools.annotate_heatmap(im, valfmt="{x:.1f}")

fig.tight_layout()
plt.show()

