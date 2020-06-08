##
## Title: Points and sufaces in mplib
## Author: Rubén Ibarrondo
## Description:
##      Surface + points.
##  Discovering the limitations
##  of matplotlib.

import numpy as np
from bitstring import BitArray

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.path as path

from mayavi import mlab
import  moviepy.editor as mpy


# ============================================
#
#                GA operators
#
# ============================================

def crossover(c1, c2, i):
    '''Returnes the crossovers at i of chromosomes
    c1 and c2.'''
    if len(c1) != len(c2):
        raise Exception("c1 and c2 must be same lenght")
    return c1[:i]+c2[i:], c2[:i]+c1[i:]

def mutation(c, i):
    '''Mutates the chromosome c at the ith position'''
    newc = c[:]
    newc[i] = c[i]^1
    return newc

def selection_probability(fitness, fsum):
    return fitness_proportional_probability(fitness, fsum)

def fitness_proportional_probability(fitness, fsum):
    return [f/fsum for f in fitness]

def fitness_enhanced_probability(fitness, fsum):
    fmin = min(fitness)
    f_enh = [f-fmin for f in fitness]
    f_enhsum = sum(f_enh)
    return [f/f_enhsum for f in f_enh]

# ============================================
#
#        Problem specific functions
#
# ============================================

def decode(chrom):
    #chrom = BitArray( chrom)
    return chrom.int/ 2**27 # -16.0 - 16.0

def func2D(x, y):
    # [posx, posy, anplitude, dispersion]
    peaks = [[-5,5,5,3], [10,-10,-5,5],[8,-3,3,6],
             [-4,-2,-3,10],[2,10,4,4],[-15,15,4.2,50],
             [7,7,6,.5],[-10, -7, 5, 3]]
    result = 10
    #print("[{:>7s},{:>7s}] {:>7s}, {:>7s}, {:>7s}".format('xi','yi','dis','anp','exp contr'))
    for xi,yi , anp, dis in peaks:
        result += anp*np.exp(-((x-xi)**2+(y-yi)**2)/dis)
        #print("[{:7.2f},{:7.2f}] {:7.2f}, {:7.2f}, {:7.2f}".format( xi, yi, dis, anp,anp*np.exp(-((x-xi)**2+(y-yi)**2)/dis) ))
    result += .5*(x+y+x*y*y)/(x*x+y*y+1)
    #result = 1-(x**2 + y**2)
    return result

# ============================================
#
#          Find the curve fit
#
# ============================================

def FindSurfaceMax(populationNumber=20, generationNumber=100,
            pc=0.7, pm=0.001):
    # pc: crossover probability
    # pm: mutation probability

    fitness = []
    fmax = []
    fmean = []
    points = []
    n = 32
    
# 1. Initial population
    population = []
    for c in range(populationNumber):
        chromosome = BitArray(list(np.random.randint(0,2, n*2)))
        population.append(chromosome)
        
# 2. Calculate fitness function
    for g in range(generationNumber):
        fitness = []
        pnts = []
        for c in population:
            fitness.append(func2D(decode(c[:n]), decode(c[n:])))
            pnts.append([decode(c[:n]), decode(c[n:])])
        points.append(pnts)
        fmax.append(max(fitness))
        fmean.append(sum(fitness)/len(fitness))
        
# 3. Offspring creation
        offspring = []

        while len(offspring) < populationNumber:
# 3.a Parent chromosome selection
            try:
                i,j = np.random.choice(range(len(population)),
                                       p=selection_probability(fitness, sum(fitness)),
                                       size=2)
            except:
                print(fitness)
            ci, cj = population[i], population[j]
            
# 3.b Apply crossover or not
            rc = np.random.random()
            if rc<pc:
                index = np.random.randint(len(ci))
                newci, newcj = crossover(ci,cj,index)
            else:
                newci, newcj = ci[:], cj[:]

# 3.c Apply mutation or not
            for index in range(len(cj)):
                rm = np.random.random()
                if rm<pm:
                    newci = mutation(newci, index)
                    newcj = mutation(newcj, index)
            
            offspring.append(newci)
            offspring.append(newcj)

# This would be used in special cases
            while len(offspring)>populationNumber:
                index = np.random.randint(len(offspring))
                offspring.pop(index)
        population = offspring

    return fitness, population, fmax, fmean, points
