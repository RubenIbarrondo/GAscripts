##
## Title: Maximize the integer
## Author: Rubén Ibarrondo
## Description:
##      Maximizes the integer
##  representation in binary
##  string.

import numpy as np
import matplotlib.pyplot as plt
import plotTools

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

# Simple selection
## def selection_funtion(fitness, fsum):
##    return [f/fsum for f in fitness]

# Unfare selection
def selection_funtion(fitness, fsum):
    if min(fitness)!=max(fitness):
        return [(f-min(fitness))/
                (fsum-len(fitness)*min(fitness))
                for f in fitness]
    else:
        return [1/len(fitness)] * len(fitness)
    
# ============================================
#
#           Maximize the integer
#
# ============================================

def MaxInteger(populationNumber=100, generationNumber=100,
            pc=0.7, pm=0.001, stop=False):
    # pc: crossover probability
    # pm: mutation probability

    fitness = []
    fmax = []
    fmean = []
# 1. Initial population
    population = []
    for c in range(populationNumber):
        chromosome = list(np.random.randint(0,2, 20))
        population.append(chromosome)

# 2. Calculate fitness function
    for g in range(generationNumber):
        fitness = []
        for c in population:
            fitness.append(sum(c[-i]*2**i for i in range(len(c))))
        fmax.append(max(fitness))
        fmean.append(sum(fitness)/len(fitness))
        
# 3. Offspring creation
        offspring = []
        while len(offspring) < populationNumber:
# 3.a Parent chromosome selection
            i,j = np.random.choice(range(len(population)),
                                   p=selection_funtion(fitness, sum(fitness)),
                                   size=2)
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

    return fitness, population, fmax, fmean

def Sweep_pm(pm_min=0.000, pm_max=0.01, pm_step=0.001, meanN = 10):
    pm = np.arange(pm_min,pm_max,pm_step)
    fmax =[]
    fmean = []
    for pmi in pm:
        f,p, fx, fn = MaxInteger(pm = pmi)
        fmax.append(fx)
        fmean.append(fn)
    return list(pm), fmax, fmean

def Sweep_pc(pc_min=0.0, pc_max=1, pc_step=0.1, meanN=10):
    pc = np.arange(pc_min,pc_max,pc_step)
    fmax =[]
    fmean = []
    for pci in pc:
        f,p, fx, fn = MaxInteger(pc = pci)
        fmax.append(fx)
        fmean.append(fn)
    return list(pc), fmax, fmean

def Sweep_populationNumber(pN_min=10, pN_max=100, pN_step=10, meanN = 10):
    pN = np.arange(pN_min,pN_max,pN_step)
    fmax =[]
    fmean = []
    for pNi in pN:
        f,p, fx, fn = MaxInteger(populationNumber = pNi)
        fmax.append(fx)
        fmean.append(fn)
    return list(pN), fmax, fmean

##
##
## Testing results
##
##

def Maxfit_Meanfit1():
    f, p, fx, fn = MaxInteger()
    fx = np.array(fx)
    fn = np.array(fn)

    plt.plot(np.log(fx)/np.log(2))
    plt.plot(np.log(fn)/np.log(2))
    plt.show()

def Maxfit_Meanfit_pNvar():
    p, fx, fn = Sweep_populationNumber()
    # open in different subfigures
    # use log2 scale
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9, 6))#,
                            #subplot_kw={'xticks': [], 'yticks': []})
    for ax, i in zip(axs.flat, range(len(p))):
        ax.plot(np.log2(np.array(fx[i])))
        ax.plot(np.log2(np.array(fn[i])))
        ax.set_title(str(p[i]))

    plt.tight_layout()
    plt.show()
        
def Maxfit_Meanfit_pcvar():
    p, fx, fn = Sweep_pc()
    # open in different subfigures
    # use log2 scale
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9, 6))#,
                            #subplot_kw={'xticks': [], 'yticks': []})
    for ax, i in zip(axs.flat, range(len(p))):
        ax.plot(np.log2(np.array(fx[i])))
        ax.plot(np.log2(np.array(fn[i])))
        ax.set_title(str(p[i]))

    plt.tight_layout()
    plt.show()

def Maxfit_Meanfit_pmvar():
    p, fx, fn = Sweep_pm()
    # open in different subfigures
    # use log2 scale
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9, 6))#,
                            #subplot_kw={'xticks': [], 'yticks': []})
    for ax, i in zip(axs.flat, range(len(p))):
        ax.plot(np.log2(np.array(fx[i])))
        ax.plot(np.log2(np.array(fn[i])))
        ax.set_title(str(p[i]))

    plt.tight_layout()
    plt.show()
