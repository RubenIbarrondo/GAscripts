##
## Title: Maximize number of ones
## Author: Rubén Ibarrondo
## Description:
##      

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
#          Maximize the number of 1's
#
# ============================================

def MaxOnes(populationNumber=100, generationNumber=100,
            pc=0.7, pm=0.001, stop=False):
    # pc: crossover probability
    # pm: mutation probability

    fitness = []
    fmax = []
# 1. Initial population
    population = []
    for c in range(populationNumber):
        chromosome = list(np.random.randint(0,2, 70))
        population.append(chromosome)

# 2. Calculate fitness function
    for g in range(generationNumber):
        fitness = []
        for c in population:
            fitness.append(c.count(1))
        fmax.append(max(fitness))
        
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
        if stop and fmax[-1] == len(population[0]):
            break

    return fitness, population, fmax

def Sweep_pm(pm_min=0.00, pm_max=1, pm_step=0.05, meanN = 10):
    # max value is found near pm=0.5
    pm = np.arange(pm_min,pm_max,pm_step)
    f =[]
    for pmi in pm:
        fi = 0
        for i in range(meanN):
            fi += len(MaxOnes(generationNumber=100, pm=pmi, stop=True)[2])/meanN
        f.append(fi)
    return list(pm), f

def Sweep_pc(pc_min=0.0, pc_max=1, pc_step=0.1, meanN=10):
    # the fitness value increases with pc
    # at rate approx 2 with pm=0.1
    pc = np.arange(pc_min,pc_max,pc_step)
    f =[]
    for pci in pc:
        fi = 0
        for i in range(meanN):
            fi += len(MaxOnes(generationNumber=100, pc=pci, stop=True)[2])/meanN
        f.append(fi)
    return list(pc), f
