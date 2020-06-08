##
## Title: Curve Fit
## Author: Rubén Ibarrondo
## Description:
##      Fits a curve for a
##  sample data.

import numpy as np
from bitstring import BitArray

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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
    return [f/fsum for f in fitness]

# ============================================
#
#        Problem specific functions
#
# ============================================

def decode(chrom):
    #chrom = BitArray( chrom)
    return chrom.int/ 2**27 # -16.0 - 16.0

def func1(x, a=10, b=0.5):
    return a*np.exp(-x*x/16) * np.sin(b*2*np.pi*x)+10

def func(x):
    return func2(x)

def func2(x):
    return (1+100*np.exp(-(x+2.5)**2*16))*np.sinc(x)**2
x = np.arange(-16, 16, 0.5)
y = np.arange(-16, 16, 0.5)
x, y = np.meshgrid(x,y)
def func2D(x, y):
    # [posx, posy, anplitude, dispersion]
    peaks = [[-5,5,5,3], [10,-10,-5,5],[8,-3,3,6],
             [-4,-2,-3,10],[2,10,4,4],[-15,15,4.2,50],
             [7,7,6,.5],[-10, -7, 5, 3]]
    result = 10
    for xi,yi , anp, dis in peaks:
        result += anp*np.exp(-((x-xi)**2+(y-yi)**2)/dis)
        #print("[{:7.2f},{:7.2f}] {:7.2f}, {:7.2f}, {:7.2f}".format( r[0], r[1], dis, anp,anp*np.exp(-((x-r[0])**2+(y-r[1])**2)/dis) ))
    result += .5*(x+y+x*y*y)/(x*x+y*y+1)
    return result

# ============================================
#
#          Find the curve fit
#
# ============================================

def FindCurveMax(populationNumber=100, generationNumber=100,
            pc=0.7, pm=0.001):
    # pc: crossover probability
    # pm: mutation probability

    fitness = []
    fmax = []
    fmean = []
    n = 32
    
# 1. Initial population
    population = []
    for c in range(populationNumber):
        chromosome = BitArray(list(np.random.randint(0,2, n*2)))
        population.append(chromosome)
        
# 2. Calculate fitness function
    for g in range(generationNumber):
        fitness = []
        for c in population:
            fitness.append(func2D(decode(c[:n]), decode(c[n:])))
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

    return fitness, population, fmax, fmean

def RunNtest(runNumber=10):
    xdata, ydata = gendata(2, 7)
    f = []
    p = []
    fmax = []
    fmean = []
    for i in range(runNumber):
        fi, pi, fmaxi, fmeani = FindCurveMax(xdata, ydata)
        f.append(fi)
        p.append(pi)
        fmax.append(fmaxi)
        fmean.append(fmeani)
    return f, p, fmax, fmean

def Sweep_pm(pm_min=0.000, pm_max=0.01, pm_step=0.001, meanN = 10):
    pm = np.arange(pm_min,pm_max,pm_step)
    fmax =[]
    fmean = []
    for pmi in pm:
        f,p, fx, fn = FindCurveMax(pm = pmi)
        fmax.append(fx)
        fmean.append(fn)
    return list(pm), fmax, fmean

def Sweep_pc(pc_min=0.0, pc_max=1, pc_step=0.1, meanN=10):
    pc = np.arange(pc_min,pc_max,pc_step)
    fmax =[]
    fmean = []
    for pci in pc:
        f,p, fx, fn = FindCurveMax(pc = pci)
        fmax.append(fx)
        fmean.append(fn)
    return list(pc), fmax, fmean

def Sweep_populationNumber(pN_min=10, pN_max=100, pN_step=10, meanN = 10):
    pN = np.arange(pN_min,pN_max,pN_step)
    fmax =[]
    fmean = []
    for pNi in pN:
        f,p, fx, fn = FindCurveMax(populationNumber = pNi)
        fmax.append(fx)
        fmean.append(fn)
    return list(pN), fmax, fmean

##
##
## Testing results
## !! not updated !!
##
##

def Maxfit_Meanfit1():
    f, p, fx, fn = FindCurveMax()
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
