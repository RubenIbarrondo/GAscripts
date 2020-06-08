##
## Title: Prisioner's Dilema
## Author: Rubén Ibarrondo
## Description:
##      1st trial with fitxed fitnes landscape
##      2nd trial with evolutionary fitness landscape

import numpy as np
import matplotlib.pyplot as plt
import plotTools
from time import time

def score(A,B):
    '''Score that each player gets after a game
    with result A,B. 1 means Cooperate and 0 Defect'''
    
    if   A == 1 and B == 1:
        return 3,3
    elif A == 1 and B == 0:
        return 0,5
    elif A == 0 and B == 1:
        return 5,0
    elif A == 0 and B == 0:
        return 1,1
    else:
        raise Exception("A or B not in set {0,1}")

# 
# The previous 3 games of each player are stored
# as 6 bit arrays. So the strategy to follow may
# be described by a 2**6 bit string, where each
# bit describes what to do for each stored 3 games.
#

def caseinfer(case):
    '''Returns the index for case. case must be an
    array like object of lenght 6 and which
    elements are in the set {0,1}.'''
    if len(case) != 6:
        raise Exception("case must have lenght 6.")
    if any(c != 0 and c != 1 for c in case):
        raise Exception("Elements in case must be 0 or 1.")
    return sum(case[5-i] * 2**i for i in range(len(case)))

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

## Some statistics
def standard_deviation(x):
    n = len(x)
    xmean = sum(x)/n
    dev = x - xmean
    var = sum(dev*dev)/(n-1)
    return np.sqrt(var)
    
def correlation(x,y):
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    meanx = sum(x)/n
    meany = sum(y)/n
    if stdev_x >0 and stdev_y>0:
        return np.dot(x-meanx, y-meany) /(n-1) /stdev_x/stdev_y
    else:
        return 0
##
    
# ======================
#
#    Fixed strategies
#
# ======================
#
#   1 means Cooperate
#   0 means Defect
#
#   case: 6 valued array,
#         with 0's and 1's
#   prev: if prev>=3 case is
#         taken into account.
#         if prev=0 then the
#         initial case is taken

def alwaysDefect(case, prev=3):
    return 0
def alwaysCooperate(case, prev=3):
    return 1
def titfortat(case, prev=3):
    if prev>=3:
        return case[5]
    else:
        return 1
def titfortat3(case, prev=3):
    if prev>=3:
        if case[1]+case[3]+case[5] > 1:
            return 1
        else:
            return 0
    else:
        return 1
def randomplayer(case, prev=3):
    return np.random.randint(low=0, high=2)
def methodic(case, prev=3):
    if prev>=3:
        if case[0] == 1 and case[3] == 1:
            return 1
        if case[2] == 1 and case[5] == 1:
            return 1
        else:
            return 0
    else:
        return 1
def punisher(case,prev=3):
    if prev>=3:
        if case[1]+case[3]+case[5] < 3:
            return 0
        else:
            return 1
    else:
        return 0
def rewarder(case,prev=3):
    if prev>=3:
        if case[1]+case[3]+case[5] >= 1:
            return 1
        else:
            return 0
    else:
        return 1

strategy_set = [ alwaysDefect, alwaysCooperate,
                 titfortat, titfortat3,
                 randomplayer, methodic,
                 punisher, rewarder]
strategy_names = [ 'alwaysDefect', 'alwaysCooperate',
                 'titfortat','titfortat3',
                 'randomplayer', 'methodic',
                 'punisher', 'rewarder']

# ============================================
#
#               Fixed strategies
#                  tournament
#
# ============================================

def FixedStrategyTournament(gameNumber=100):
    case_memory = np.random.randint(0,2,
                                    (len(strategy_set),
                                     len(strategy_set),6))
##    case_memory = np.zeros((len(strategy_set), len(strategy_set),6))
    case_memory = case_memory.tolist()
                
    score_memory = [0] * len(strategy_set)
    
    for n in range(gameNumber):
        for i in range(len(strategy_set)):
            for j in range(i+1,len(strategy_set)):
                a = strategy_set[i](case_memory[i][j])
                b = strategy_set[j](case_memory[j][i])
                
                case_memory[i][j] = case_memory[i][j][2:] + [a,b]
                case_memory[j][i] = case_memory[j][i][2:] + [b,a]
                s = score(a, b)
                score_memory[i] += s[0]/gameNumber/(len(strategy_set)-1)
                score_memory[j] += s[1]/gameNumber/(len(strategy_set)-1)
                
    return score_memory

def FixedStrategyTournament_Sweep(imax=20,gameNumber=100):
    smean = [0] * len(strategy_set)
    smean2 = [0] * len(strategy_set) 
    for i in range(imax):
        s = FixedStrategyTournament(gameNumber)
        smean = [sm + sp/imax for sm, sp in zip(smean,s)]
        smean2 = [sm2 + sp*sp/imax for sm2, sp in zip(smean2,s)]
    dev = [np.sqrt(sm2-sm*sm) for sm2, sm in zip(smean2, smean)]
    return smean, dev

# ============================================
#
#          Static fitness landscape GA
#           agains Fixed strategies
#
# ============================================

def FixedLandscapeGA(gameNumber=100, populationNumber=10,
                     generationNumber=100, pc=0.7, pm=0.001):
    # pc: crossover probability
    # pm: mutation probability

    # Simple selection
    ##selection_funtion = lambda fitness, fsum: [f/fsum for f in fitness]

    # Unfare selection
    selection_funtion = lambda fitness, fsum: [(f-min(fitness))/
                                               (fsum-len(fitness)*min(fitness))
                                               for f in fitness]

    fitness = []
    fmax = []
    # 1. Initial population
    population = []
    for g in range(populationNumber):
        chromosome = list(np.random.randint(0,2, 70))
        population.append(chromosome)

    for g in range(generationNumber):
        case_memory = np.random.randint(0,2,
                                         (len(population),
                                          len(strategy_set),6))
        case_memory = case_memory.tolist()

        # 2. Calculate fitness function
        score_memory = [0] * populationNumber
        for c in range(len(population)):
            for n in range(gameNumber): ## should these fors' order be changed?
                for i in range(len(strategy_set)):
                    a = population[c][caseinfer(case_memory[c][i])]
                    b = strategy_set[i](case_memory[c][i][::-1])
                    
                    case_memory[c][i] = case_memory[c][i][2:] + [a,b]
                    s = score(a, b)
                    score_memory[c] += s[0]/gameNumber/(len(strategy_set))
        fitness = score_memory
        fmax.append(max(fitness))
        
        # 3. Offspring creation
        offspring = []
        while len(offspring) < populationNumber:
            # 3.a Parent chromosome selection
            i,j = np.random.choice(range(len(population)), p=selection_funtion(fitness, sum(fitness)), size=2)
            ci, cj = population[i], population[j]
            
            # 3.b Apply crossover or not
            rc = np.random.random()
            if rc<pc:
                index = np.random.randint(len(ci))
                newci, newcj = crossover(ci,cj,index)
            else:
                newci, newcj = ci, cj

            # 3.c Apply mutation or not
            rm = np.random.random()
            if rm<pm:
                index = np.random.randint(len(ci))
                newci = mutation(ci, index)
                newcj = mutation(cj, index)
            
            offspring.append(newci)
            offspring.append(newcj)

            # This would be used in special cases
            while len(offspring)>populationNumber:
                index = np.random.randint(len(offspring))
                offspring.pop(index)
        population = offspring
        
    return fitness, fmax
    
# ============================================
#
#          Evolving fitness landscape GA
#
# ============================================

def Tournament(population, gameNumber):
    populationNumber = len(population)
    case_memory = np.random.randint(0,2,
                                    (len(population),len(population),6))
    case_memory = case_memory.tolist()

    score_memory = [0] * populationNumber
    for n in range(gameNumber):
        for ci in range(populationNumber):
            for cj in range(ci, populationNumber):
                if n < 3:
                    a = population[ci][caseinfer(population[ci][-6:])]
                    b = population[cj][caseinfer(population[cj][-6:])]
                else:
                    a = population[ci][caseinfer(case_memory[ci][cj])]
                    b = population[cj][caseinfer(case_memory[cj][ci])]
                
                case_memory[ci][cj] = case_memory[ci][cj][2:] + [a,b]
                case_memory[cj][ci] = case_memory[cj][ci][2:] + [b,a]
                s = score(a, b)
                score_memory[ci] += s[0]/gameNumber/(populationNumber)
                score_memory[cj] += s[1]/gameNumber/(populationNumber)
    return score_memory

def EvolvingLandscapeGA(gameNumber=100, populationNumber=20,
                     generationNumber=50, pc=0.7, pm=0.001):
    # pc: crossover probability
    # pm: mutation probability

    # Fitness−proportional selection
    selection_funtion = lambda fitness, fsum: [f/fsum for f in fitness]

    # Unfare selection
##    selection_funtion = lambda fitness, fsum: [(f-min(fitness))/
##                                               (fsum-len(fitness)*min(fitness))
##                                               for f in fitness]

    fitness = []
    fmax = []
    fmean = []
    fmin = []

    # 1. Initial population
    population = []
    for g in range(populationNumber):
        chromosome = list(np.random.randint(0,2, 70))
        population.append(chromosome)

    for g in range(generationNumber):
        
        # 2. Calculate fitness function
        fitness = Tournament(population, gameNumber)
        
        fmax.append(max(fitness))
        fmean.append(sum(fitness)/len(fitness))
        fmin.append(min(fitness))
        
        
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
            for index in range(len(ci)):
                rmi, rmj = np.random.random(size=2)
                if rmi<pm:
                    newci = mutation(newci, index)
                if rmj<pm:
                    newcj = mutation(newcj, index)
            
            offspring.append(newci)
            offspring.append(newcj)

            # This would be used in cases where
            # populationNumber was not even
            while len(offspring)>populationNumber:
                index = np.random.randint(len(offspring))
                offspring.pop(index)
        population = offspring
    return population, fitness, fmax, fmean, fmin

def EvolvingLandscapeGA_TestingRoutine(
    path='PrisonersDilemma_pc0_Tests/PrisionersDilema_pc0_Test',runNumber=10):
    names = ['Final Population', 'Final Fitness',
             'Maximun fitness evolution', 'Mean fitness evolution',
             'Minimum fitness evolution']
    for i in range(1, runNumber+1):
        t0 = time()
        result = EvolvingLandscapeGA(pc=0)
        t1 = time()
        file = open(path+str(i)+'.data', 'w')
        print('---------------------------------------', file=file) 
        print('PRISIONERS DILEMA with pc=0\n',file=file)
        print('    Run number:    {:16d}'.format(i), file=file)
        print("    Time duration: {:16.4f}".format(t1-t0), file=file)
        print('\n---------------------------------------\n', file=file)
        for element in range(len(result)):
            print(names[element] , file=file)
            print(result[element], file=file)
            print(file=file)
            print(file=file)
        file.close()

def ParseTestData(
    path='PrisonersDilemma_pc0_Tests/PrisionersDilema_pc0_Test', runNumber=10):
    population_list = []
    fitness_list = []
    fmax_list = []
    fmean_list = []
    fmin_list = []
    
    for i in range(1, runNumber+1):
        file = open(path+str(i)+'.data', 'r')
        file_lines = file.readlines()
        population_list.append(eval(file_lines[9][:-1]))
        fitness_list.append(eval(file_lines[13][:-1]))
        fmax_list.append(eval(file_lines[17][:-1]))
        fmean_list.append(eval(file_lines[21][:-1]))
        fmin_list.append(eval(file_lines[25][:-1]))
        file.close()
    return population_list, fitness_list, fmax_list, fmean_list, fmin_list
