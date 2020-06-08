##
## Title: Prisioner's Dilema
## Author: Rubén Ibarrondo
## Description:
##      Behaviour analysis

import numpy as np
import matplotlib.pyplot as plt
from PrisionersDilema import *
import plotTools
from time import time

# ============================================
#
#          Evolving fitness landscape GA
#
# ============================================

def Tournament_behaviour(population, gameNumber):
    populationNumber = len(population)
    case_memory = np.random.randint(0,2,
                                    (len(population),len(population),6))
    case_memory = case_memory.tolist()

    score_memory = [0] * populationNumber
    corate = 0
    m = 0
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
                if ci!=cj:
                    score_memory[ci] += s[0]/gameNumber/(populationNumber)
                    score_memory[cj] += s[1]/gameNumber/(populationNumber)
                else:
                    score_memory[ci] += s[0]/gameNumber/(populationNumber)/2
                    score_memory[cj] += s[1]/gameNumber/(populationNumber)/2
                m +=1
                if a== 1 and b ==1:
                    corate +=1
                    
    return score_memory, corate/m

def EvolvingLandscapeGA_behaviour(gameNumber=100, populationNumber=20,
                     generationNumber=50, pc=0.7, pm=0.001):
    # pc: crossover probability
    # pm: mutation probability

    # Fitness−proportional selection
    selection_funtion = lambda fitness, fsum: [f/fsum for f in fitness]

    fitness = []
    fmax = []
    fmean = []
    fmin = []
    corates = []
    total_corate = []
    coco_rate = []
    deco_rate = []

    # 1. Initial population
    population = []
    for g in range(populationNumber):
        chromosome = list(np.random.randint(0,2, 70))
        population.append(chromosome)

    for g in range(generationNumber):
        
        # 2. Calculate fitness function
        fitness, corate = Tournament_behaviour(population, gameNumber)
        
        fmax.append(max(fitness))
        fmean.append(sum(fitness)/len(fitness))
        fmin.append(min(fitness))
        corates.append(corate)
        total_corate.append(sum(sum(c)/len(c) for c in population)/len(population))
        coco_rate.append(sum(sum(c[1::2])/len(c[1::2]) for c in population)/len(population))
        deco_rate.append(sum(sum(c[::2])/len(c[::2]) for c in population)/len(population))
        
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
    return population, fitness, fmax, fmean, fmin, corates, total_corate, coco_rate, deco_rate


def EvolvingLandscapeGA_TestingRoutine(
    path='PrisionersDilema_behaviuor_Tests/PrisionersDilema_behaviuor_Test',runNumber=10):
    names = ['Final Population', 'Final Fitness',
             'Maximun fitness evolution', 'Mean fitness evolution',
             'Minimum fitness evolution', 'Cooperation rate evolution',
             'Strategy cooperation expectation', 'Co-Co strategy rate',
             'De-Co strategy rate']
    for i in range(1, runNumber+1):
        t0 = time()
        result = EvolvingLandscapeGA_behaviour()
        t1 = time()
        file = open(path+str(i)+'.data', 'w')
        print('---------------------------------------', file=file) 
        print('PRISIONERS DILEMA\n',file=file)
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
    path='PrisionersDilema_behaviuor_Tests/PrisionersDilema_behaviuor_Test', runNumber=10):
    population_list = []
    fitness_list = []
    fmax_list = []
    fmean_list = []
    fmin_list = []
    corate_list = []
    tot_corate = []
    coco_rate = []
    deco_rate = []
    
    for i in range(1, runNumber+1):
        file = open(path+str(i)+'.data', 'r')
        file_lines = file.readlines()
        population_list.append(eval(file_lines[9][:-1]))
        fitness_list.append(eval(file_lines[13][:-1]))
        fmax_list.append(eval(file_lines[17][:-1]))
        fmean_list.append(eval(file_lines[21][:-1]))
        fmin_list.append(eval(file_lines[25][:-1]))
        corate_list.append(eval(file_lines[29][:-1]))
        tot_corate.append(eval(file_lines[33][:-1]))
        coco_rate.append(eval(file_lines[37][:-1]))
        deco_rate.append(eval(file_lines[41][:-1]))
        file.close()
    return population_list, fitness_list, fmax_list, fmean_list, fmin_list, corate_list, tot_corate, coco_rate, deco_rate
