from copy import deepcopy
import numpy as np
from numpy.random import random_sample
from miscellaneous.sampling import haltonsampling
from miscellaneous.optim_support import SBX, mutation


def uncGA (fitnessfcn, lb, ub, opt,disp=False,**kwargs):
    # num only required for multi-objective fcn
    if isinstance(ub, int) or isinstance(ub, float):
        nvar = 1
        ub = np.array([ub])
        lb = np.array([lb])
    else:
        nvar = len(ub)
    npop = kwargs.get('npop', 150)    #Default number of initial population
    maxg = kwargs.get('maxg', 250)      #maximum generation
    pmut = 0.1     #mutation probability
    pcross = 0.95   #crossover probability
    history  = np.zeros(shape=[maxg,2]) #ask Kemas for explanation
    num = kwargs.get('num',None)


    #Initialize population
    samplenorm = haltonsampling.halton(nvar, npop)
    population = np.zeros(shape=[npop,nvar+1])
    for i in range(0, npop):
        for j in range(0, nvar):
            population[i, j] = (samplenorm[i, j] * (ub[j] - lb[j])) + lb[j]
            # for i in range (0,npop):
            #     for j in range (0,nvar):
            #         population[i,j] = lb[j] + (ub[j]-lb[j])*random_sample()
        if num == None:
            temp= fitnessfcn(population[i, 0:nvar])
        else:
            temp= fitnessfcn(population[i,0:nvar],num)
        population[i,nvar] = deepcopy(temp)

    #Evolution loop
    generation = 1
    oldFitness = 0
    while generation <= maxg:
        # for generation 1:1
        tempopulation = deepcopy(population)

        #Tournament Selection
        matingpool = np.zeros(shape=[npop,nvar])
        for kk in range (0,npop):
            ip1 = int(np.ceil(npop*random_sample())) #random number 1
            ip2 = int(np.ceil(npop*random_sample())) #random number 2
            while ip1 >= npop or ip2 >=npop:
                ip1 = int(np.ceil(npop * random_sample()))
                ip2 = int(np.ceil(npop * random_sample()))
            if ip2 == ip1: #In case random number 1 = random number 2
                while ip2 == ip1 or ip2>=npop:
                    ip2 = int(np.ceil(npop*random_sample()))

            lst  = np.arange(0,nvar)
            Ft1  = population[ip1,lst]
            Ft2  = population[ip2,lst]
            Fit1 = population[ip1,nvar]
            Fit2 = population[ip2,nvar]

            #Switch case, in Python we use if and elif instead of switch-case
            if opt == "max":
                if Fit1>Fit2:
                    matingpool [kk,:] = Ft1
                else :
                    matingpool [kk,:] = Ft2
            elif opt == "min":
                if Fit1<Fit2:
                    matingpool [kk,:] = Ft1
                else :
                    matingpool [kk,:] = Ft2
            else:
                pass


        #Crossover with tournament seelection
        child = np.zeros(shape=[2,nvar])
        lst = np.arange(0, nvar)
        for jj in range (0,npop,2):
            idx1 = int(np.ceil(npop*random_sample()))
            idx2 = int(np.ceil(npop*random_sample()))
            while idx1 >= npop or idx2 >= npop or idx1==idx2:
                idx1 = int(np.ceil(npop * random_sample()))
                idx2 = int(np.ceil(npop * random_sample()))
            if (random_sample() < pcross):
                child = SBX.SBX(matingpool[idx1, :], matingpool[idx2, :], nvar, lb, ub)
                tempopulation[jj,0:nvar] = child [0,:]
                tempopulation[jj+1,0:nvar] = child [1,:]
            else:
                tempopulation[jj, 0:nvar] = matingpool [idx1,:]
                tempopulation[jj + 1, 0:nvar] = matingpool [idx2,:]
            if num == None:
                tempopulation[jj, nvar]= fitnessfcn(tempopulation[jj, lst])
                tempopulation[jj + 1, nvar]= fitnessfcn(tempopulation[jj + 1, lst])
            else:
                tempopulation[jj,nvar]= fitnessfcn(tempopulation[jj,lst],num)
                tempopulation[jj+1,nvar]= fitnessfcn(tempopulation[jj+1,lst],num)

        #Combined Population for Elitism
        compopulation = np.vstack((population,tempopulation))

        #Sort Population based on their fitness value
        if opt == 'max':
            i = np.argsort(compopulation[:,nvar]) [::-1]
            compopulation = compopulation[i,:]
        elif opt == 'min':
            i = np.argsort(compopulation[:, nvar])
            compopulation = compopulation[i,:]

        #Record Optimum Solution
        bestFitness = compopulation[0,nvar]
        bestx   = compopulation[0,0:nvar]

        #Mutation
        for kk in range (1,(2*npop)):
            compopulation[kk,0:nvar] = mutation.gaussmut(compopulation[kk, 0:nvar], nvar, pmut, ub, lb)
            if num == None:
                compopulation[kk, nvar]= fitnessfcn(compopulation[kk, 0:nvar])
            else:
                compopulation[kk,nvar]= fitnessfcn (compopulation[kk,0:nvar],num)

        history[generation-1,0]=generation
        history[generation-1,1]=bestFitness

        fiterr =  100*(abs(bestFitness-oldFitness))/bestFitness
        if disp:
            print("Done, generation ", generation, " | Best X = ", bestx, " | Fitness Error (%)= ", fiterr)
        generation = generation+1
        if fiterr <= 10**(-2) and generation >= 50:
            break

        oldFitness = bestFitness
        #Next Population
        for i in range (0,npop):
            population[i,:] = compopulation[i,:]


    #Show Best Fitness and Design Variables
    # print("Best Fitness = ",bestFitness)
    # for i in range (0,nvar):
    #     print("X",i+1," = ",bestx[i])

    return (bestx,bestFitness,history)

def GAEEI (fitnessfcn, lb, ub, opt,num):
    nvar = len(ub)
    npop = 100      #number of initial population
    maxg = 100      #maximum generation
    pmut = 0.1     #mutation probability
    pcross = 0.95   #crossover probability
    history  = np.zeros(shape=[maxg,2]) #ask Kemas for explanation

    #Initialize population
    population = np.zeros(shape=[npop,nvar])
    # samplenorm = haltonsampling.halton(nvar, npop)
    # for i in range(0, npop):
    #     for j in range(0, nvar):
    #         population[i, j] = (samplenorm[i, j] * (ub[j] - lb[j])) + lb[j]
    for i in range (0,npop):
        for j in range (0,nvar):
            population[i,j] = lb[j] + (ub[j]-lb[j])*random_sample()

    #Evolution loop
    generation = 1
    while generation <= maxg:
        # Defining first fitness
        x = population[0,:]
        expimp,_,_ = fitnessfcn(x,num)
        fit = np.zeros(shape=npop)
        fit[0] = expimp
        bestindex = 1
        bestFitness = fit[0]
        bestx = x

        for i in range (1,npop):
            x = population[i,:]
            expimp, _, _ = fitnessfcn(x, num)
            fit[i] = expimp
            if fit[i] > bestFitness:
                bestFitness = fit[i]
                bestindex = i
                bestx = x

        # for generation 1:1
        tempopulation = deepcopy(population)

        # Elitism
        # Copy 2 best individuals to the new population
        startitr = 2
        tempopulation[0,:] = population[bestindex,:]
        tempopulation[1,:] = population[bestindex,:]

        #Tournament Selection
        matingpool = np.zeros(shape=[npop,nvar])
        for kk in range (0,npop):
            ip1 = int(np.ceil(npop*random_sample())) #random number 1
            ip2 = int(np.ceil(npop*random_sample())) #random number 2
            while ip1 >= 100 or ip2 >=100:
                ip1 = int(np.ceil(npop * random_sample()))
                ip2 = int(np.ceil(npop * random_sample()))
            if ip2 == ip1: #In case random number 1 = random number 2
                while ip2 == ip1 or ip2>=100:
                    ip2 = int(np.ceil(npop*random_sample()))

            Ft1  = population[ip1,:]
            Ft2  = population[ip2,:]
            Fit1 = fit[ip1]
            Fit2 = fit[ip2]

            #Switch case, in Python we use if and elif instead of switch-case
            if opt == "max":
                if Fit1>Fit2:
                    matingpool [kk,:] = Ft1
                else :
                    matingpool [kk,:] = Ft2
            elif opt == "min":
                if Fit1<Fit2:
                    matingpool [kk,:] = Ft1
                else :
                    matingpool [kk,:] = Ft2
            else:
                pass


        #Crossover with tournament seelection
        child = np.zeros(shape=[2,nvar])
        lst = np.arange(0, nvar)
        for jj in range (startitr,npop,2):
            idx1 = int(np.ceil(npop*random_sample()))
            idx2 = int(np.ceil(npop*random_sample()))
            while idx1 >= npop or idx2 >= npop or idx1==idx2:
                idx1 = int(np.ceil(npop * random_sample()))
                idx2 = int(np.ceil(npop * random_sample()))
            if (random_sample() < pcross):
                child = SBX.SBX(matingpool[idx1, :], matingpool[idx2, :], nvar, lb, ub)
                tempopulation[jj,0:nvar] = child [0,:]
                tempopulation[jj+1,0:nvar] = child [1,:]
            else:
                tempopulation[jj, 0:nvar] = matingpool [idx1,:]
                tempopulation[jj + 1, 0:nvar] = matingpool [idx2,:]

        #Mutation
        for kk in range (startitr,npop):
            tempopulation[kk,0:nvar] = mutation.gaussmut(tempopulation[kk, 0:nvar], nvar, pmut, ub, lb)

        history[generation-1,0]=generation
        history[generation-1,1]=bestFitness

        generation = generation+1


        #Next Population
        for i in range (0,npop):
            population[i,:] = tempopulation[i,:]


    #Show Best Fitness and Design Variables
    # print("Best Fitness = ",bestFitness)
    # for i in range (0,nvar):
    #     print("X",i+1," = ",bestx[i])

    return (bestx,bestFitness,history)

def GAfit3 (fitnessfcn, in2, in3, lb, ub, opt,num):
    nvar = len(ub)
    npop = 100      #number of initial population
    maxg = 100      #maximum generation
    pmut = 0.15     #mutation probability
    pcross = 0.95   #crossover probability
    history  = np.zeros(shape=[maxg,2]) #ask Kemas for explanation
    ub = np.asarray(ub)
    lb = np.asarray(lb)

    #Initialize population
    # samplenorm = haltonsampling.halton(nvar, npop)
    population = np.zeros(shape=[npop,nvar+1])
    # for i in range(0, npop):
    #     for j in range(0, nvar):
    #         population[i, j] = (samplenorm[i, j] * (ub[j] - lb[j])) + lb[j]
    for i in range (0,npop):
        for j in range (0,nvar):
            population[i,j] = lb[j] + (ub[j]-lb[j])*random_sample()
        temp = fitnessfcn(population[i,0:nvar],in2,in3,num)
        population[i,nvar] = deepcopy(temp)

    #Evolution loop
    generation = 1
    while generation <= maxg:
        # for generation 1:1
        tempopulation = deepcopy(population)

        #Tournament Selection
        matingpool = np.zeros(shape=[npop,nvar])
        for kk in range (0,npop):
            ip1 = int(np.ceil(npop*random_sample())) #random number 1
            ip2 = int(np.ceil(npop*random_sample())) #random number 2
            while ip1 >= 100 or ip2 >=100:
                ip1 = int(np.ceil(npop * random_sample()))
                ip2 = int(np.ceil(npop * random_sample()))
            if ip2 == ip1: #In case random number 1 = random number 2
                while ip2 == ip1 or ip2>=100:
                    ip2 = int(np.ceil(npop*random_sample()))

            lst  = np.arange(0,nvar)
            Ft1  = population[ip1,lst]
            Ft2  = population[ip2,lst]
            Fit1 = population[ip1,nvar]
            Fit2 = population[ip2,nvar]

            #Switch case, in Python we use if and elif instead of switch-case
            if opt == "max":
                if Fit1>Fit2:
                    matingpool [kk,:] = Ft1
                else :
                    matingpool [kk,:] = Ft2
            elif opt == "min":
                if Fit1<Fit2:
                    matingpool [kk,:] = Ft1
                else :
                    matingpool [kk,:] = Ft2
            else:
                pass


        #Crossover with tournament seelection
        child = np.zeros(shape=[2,nvar])
        lst = np.arange(0, nvar)
        for jj in range (0,npop,2):
            idx1 = int(np.ceil(npop*random_sample()))
            idx2 = int(np.ceil(npop*random_sample()))
            while idx1 >= npop or idx2 >= npop or idx1==idx2:
                idx1 = int(np.ceil(npop * random_sample()))
                idx2 = int(np.ceil(npop * random_sample()))
            if (random_sample() < pcross):
                child = SBX.SBX(matingpool[idx1, :], matingpool[idx2, :], nvar, lb, ub)
                tempopulation[jj,0:nvar] = child [0,:]
                tempopulation[jj+1,0:nvar] = child [1,:]
            else:
                tempopulation[jj, 0:nvar] = matingpool [idx1,:]
                tempopulation[jj + 1, 0:nvar] = matingpool [idx2,:]

            tempopulation[jj,nvar] = fitnessfcn(tempopulation[jj,lst],in2,in3,num)
            tempopulation[jj+1,nvar] = fitnessfcn(tempopulation[jj+1,lst],in2,in3,num)

        #Combined Population for Elitism
        compopulation = np.vstack((population,tempopulation))

        #Sort Population based on their fitness value
        if opt == 'max':
            i = np.argsort(compopulation[:,nvar]) [::-1]
            compopulation = compopulation[i,:]
        elif opt == 'min':
            i = np.argsort(compopulation[:, nvar])
            compopulation = compopulation[i,:]

        #Record Optimum Solution
        bestFitness = compopulation[0,nvar]
        bestx   = compopulation[0,0:nvar]

        #Mutation
        for kk in range (1,(2*npop)):
            compopulation[kk,0:nvar] = mutation.gaussmut(compopulation[kk, 0:nvar], nvar, pmut, ub, lb)
            compopulation[kk,nvar] = fitnessfcn (compopulation[kk,0:nvar],in2,in3,num)

        history[generation-1,0]=generation
        history[generation-1,1]=bestFitness

        generation = generation+1

        #Next Population
        for i in range (0,npop):
            population[i,:] = compopulation[i,:]


    #Show Best Fitness and Design Variables
    # print("Best Fitness = ",bestFitness)
    # for i in range (0,nvar):
    #     print("X",i+1," = ",bestx[i])

    return (bestx,bestFitness,history)
