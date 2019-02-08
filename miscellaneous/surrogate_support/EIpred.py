"""
This code was written by Ghifari Adam F, 2018

The original matlab code belong to:
    Copyright 2007 A I J Forrester

    This program is free software: you can redistribute it and/or modify  it
    under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or any
    later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
    General Public License for more details.

    You should have received a copy of the GNU General Public License and GNU
    Lesser General Public License along with this program. If not, see
    <http://www.gnu.org/licenses/>.
"""

import numpy as np
import globvar
from numpy.linalg import solve as mldivide
from math import erf; from copy import deepcopy

def eipred(x,**kwargs):
    # metric=predictor(x)
    #
    # Calculates the Kriging prediction, RMSE, -log(E[I(x)]) or -log(P[I(x)])
    #
    # Inputs:	# x - 1 x k vector of design variables
    #
    # Global variables used:	# ModelInfo.X - n x k matrix of sample locations	# ModelInfo.y - n x 1 vector of observed data
    #   ModelInfo.Theta - 1 x k vector of log(theta)
    #   ModelInfo.U - n x n Cholesky factorisation of Psi
    #   ModelInfo.Option - string: 'Pred', 'RMSE', 'NegLogExpImp' or 'NegProbImp'
    #
    # Outputs:	# metric - prediction, RMSE, -log(E[I(x)]) or -log(P[I(x)]), determined	# by ModelInfo.option
    num = kwargs.get('num', None)  # Means objective function number XX
    X = globvar.X
    p = 2  # from reference
    if num != None:
        y = globvar.y[num]
        theta = 10**globvar.Theta[num]
        U = globvar.U[num]
    else:
        y = globvar.y
        theta = 10 ** globvar.Theta
        U = globvar.U

    # Calculate number of sample points
    n = np.ma.size(X, axis=0)

    # vector of ones
    one = np.ones((n, 1), float)

    # calculate mu
    temp11 = mldivide(np.transpose(U), y)  # just a temporary variable for debugging
    temp1 = (mldivide(U, temp11))  # just a temporary variable for debugging
    temp21 = mldivide(np.transpose(U), one)  # just a temporary variable for debugging
    temp2 = (mldivide(U, temp21))  # just a temporary variable for debugging
    mu = np.dot(np.transpose(one), temp1) / np.dot(np.transpose(one), temp2)

    # calculate sigma sqr
    temp31 = mldivide(np.transpose(U), (y - one * mu))  # just a temporary variable for debugging
    temp3 = mldivide(U, temp31)  # just a temporary variable for debugging
    SigmaSqr = (np.dot(np.transpose(y - one * mu), (temp3))) / n

    # initialise psi to vector ones
    psi = np.ones((n, 1), float)

    # fill psi vector
    for i in range(0, n):
        psi[i] = np.exp(-1 * np.sum(theta * abs(X[i, :] - x) ** p))

    # calculate prediction
    f = mu + np.dot(np.transpose(psi), mldivide(U, mldivide(np.transpose(U), (y - one * mu))))

    #switch case
    if globvar.Option != 'Pred':
        SSqr = SigmaSqr*(1-(np.dot(np.transpose(psi),mldivide(U,mldivide(np.transpose(U),psi)))))
        s = (abs(SSqr))**0.5
        if globvar.Option != 'RMSE':
            if globvar.ConstraintLimit == []:
                yBest = np.min(y)
            else:
                yBest = globvar.ConstraintLimit
            if globvar.Option == 'NegProbImp':
                ProbImp = 0.5 + 0.5* erf((1/np.sqrt(2))*((yBest-f)/s))
            else:
                EITermOne = (yBest-f)*(0.5+0.5*erf((1/np.sqrt(2))*((yBest-f)/s)))
                EITermTwo = s*(1/np.sqrt(2*np.pi))*np.exp(-(1/2)*(((yBest-f)**2)/SSqr))
                ExpImp = np.log10(EITermOne+EITermTwo+1e-10)

    if globvar.Option == 'Pred':
        metric = f
    elif globvar.Option == 'RMSE':
        metric = s
    elif globvar.Option == 'NegLogExpImp':
        metric = -ExpImp
    elif globvar.Option == 'NegProbImp':
        metric = -ProbImp
    else:
        raise NameError('ERROR!')
    return (metric,0,0)

def multiei (x,num):
    # Calculates the expection of f_1(x) and f_2(x) improving
    # on the Pareto front defined by ObjectiveInfo{1:2}.y
    #
    # Inputs:
    #	x - 1 x k vetor of design variables
    #
    # Global variables used:
    #	ObjectiveInfo{1:2}.X - n x k matrix of sample locations for each
    #	objective
    #	ObjectiveInfo{1:2}.y - n x 1 vector of observed data for each objective
    #   ObjectiveInfo{1:2}.Theta - 1 x k vector of log(theta) for each
    #   objective
    #   ObjectiveInfo{1:2}.U - n x n Cholesky factorisation of Psi  for each
    #   objective
    #   ModelInfo.Option - string: 'NegLogExpImp' or 'NegProbImp'
    #
    # Outputs:
    #	metric - either -log(E[I(x*)]) -P[I(x*)], determined by
    #	ModelInfo.option
    #   Py1,Py2 - non-dominated objective function values on the Pareto front
    #   PX - locations of non-dominated solutions
    #
    # Calls:
    #   predictor.m
    X = globvar.X
    y1 = globvar.y[0]
    y2 = globvar.y[1]
    k = np.ma.size(globvar.y[1],axis = 1)
    globvar.Option = "NegLogExpImp"

    ## Find points which satisfy constraint (if present)
    ## This section only works for constrained case
    y1temp  = y1
    y2temp  = y2
    Xtemp   = X
    try:
        (ConstraintInfo,var)
    except:
        pass
    else:
        for i in range (0,len(y1)+1):
            for j in range (ConstraintInfo,2):
                if Constraintinfo.y[j] > ConstraintInfo.Limit:
                    y1temp[i] = np.nan
                    y2temp[i] = np.nan

        Xtemp  = Xtemp[~np.isnan(y2temp),:]
        y1temp = y1temp[~np.isnan(y1temp)]
        y2temp = y2temp[~np.isnan(y2temp)]
    ##End Section

    ##Find Pareto Set
    b = np.argsort(y1temp[:,0])
    PX = [] ; PX.append(Xtemp[b[0],0:k])
    Py1 = []; Py1.append(y1temp[b[0],0])
    Py2 = []; Py2.append(y2temp[b[0],0])
    Pnum = 0;
    for i in range (1,np.size(y1temp,axis=0)):
        if y2temp[b[i],0] <= Py2[-1]:
            Pnum = Pnum+1
            PX.append(Xtemp[b[i],0:k])
            Py1.append(y1temp[b[i],0])
            Py2.append(y2temp[b[i],0])

    ##Prediction of each objective
    globvar.Option = "Pred"
    pred1temp,_,_ = eipred(x,0)
    pred2temp,_,_ = eipred(x,1)
    pred1 = deepcopy(pred1temp[0, 0]); pred2 = deepcopy(pred2temp[0, 0])
    ##RMSE of each objective
    globvar.Option = "RMSE"
    s1temp,_,_ = eipred(x,0)
    s2temp,_,_ = eipred(x,1)
    s1 = deepcopy(s1temp[0,0]); s2 = deepcopy(s2temp[0,0])
    ##Probability of Improvement Calc
    globvar.Option = "NegProbImp"
    Piterm1 = (0.5+0.5*erf((1/(2**0.5))*((Py1[0]-pred1)/s1)))
    Piterm3 = (1-(0.5+0.5*erf((1/(2**0.5))*((Py1[-1]-pred1)/s1))))\
    * (0.5+0.5*erf((1/(2**0.5))*((Py2[-1]-pred2)/s2)))
    z = len(Py1)
    Piterm2calc = (np.zeros(shape=[z-1]))
    if Pnum > 0:
        for ii in range(0,len(Py1)-1):
            Piterm2calc[ii] = ((0.5+0.5*erf((1/(2**0.5))*((Py1[ii+1]-pred1)/s1)))\
                               -(0.5+0.5*erf((1/(2**0.5))*((Py1[ii]-pred1)/s1))))\
                                *(0.5+0.5*erf((1/(2**0.5))*((Py2[ii+1]-pred2)/s2)))
        Piterm2 = np.sum(Piterm2calc)
        Pi = (Piterm1+Piterm2+Piterm3)
    else:
        Pi = (Piterm1+Piterm3)

    Ybar1Term1 = pred1*(0.5+0.5*erf((1/(2**0.5))*((Py1[0]-pred1)/s1)))\
                - s1*(1/((2*np.pi)**0.5))*np.exp(-0.5*((Py1[0]-pred1)**2 / s1**2))

    Ybar1Term3 = (pred1*(0.5+0.5*erf((1/(2**0.5))*((pred1-Py1[-1])/s1)))\
                +s1*(1/((2*np.pi)**0.5))*np.exp(-0.5*((pred1-Py1[-1])**2 / s1**2)))\
                *(0.5+0.5*erf((1/(2**0.5))*((Py2[-1]-pred2) / s2)))
    Ybar1Term2calc = (np.zeros(shape=[len(Py1)-1]))
    if Pnum>0:
        for I in range (0,len(Py1)-1):
            Ybar1Term2calc[I] = ((pred1*(0.5+0.5*erf((1/(2**0.5))*((Py1[I+1]-pred1)/s1)))\
            -s1*(1/((2*np.pi)**0.5))*np.exp(-0.5*((Py1[I+1]-pred1)**2/s1**2)))\
            -(pred1*(0.5+0.5*erf((1/(2**0.5))*((Py1[I]-pred1)/s1)))\
            -s1*(1/((2*np.pi)**0.5))*np.exp(-0.5*((Py1[I]-pred1)**2/s1**2)))) \
            * (0.5+0.5*erf((1/(2**0.5))*((Py2[I + 1] - pred2) / s2)))
        Ybar1Term2 = np.sum(Ybar1Term2calc)
        Ybar1 = (Ybar1Term1+Ybar1Term2+Ybar1Term3)/Pi
    else:
        Ybar1 = (Ybar1Term1+Ybar1Term3)/Pi

    Ybar2Term1 = pred2*(0.5+0.5*erf((1/(2**0.5))*((Py2[-1]-pred2)/s2))) \
                 -s2*(1/((2*np.pi)**0.5))*np.exp(-0.5*((Py2[-1]-pred2)**2 / s2**2))

    Ybar2Term3 = (pred2*(0.5+0.5*erf((1/(2**0.5))*((pred2-Py2[0])/s2))) \
                  +s2*(1/((2*np.pi)**0.5))*np.exp(-0.5*((pred2-Py2[0])**2 / s2**2))) \
                 *(0.5+0.5*erf((1/(2**0.5))*((Py1[0]-pred1) / s1)))
    Ybar2Term2calc = (np.zeros(shape=[len(Py1) - 1]))
    if Pnum>0:
        for I in range(len(Py2),1,-1):
            Ybar2Term2calc[I-2] =  ((pred2*(0.5+0.5*erf((1/(2**0.5))*((Py2[I-2]-pred2)/s2))) \
                                -s2*(1/((2*np.pi)**0.5))*np.exp(-0.5*((Py2[I-2]-pred2)**2 / s2**2))) \
                                -(pred2*(0.5+0.5*erf((1/(2**0.5))*((Py2[I-1]-pred2)/s2))) \
                                -s2*(1/((2*np.pi)**0.5))*np.exp(-0.5*((Py2[I-1]-pred2)**2 / s2**2))))\
                                *(0.5+0.5*erf((1/(2**0.5))*((Py1[I-2]-pred1)/s1)))
        Ybar2Term2 = np.sum(Ybar2Term2calc)
        Ybar2 = (Ybar2Term1+Ybar2Term2+Ybar2Term3)/Pi
    else:
        Ybar2 = (Ybar2Term1+Ybar2Term3)/Pi

    #Find Closest Point on Front
    dist = np.zeros(shape=[Pnum])
    for i in range(0,Pnum):
        dist[i] = np.sqrt((Ybar1-Py1[i])**2 + (Ybar2-Py2[i])**2)

    a = np.sort(dist)

    #Expected Improvement Calculation
    if Pi == 0:
        EI = 0
    else:
        EI = Pi*a[0]

    metric = -np.log10(EI+np.finfo(float).tiny)
    return (metric,Py1,Py2)