# [NegLnLike,Psi,U]=likelihood(x)
# Calculates the negative of the concentrated ln-likelihood
# Inputs:
# x - vector of log(theta) parameters
# Global variables used:
# ModelInfo.X - n x k matrix of sample locations
# ModelInfo.y - n x 1 vector of observed data
# Outputs:
# NegLnLike - concentrated log-likelihood *-1 for minimising
# Psi - correlation matrix
# U - Choleski factorisation of correlation matrix
# Copyright 2007 A I J Forrester
# This program is free software: you can redistribute it and/or modify  it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any
# later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License and GNU
# Lesser General Public License along with this program. If not, see
# <http://www.gnu.org/licenses/>.
import globvar
import numpy as np
from numpy.linalg import solve as mldivide
import cma.evolution_strategy as cmaes

def likelihood (x,**num):
    num = num.get('num',None)
    X = globvar.X

    if globvar.multiobj == True:
        num = globvar.num

    if num == None:
        y = globvar.y
        plscoeff = globvar.plscoeff
    else:
        y = globvar.y[num]
        plscoeff = globvar.plscoeff[num]

    theta = 10**x
    p = 2 #from reference
    n = np.ma.size(X,axis=0)
    one = np.ones((n,1),float)
    eps = 10.*np.finfo(np.double).eps

    #Pre-allocate memory
    Psi = np.zeros(shape=[n,n])


    #Build upper half of correlation matrix
    if globvar.type == "kriging":
        for i in range (0,n):
            for j in range (i+1,n):
                Psi[i,j] = np.exp(-1*np.sum(theta*abs(X[i,:]-X[j,:])**p))

    elif globvar.type == "kpls":
        for i in range (0,n):
            for j in range (i+1,n):
                Psi[i,j] = np.exp(-1*np.sum(theta* np.dot( ((X[i,:] - X[j,:])**p) , (plscoeff**p) )))
        pass

    #Add upper and lower halves and diagonal of ones plus
    #small number to reduce ill-conditioning
    Psi = Psi + np.transpose(Psi) + np.eye(n) + (np.eye(n)*(eps)) #(np.eye(n)*eps)
    testeig = np.linalg.eigvals(Psi)
    #print("eigen = ",testeig)

    #try:
    #Cholesky Factorisation
    Utemp = np.linalg.cholesky(Psi) #Cholesky in Python Produce lower triangle
    U = np.transpose(Utemp) #Cholesky in Matlab Produce Upper Triangle

    #np.savetxt("tespsi.txt",Psi)
    #np.savetxt("testu.txt", U)
    #Use back-substitution of Cholesky instead of inverse
    try:
        # Sum lns of diagonal to find ln(abs(det(Psi)))
        LnDetPsi = 2 * np.sum(np.log(abs(np.diag(U))))

        temp11 = mldivide(np.transpose(U),y) #just a temporary variable for debugging
        temp1  = (mldivide(U,temp11)) #just a temporary variable for debugging
        temp21 = mldivide(np.transpose(U),one) #just a temporary variable for debugging
        temp2  = (mldivide(U,temp21)) #just a temporary variable for debugging
        tempmu     = np.dot(np.transpose(one),temp1)/np.dot(np.transpose(one),temp2)
        mu = tempmu[0,0]
        temp31 = mldivide(np.transpose(U),(y-one*mu)) #just a temporary variable for debugging
        temp3  = mldivide(U,temp31) #just a temporary variable for debugging
        SigmaSqr     = (np.dot(np.transpose(y - one * mu),(temp3)))/n
        tempNegLnLike    = -1*(-(n/2)*np.log(SigmaSqr) - 0.5*LnDetPsi)
        NegLnLike = tempNegLnLike[0,0]
    except:
        NegLnLike = 10000

    if num == None:
        globvar.U = U
        globvar.Psi = Psi
    else:
        globvar.U[num] = np.array(U)
        globvar.Psi[num] = np.array(Psi)
    # return (NegLnLike,U,Psi)
    return NegLnLike
    print ("Psi = ",Psi)






