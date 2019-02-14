# f=pred(x)
#
# Calculates a Kriging prediction at x
#
# Inputs:
# 	x - 1 x k vetor of design variables
#
# Global variables used:
# 	ModelInfo.X - n x k matrix of sample locations
# 	ModelInfo.y - n x 1 vector of observed data
#   ModelInfo.Theta - 1 x k vector of log(theta)
#   ModelInfo.U - n x n Cholesky factorisation of Psi
#
# Outputs:
# 	f - scalar kriging prediction
#
# Copyright 2007 A I J Forrester
#
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

def prediction (x,**kwargs):
    # extract variables from data structure
    # slower, but makes code easier to follow
    num = kwargs.get('num',None) # Num means Objective Function number XX
    X = globvar.X
    p = 2  # from reference
    if num == None:
        y = globvar.y
        theta = 10 ** globvar.Theta
        U = globvar.U
    else:
        y = globvar.y[num]
        theta = 10**globvar.Theta[num]
        U = globvar.U[num]

    if globvar.standardization == True:
        x = (x-globvar.X_mean)/globvar.X_std

    #Calculate number of sample points
    n = np.ma.size(X, axis=0)

    #vector of ones
    one = np.ones((n, 1), float)

    #calculate mu
    temp11 = mldivide(np.transpose(U), y)  # just a temporary variable for debugging
    temp1 = (mldivide(U, temp11))  # just a temporary variable for debugging
    temp21 = mldivide(np.transpose(U), one)  # just a temporary variable for debugging
    temp2 = (mldivide(U, temp21))  # just a temporary variable for debugging
    mu = np.dot(np.transpose(one), temp1) / np.dot(np.transpose(one), temp2)

    #initialise psi to vector ones
    psi = np.ones((n,1),float)

    #fill psi vector
    if globvar.type == "kriging":
        for i in range (0,n):
            psi[i]= np.exp(-1*np.sum(theta*abs(X[i,:]-x)**p))

    elif globvar.type == "kpls":
        for i in range(0, n):
            psi[i] = np.exp(-1 * np.sum(theta * np.dot(((X[i, :] - x) ** p), (globvar.plscoeff ** p))))

    #calculate prediction
    f = mu + np.dot(np.transpose(psi), mldivide(U,mldivide(np.transpose(U),(y - one*mu))))
    if num == None:
        if globvar.standardization == True:
            f = (globvar.y_mean + globvar.y_std*f).ravel()
    else:
        if globvar.standardization == True:
            f = (globvar.y_mean[num] + globvar.y_std[num]*f).ravel()

    return f