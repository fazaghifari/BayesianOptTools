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

import numpy as np
from numpy.linalg import solve as mldivide
from miscellaneous.surrogate_support.kernel import calckernel
from miscellaneous.sampling.samplingplan import standardize
from miscellaneous.surrogate_support.trendfunction import compute_regression_mat

def prediction (x,KrigInfo,**kwargs):
    # extract variables from data structure
    # slower, but makes code easier to follow
    num = kwargs.get('num',None) # Num means Objective Function number XX
    nvar = KrigInfo["nvar"]
    kernel = KrigInfo["kernel"]
    nkernel = KrigInfo["nkernel"]
    wgkf = KrigInfo["wgkf"]
    idx = KrigInfo["idx"]
    p = 2  # from reference

    if KrigInfo["multiobj"] == True:
        num = KrigInfo["num"]

    if num == None:
        if KrigInfo["standardization"] == False:
            X = KrigInfo["X"]
            y = KrigInfo["y"]
        else:
            X = KrigInfo["X_norm"]
            if "y_norm" in KrigInfo:
                y = KrigInfo["y_norm"]
            else:
                y = KrigInfo["y"]
        theta = 10 ** KrigInfo["Theta"]
        U = KrigInfo["U"]
        PHI = KrigInfo["F"]
        BE = KrigInfo["BE"]
        if KrigInfo["type"].lower() == "kpls":
            plscoeff = KrigInfo["plscoeff"]
    else:
        if KrigInfo["standardization"] == False:
            X = KrigInfo["X"][num]
            y = KrigInfo["y"][num]
        else:
            X = KrigInfo["X_norm"][num]
            if "y_norm" in KrigInfo:
                y = KrigInfo["y_norm"][num]
            else:
                y = KrigInfo["y"][num]
        theta = 10**KrigInfo["Theta"][num]
        U = KrigInfo["U"][num]
        PHI = KrigInfo["F"][num]
        BE = KrigInfo["BE"][num]
        if KrigInfo["type"].lower() == "kpls":
            plscoeff = KrigInfo["plscoeff"][num]

    if KrigInfo["standardization"] == True:
        if KrigInfo["normtype"] == "default":
            x = standardize(x, 0, type=KrigInfo["normtype"], range=np.vstack((KrigInfo["lb"],KrigInfo["ub"])))
        elif KrigInfo["normtype"] == "std":
            x = (x - KrigInfo["X_mean"]) / KrigInfo["X_std"]


    #Calculate number of sample points
    n = np.ma.size(X, axis=0)
    npred = np.size(x,axis=0)

    #Construct regression matrix for prediction
    bound = np.vstack((- np.ones(shape=[1, KrigInfo["nvar"]]), np.ones(shape=[1, KrigInfo["nvar"]])))
    PC = compute_regression_mat(idx,x,bound,np.ones(shape=[KrigInfo["nvar"]]))
    fpc = np.dot(PC,BE)

    #initialise psi to vector ones
    psi = np.ones((n,1),float)
    PsiComp = np.zeros(shape=[n, npred, nvar])

    #fill psi vector
    if KrigInfo["type"].lower() == "kriging":
        for ii in range(0,nkernel):
            PsiComp[:,:,ii] = wgkf[ii]*calckernel(X,x,theta,nvar,type=kernel[ii])
        psi = np.sum(PsiComp,2)
        # for i in range (0,n):
        #     psi[i]= np.exp(-1*np.sum(theta*abs(X[i,:]-x)**p))

    elif KrigInfo["type"].lower() == "kpls":
        for i in range(0, n):
            psi[i] = np.exp(-1 * np.sum(theta * np.dot(((X[i, :] - x) ** p), (plscoeff ** p))))

    #calculate prediction
    f = fpc + np.dot(np.transpose(psi), mldivide(U,mldivide(np.transpose(U),(y - PHI*BE))))
    if num == None:
        if KrigInfo["norm_y"] == True:
            f = (KrigInfo["y_mean"] + KrigInfo["y_std"]*f)
    else:
        if KrigInfo["norm_y"] == True:
            f = (KrigInfo["y_mean"][num] + KrigInfo["y_std"][num]*f)

    return f