import numpy as np
from numpy.linalg import solve as mldivide
import cma.evolution_strategy as cmaes
from miscellaneous.surrogate_support.kernel import calckernel
from miscellaneous.surrogate_support.hyp_trf import rescale

def likelihood (x,KrigInfo,num=None,**kwargs):
    """
    [NegLnLike,Psi,U]=likelihood(x)
    Calculates the negative of the concentrated ln-likelihood

    Inputs:
    x - vector of log(theta) parameters
    Global variables used:
    ModelInfo.X - n x k matrix of sample locations
    ModelInfo.y - n x 1 vector of observed data

    Outputs:
    NegLnLike - concentrated log-likelihood *-1 for minimising
    Psi - correlation matrix
    U - Choleski factorisation of correlation matrix

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
    mode = kwargs.get('retresult', "default")
    nvar = KrigInfo["nvar"]
    F = KrigInfo["F"]
    kernel = KrigInfo["kernel"]
    nkernel = KrigInfo["nkernel"]

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
        if KrigInfo["type"].lower() == "kpls":
            plscoeff = KrigInfo["plscoeff"]
    else:
        if KrigInfo["standardization"] == False:
            X = KrigInfo["X"]
            y = KrigInfo["y"][num]
        else:
            X = KrigInfo["X_norm"]
            F = KrigInfo["F"][num]
            if "y_norm" in KrigInfo:
                y = KrigInfo["y_norm"][num]
            else:
                y = KrigInfo["y"][num]
        if KrigInfo["type"].lower() == "kpls":
            plscoeff = KrigInfo["plscoeff"][num]

    if type(x) is float or type(x) is int or type(x) is np.float64 or type(x) is np.int64:
        x = np.array([x])
    if "n_princomp" in KrigInfo:
        nvar = KrigInfo["n_princomp"]

    if len(x) == nvar: # Nugget is not tunable, single kernel
        nugget = KrigInfo["nugget"]
        eps = 10. ** nugget
        wgkf = np.array([1])
    elif len(x) == nvar+1: # Nugget is tunable, single kernel
        # nugget = rescale(x[nvar],KrigInfo["lbhyp"][0],KrigInfo["ubhyp"][0],KrigInfo["lbhyp"][nvar],KrigInfo["ubhyp"][nvar])[0]
        nugget = x[nvar]
        eps = 10. ** nugget
        wgkf = np.array([1])
    elif len(x) == nvar+nkernel: # Nugget is not tunable, multiple kernels
        nugget = KrigInfo["nugget"]
        eps = 10. ** nugget
        # weight = rescale(x[nvar:nvar+nkernel],KrigInfo["lbhyp"][0],KrigInfo["ubhyp"][0],0,1)
        weight = x[nvar:nvar+nkernel]
        wgkf = weight/np.sum(weight)
    elif len(x) == nvar+nkernel+1:
        nugget = x[nvar]
        eps = 10. ** nugget
        weight = x[nvar+1:nvar+nkernel+1]
        wgkf = weight / np.sum(weight)


    theta = 10**(x[0:nvar])
    if num == None:
        KrigInfo["Theta"] = x[0:nvar]
        KrigInfo["nugget"] = nugget
        KrigInfo["wgkf"] = wgkf
    else:
        KrigInfo["Theta"][num] = x[0:nvar]
        KrigInfo["nugget"] = nugget
        KrigInfo["wgkf"][num] = wgkf
    p = 2 #from reference
    n = np.ma.size(X,axis=0)
    # one = np.ones((n,1),float)

    #Pre-allocate memory
    Psi = np.zeros(shape=[n,n])
    PsiComp = np.zeros(shape=[n,n,nkernel])


    #Build upper half of correlation matrix
    if KrigInfo["type"].lower() == "kriging":
        for ii in range(0,nkernel):
            PsiComp[:,:,ii] = wgkf[ii]*calckernel(X,X,theta,nvar,type=kernel[ii])
        Psi = np.sum(PsiComp,2)

    elif KrigInfo["type"].lower() == "kpls":
        nvar = KrigInfo["nvar"]
        for ii in range(0,nkernel):
            PsiComp[:,:,ii] = wgkf[ii]*calckernel(X,X,theta,nvar,type=kernel[ii],plscoeff=plscoeff)
        Psi = np.sum(PsiComp,2)


    #Add upper and lower halves and diagonal of ones plus
    #small number to reduce ill-conditioning
    # Psi = Psi + np.transpose(Psi) + np.eye(n) + (np.eye(n)*(eps)) #(np.eye(n)*eps)
    Psi = Psi + (np.eye(n) * (eps))
    testeig = np.linalg.eigvals(Psi)
    if np.any (np.linalg.eigvals(Psi)<0):
        print("Not positive definite")
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

        # Compute the coefficients of regression function
        temp11 = mldivide(np.transpose(U),y) #just a temporary variable for debugging
        temp1  = (mldivide(U,temp11)) #just a temporary variable for debugging
        temp21 = mldivide(np.transpose(U),F) #just a temporary variable for debugging
        temp2  = (mldivide(U,temp21)) #just a temporary variable for debugging
        tempmu     = mldivide(np.dot(np.transpose(F),temp2),np.dot(np.transpose(F),temp1))#np.dot(np.transpose(F),temp1)/np.dot(np.transpose(F),temp2)
        BE = tempmu

        # Use back-substitution of Cholesky instead of inverse
        temp31 = mldivide(np.transpose(U),(y - np.dot(F,BE) )) #just a temporary variable for debugging
        temp3  = mldivide(U,temp31) #just a temporary variable for debugging
        SigmaSqr = (np.dot(np.transpose(y - np.dot(F,BE)),(temp3)))/n

        tempNegLnLike    = -1*(-(n/2)*np.log(SigmaSqr) - 0.5*LnDetPsi)
        NegLnLike = tempNegLnLike[0,0]
    except:
        NegLnLike = 10000
        print("Matrix is ill-conditioned, penalty is used for NegLnLike value")
        print("Are you sure want to continue?")
        input("Press Enter to continue...")

    if num == None:
        KrigInfo["U"] = U
        KrigInfo["Psi"] = Psi
        KrigInfo["BE"] = BE
        KrigInfo["SigmaSqr"] = SigmaSqr[0,0]
    else:
        KrigInfo["U"][num] = np.array(U)
        KrigInfo["Psi"][num] = np.array(Psi)
        KrigInfo["BE"][num] = np.array(BE)
        KrigInfo["SigmaSqr"][num] = SigmaSqr[0,0]

    if mode.lower() == "default":
        return NegLnLike
    elif mode.lower() == "all":
        return KrigInfo
    else:
        raise TypeError("Only have two modes, default and all, default return NegLnLike, all return KrigInfo")
    print ("Psi = ",Psi)






