"""
This module contains any forms of kriging
**kwargs:
 - (num=) : for multi-objective optimization, stands for objective number XX
 - (ubvalue=) : upper bound value of the domain space, by default is 3
 - (lbvalue=) : lower bound value of the domain space, by default is -2
"""
import numpy as np
import globvar
from optim_tools.GAv1 import uncGA
from miscellaneous.surrogate_support import likelihood
from miscellaneous.surrogate_support.prediction import prediction
from sklearn.cross_decomposition.pls_ import PLSRegression as pls
from miscellaneous.sampling.samplingplan import standardize

def ordinarykrig (X,Y,ndim,**kwargs):
    globvar.type = "kriging"
    globvar.X = X
    globvar.y = Y

    num = kwargs.get('num',None) #Means objective function number XX
    ubvalue = kwargs.get('ub', 3)
    lbvalue = kwargs.get('ub', -2)
    standardization = kwargs.get('standardization', False)

    # upperbound and lowerbound for Theta
    ub = np.zeros(shape=[ndim]);
    ub[:] = ubvalue
    lb = np.zeros(shape=[ndim]);
    lb[:] = lbvalue
    opt = "min"

    # Standardize X and y
    if standardization == True:
        globvar.X_norm, globvar.y_norm, globvar.X_mean, globvar.y_mean, \
        globvar.X_std, globvar.y_std = standardize(X, Y)
        globvar.X = globvar.X_norm
        globvar.y = globvar.y_norm

    if num != None:
        print("Multi Objective, train hyperparam, begin.")
        # Use GA to find optimum value of Theta
        best_x,MinNegLnLikelihood,_ = uncGA(likelihood.likelihood,lb,ub,opt,disp=True,num=num)
        globvar.Theta[num] = best_x
        print("Multi Objective, train hyperparam, end.")
        NegLnLike,U,Psi = likelihood.likelihood(best_x,num)
        globvar.U[num] = U
        globvar.Psi[num] = Psi
    else:
        print("Single Objective, train hyperparam, begin.")
        # Use GA to find optimum value of Theta
        best_x, MinNegLnLikelihood, _ = uncGA(likelihood.likelihood, lb, ub, opt,disp=True)
        globvar.Theta = best_x
        print("Single Objective, train hyperparam, end.")
        NegLnLike, U, Psi = likelihood.likelihood(best_x)
        globvar.U = U
        globvar.Psi = Psi

    return (NegLnLike,U,Psi)

def kpls (X,Y,ndim,**kwargs):
    globvar.type = "kpls"
    globvar.X = X
    globvar.y = Y

    num = kwargs.get('num', None)  # Means objective function number XX
    ubvalue = kwargs.get('ub', 3)
    lbvalue = kwargs.get('ub', -2)
    n_princomp = kwargs.get('principalcomp', 1)
    standardization = kwargs.get('standardization', False)

    # Calculate PLS coeff
    _pls = pls(n_princomp)
    coeff_pls = _pls.fit(X.copy(), Y.copy()).x_rotations_
    globvar.plscoeff = coeff_pls

    # Standardize X and y
    if standardization == True:
        globvar.X_norm, globvar.y_norm, globvar.X_mean, globvar.y_mean, \
        globvar.X_std, globvar.y_std = standardize(X, Y)
        globvar.X = globvar.X_norm
        globvar.y = globvar.y_norm

    # upperbound and lowerbound for Theta
    ub = np.zeros(shape=[n_princomp]);
    ub[:] = ubvalue
    lb = np.zeros(shape=[n_princomp]);
    lb[:] = lbvalue
    opt = "min"

    if num != None:
        print("Multi Objective, train hyperparam, begin.")
        # Use GA to find optimum value of Theta
        best_x,MinNegLnLikelihood,_ = uncGA(likelihood.likelihood,lb,ub,opt,disp=True,num=num)
        globvar.Theta[num] = best_x
        print("Multi Objective, train hyperparam, end.")
        NegLnLike,U,Psi = likelihood.likelihood(best_x,num)
        globvar.U[num] = U
        globvar.Psi[num] = Psi
    else:
        print("Single Objective, train hyperparam, begin.")
        # Use GA to find optimum value of Theta
        best_x, MinNegLnLikelihood, _ = uncGA(likelihood.likelihood, lb, ub, opt,disp=True)
        globvar.Theta = best_x
        print("Single Objective, train hyperparam, end.")
        NegLnLike, U, Psi = likelihood.likelihood(best_x)
        globvar.U = U
        globvar.Psi = Psi


