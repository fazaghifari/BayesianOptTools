"""
This module contains any forms of kriging
**kwargs:
 - (num=) : for multi-objective optimization, stands for objective number XX
 - (ubvalue=) : upper bound value of the domain space, by default is 3
 - (lbvalue=) : lower bound value of the domain space, by default is -2
"""
import numpy as np
import globvar
from copy import deepcopy
from optim_tools.GAv1 import uncGA
from miscellaneous.surrogate_support import likelihood
from miscellaneous.surrogate_support.prediction import prediction
from sklearn.cross_decomposition.pls_ import PLSRegression as pls
from scipy.optimize import minimize_scalar
from miscellaneous.sampling.samplingplan import standardize
from cma import fmin2

def ordinarykrig (X,Y,ndim,**kwargs):
    globvar.type = "kriging"
    globvar.X = X

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

    if num != None:

        globvar.multiobj = True
        globvar.num = num
        globvar.y = [0]*(num+1)
        globvar.Theta = [0] * (num + 1)
        globvar.U = [0] * (num + 1)
        globvar.Psi = [0] * (num + 1)
        globvar.y_mean = [0] * (num + 1)
        globvar.y_std = [0] * (num + 1)
        globvar.y[num]= Y

        # Standardize X and y
        if standardization == True:
            globvar.X_norm, globvar.y_norm, globvar.X_mean, globvar.y_mean[num], \
            globvar.X_std, globvar.y_std[num] = standardize(X, Y)
            globvar.X = globvar.X_norm
            globvar.y[num] = globvar.y_norm
            globvar.standardization = True

        print("Multi Objective, train hyperparam, begin.")

        # Find optimum value of Theta
        if ndim <= 1:
            res = minimize_scalar(likelihood.likelihood, bounds=(lbvalue, ubvalue), method='golden')
            best_x = np.array([res.x])
        else:
            best_x, es = fmin2(likelihood.likelihood, ndim * [0], 3, options={'popsize': 150})

        globvar.Theta[num] = best_x
        print("Multi Objective, train hyperparam, end.")
        NegLnLike= likelihood.likelihood(best_x,num=num)
        U = globvar.U[num]
        Psi = globvar.Psi[num]

    else:
        globvar.y = Y

        # Standardize X and y
        if standardization == True:
            globvar.X_norm, globvar.y_norm, globvar.X_mean, globvar.y_mean, \
            globvar.X_std, globvar.y_std = standardize(X, Y)
            globvar.X = globvar.X_norm
            globvar.y = globvar.y_norm
            globvar.standardization = True

        print("Single Objective, train hyperparam, begin.")

        # Find optimum value of Theta
        if ndim <= 1:
            res = minimize_scalar(likelihood.likelihood, bounds=(lbvalue, ubvalue), method='golden')
            best_x = np.array([res.x])
        else:
            best_x,es = fmin2(likelihood.likelihood,ndim*[0],3,options={'popsize':150})

        globvar.Theta = best_x
        print("Single Objective, train hyperparam, end.")
        NegLnLike= likelihood.likelihood(best_x)
        U = globvar.U
        Psi = globvar.Psi

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

    # upperbound and lowerbound for Theta
    ub = np.zeros(shape=[n_princomp]);
    ub[:] = ubvalue
    lb = np.zeros(shape=[n_princomp]);
    lb[:] = lbvalue
    opt = "min"

    if num != None:

        globvar.multiobj = True
        globvar.num = num
        globvar.y = [0] * (num + 1)
        globvar.Theta = [0] * (num + 1)
        globvar.U = [0] * (num + 1)
        globvar.Psi = [0] * (num + 1)
        globvar.y_mean = [0] * (num + 1)
        globvar.y_std = [0] * (num + 1)
        globvar.y[num] = Y

        # Standardize X and y
        if standardization == True:
            globvar.X_norm, globvar.y_norm, globvar.X_mean, globvar.y_mean[num], \
            globvar.X_std, globvar.y_std[num] = standardize(X, Y)
            globvar.X = globvar.X_norm
            globvar.y[num] = globvar.y_norm
            globvar.standardization = True

        print("Multi Objective, train hyperparam, begin.")

        # Find optimum value of Theta
        if n_princomp <= 1:
            res = minimize_scalar(likelihood.likelihood, bounds=(lbvalue, ubvalue), method='golden')
            best_x = np.array([res.x])
            # best_x, MinNegLnLikelihood, _ = uncGA(likelihood.likelihood, lb, ub, opt, disp=True, num=num)
        else:
            best_x, es = fmin2(likelihood.likelihood, n_princomp * [0], 3, options={'popsize': 150})

        globvar.Theta[num] = best_x
        print("Multi Objective, train hyperparam, end.")
        NegLnLike = likelihood.likelihood(best_x,num=num)
        U = globvar.U[num]
        Psi = globvar.Psi[num]

    else:
        globvar.y = Y

        # Standardize X and y
        if standardization == True:
            globvar.X_norm, globvar.y_norm, globvar.X_mean, globvar.y_mean, \
            globvar.X_std, globvar.y_std = standardize(X, Y)
            globvar.X = globvar.X_norm
            globvar.y = globvar.y_norm
            globvar.standardization = True

        print("Single Objective, train hyperparam, begin.")

        # Find optimum value of Theta
        if n_princomp <= 1:
            res = minimize_scalar(likelihood.likelihood, bounds=(lbvalue, ubvalue), method='golden')
            best_x = np.array([res.x])
            # best_x, MinNegLnLikelihood, _ = uncGA(likelihood.likelihood, lb, ub, opt, disp=True)
        else:
            best_x, es = fmin2(likelihood.likelihood, n_princomp * [0], 3, options={'popsize': 150})

        globvar.Theta = best_x
        print("Single Objective, train hyperparam, end.")
        NegLnLike = likelihood.likelihood(best_x)
        U = globvar.U
        Psi = globvar.Psi

    return (NegLnLike, U, Psi)
