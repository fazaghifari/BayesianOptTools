"""
Subroutines:
 - sampling : return normalized sample in range between 0 and 1
 - realval : return actual sample based on bound value

"""
import numpy as np
import globvar
from miscellaneous.sampling.haltonsampling import halton
from miscellaneous.sampling.rlh import rlh
from miscellaneous.sampling.sobol_seq import i4_sobol_generate

def sampling(option,nvar,nsamp,**kwargs):
    ret = kwargs.get('result', "normalized")
    ub = kwargs.get('upbound', np.array([None]))
    lb = kwargs.get('lobound', np.array([None]))

    if option.lower() == "halton":
        samplenorm = halton(nvar,nsamp)
    elif option.lower() == "sobol":
        samplenorm = i4_sobol_generate(nvar,nsamp)
    elif option.lower() == "rlh":
        samplenorm = rlh(nvar, nsamp)
    else:
        raise NameError("sampling plan unavailable!")

    if ret.lower() == "real" and lb.any() != None and ub.any() != None:
        checker = ub - lb;
        for numbers in checker:
            if numbers < 0:
                raise ValueError("Upper bound must bigger than lower bound!")
        sample = realval(lb, ub, samplenorm)
    elif ret.lower() == "real" and (lb.any == None or ub.any == None):
        raise ValueError("lb and ub must have value")

    if ret.lower() == "real":
        return samplenorm,sample
    else:
        print("real value returned as zero matrix")
        return samplenorm,np.zeros(np.shape(samplenorm))

def realval(lb,ub,samp):
    if len(ub) != len(lb):
        raise NameError("Lower and upper bound have to be in the same dimension")
    if len(ub) != np.size(samp,axis=1):
        raise NameError("sample and bound are not in the same dimension")
    ndim = len(ub)
    nsamp = np.size(samp,axis=0)
    realsamp = np.zeros(shape=[nsamp,ndim])
    for i in range(0, ndim):
        for j in range(0, nsamp):
            realsamp[j, i] = (samp[j, i] * (ub[i] - lb[i])) + lb[i]
    return realsamp

def standardize(X,y,**kwargs):
    type = kwargs.get('type', "default")
    norm_y = kwargs.get('normy',False)
    ranges = kwargs.get('range', np.array([None]))

    if type.lower()=='default':
        X_norm = np.empty(np.shape(X))
        y_norm = np.empty(np.shape(y))
        if ranges.any() == None:
            raise ValueError("Default normalization requires range value!")
        if norm_y == True:
            # Normalize to [0,1]
            for i in range(0, np.size(X, 1)):
                X_norm[:, i] = (X[:, i] - ranges[0, i]) / (ranges[1, i] - ranges[0, i])
            for i in range(0, np.size(y, 1)):
                y_norm[:, i] = (y[:, i] - ranges[0, i]) / (ranges[1, i] - ranges[0, i])
            # Normalize to [-1,1]
            X_norm = (X_norm - 0.5) * 2
            y_norm = (y_norm - 0.5) * 2
            return X_norm,y_norm
        else:
            #Normalize to [0,1]
            for i in range(0,np.size(X,1)):
                X_norm[:,i] = (X[:,i]-ranges[0,i])/(ranges[1,i]-ranges[0,i])
            #Normalize to [-1,1]
            X_norm = (X_norm-0.5)*2
            return X_norm

    elif type.lower() == 'std':
        if norm_y == True:
            X_mean = np.mean(X, axis=0)
            X_std = X.std(axis=0, ddof=1)
            y_mean = np.mean(y, axis=0)
            y_std = y.std(axis=0, ddof=1)
            X_std[X_std == 0.] = 1.
            y_std[y_std == 0.] = 1.

            X_norm = (X - X_mean) / X_std
            y_norm = (y - y_mean) / y_std
            return X_norm, y_norm, X_mean, y_mean, X_std, y_std
        else:
            X_mean = np.mean(X, axis=0)
            X_std = X.std(axis=0, ddof=1)
            X_std[X_std == 0.] = 1.

            X_norm = (X - X_mean) / X_std
            return X_norm, X_mean, X_std