import numpy as np
import globvar
from copy import deepcopy
from optim_tools.GAv1 import uncGA
from miscellaneous.surrogate_support import likelihood
from miscellaneous.surrogate_support.prediction import prediction
from miscellaneous.sampling.samplingplan import sampling
from sklearn.cross_decomposition.pls_ import PLSRegression as pls
from scipy.optimize import minimize_scalar
from miscellaneous.sampling.samplingplan import standardize
from miscellaneous.surrogate_support.trendfunction import polytruncation, compute_regression_mat
from scipy.optimize import minimize
import cma

def ordinarykrig (KrigInfo,**kwargs):
    """
    Create Kriging model based on the information from inputs and global variables.
    Inputs:
     - X : Experimental design
     - Y : Responses of the experimental design

    **kwargs:
     - (num=) : for multi-objective optimization, stands for objective number XX
     - (ubvalue=) : upper bound value of the domain space, by default is 3
     - (lbvalue=) : lower bound value of the domain space, by default is -2
    """
    KrigInfo["type"] = "kriging"
    eps = np.finfo(float).eps

    num = kwargs.get('num',None) #Means objective function number XX
    disp = kwargs.get('disp',None)
    ubvalue = kwargs.get('ub', 3)
    lbvalue = kwargs.get('lb', -3)
    standardization = kwargs.get('standardization', False)
    standtype = kwargs.get('normtype', "default")
    normy = kwargs.get('normalize_y', False)
    nbhyp = KrigInfo["nvar"]
    Y = KrigInfo["y"]
    X = KrigInfo["X"]
    sigmacmaes = (ubvalue-lbvalue)/5

    # upperbound and lowerbound for Theta
    ubtheta = np.zeros(shape=[nbhyp]);
    ubtheta[:] = ubvalue
    lbtheta = np.zeros(shape=[nbhyp]);
    lbtheta[:] = lbvalue
    opt = "min"

    # If conditional
    # Number of restart
    if 'nrestart' not in KrigInfo:
        KrigInfo['nrestart'] = 1

    #Polynomial Order
    if "TrendOrder " not in KrigInfo:
        KrigInfo["TrendOrder"] = 0
    elif KrigInfo["TrendOrder"] < 0:
        raise ValueError("The order of the polynomial trend should be a positive value.")

    #Nugget Setting
    if "nugget" not in KrigInfo:
        KrigInfo["nugget"] = -6
        KrigInfo["nuggetparam"] = "fixed"
        print("Nugget is not defined, set nugget to 1e-06")
    elif type(KrigInfo["nugget"]) is not list:
        KrigInfo["nuggetparam"] = "fixed"
        nnugget = 1
    elif len(KrigInfo["nugget"]) == 2:
        KrigInfo["nuggetparam"] = "tuned"
        nnugget = len(KrigInfo["nugget"]);
        if KrigInfo["nuggetparam"][0]> KrigInfo["nuggetparam"][1]:
            raise TypeError("The lower bound of the nugget should be lower than the upper bound.")

    #Kernel Setting
    if "kernel" not in KrigInfo:
        KrigInfo["kernel"] = ["gaussian"]
        nkernel = 1
        print("Kernel is not defined, set kernel to gaussian")
    elif type(KrigInfo["kernel"]) is not list :
        nkernel = 1
        KrigInfo["kernel"] = [KrigInfo["kernel"]]
    else:
        nkernel = len(KrigInfo["kernel"])

    lbhyp = lbtheta
    ubhyp = ubtheta
    KrigInfo["nkernel"] = nkernel

    #Overall Hyperparam
    if nnugget == 1 and nkernel == 1: #Fixed nugget, one kernel function
        pass
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
    elif nnugget > 1 and nkernel ==1: #Tunable nugget, one kernel function
        lbhyp = np.hstack((lbhyp,KrigInfo["nugget"][0]))
        ubhyp = np.hstack((ubhyp, KrigInfo["nugget"][1]))
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
        scaling[-1] = (KrigInfo["nugget"][1] - KrigInfo["nugget"][0])/(ubvalue-lbvalue)
    elif nnugget == 1 and nkernel > 1: #Fixed nugget, multiple kernel functions
        lbhyp = np.hstack((lbhyp, np.zeros(shape=[nkernel])+eps))
        ubhyp = np.hstack((ubhyp, np.ones(shape=[nkernel])))
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
        scaling[-nkernel:] = 1/(ubvalue-lbvalue)
    elif nnugget > 1 and nkernel > 1: #Tunable nugget, multiple kernel functions
        lbhyp = np.hstack((lbhyp, KrigInfo["nugget"][0],np.zeros(shape=[nkernel])+eps))
        ubhyp = np.hstack((ubhyp, KrigInfo["nugget"][1],np.ones(shape=[nkernel])))
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
        scaling[-nkernel-1] = (KrigInfo["nugget"][1] - KrigInfo["nugget"][0]) / (ubvalue - lbvalue)
        scaling[-nkernel:] = 1 / (ubvalue - lbvalue)
    KrigInfo["lbhyp"] = lbhyp
    KrigInfo["ubhyp"] = ubhyp


    #Optimize Hyperparam
    if num != None:
        # MULTI OBJECTIVE
        KrigInfo["multiobj"] = True
        KrigInfo["num"] = num

        # KrigInfo["y"][num]= np.transpose(np.array([Y[:,num]]))

        # Create regression matrix
        KrigInfo["idx"][num] = polytruncation(KrigInfo["TrendOrder"], KrigInfo["nvar"], 1)
        # Standardize X and y
        if standardization == True:
            if standtype.lower()=="default":
                KrigInfo["normtype"] = "default"
                bound = np.vstack((- np.ones(shape=[1, KrigInfo["nvar"]]), np.ones(shape=[1, KrigInfo["nvar"]])))
                if normy == True:
                    KrigInfo["X_norm"], KrigInfo["y_norm"][num] = standardize(X, Y, type=standtype.lower(), normy=True, range=np.vstack((KrigInfo["lb"],KrigInfo["ub"])))
                    KrigInfo["norm_y"] = True
                else:
                    KrigInfo["X_norm"] = standardize(X, Y,type=standtype.lower(), range=np.vstack((KrigInfo["lb"],KrigInfo["ub"])))
                    KrigInfo["norm_y"] = False
            else:
                KrigInfo["normtype"] = "std"
                if normy == True:
                    KrigInfo["X_norm"], KrigInfo["y_norm"][num], KrigInfo["X_mean"], KrigInfo["y_mean"][num], \
                    KrigInfo["X_std"], KrigInfo["y_std"][num] = standardize(X, Y, type=standtype.lower(), normy=True)
                    KrigInfo["norm_y"] = True
                else:
                    KrigInfo["X_norm"], KrigInfo["X_mean"], KrigInfo["X_std"] = standardize(X, Y, type=standtype.lower())
                    KrigInfo["norm_y"] = False
            KrigInfo["standardization"] = True
        else:
            KrigInfo["standardization"] = False
            KrigInfo["norm_y"] = False
        KrigInfo["F"][num] = compute_regression_mat(KrigInfo["idx"][num], KrigInfo["X_norm"], bound, np.ones(shape=[KrigInfo["nvar"]]))

        if disp == True:
            print("Multi Objective, train hyperparam, begin.")

        # Find optimum value of Theta
        if KrigInfo["nrestart"] <=1:
            xhyp = nbhyp*[0]
        else:
            _,xhyp = sampling('sobol',nbhyp,KrigInfo['nrestart'],result="real",upbound=ubhyp,lobound=lbhyp)

        bestxcand = np.zeros(shape=[KrigInfo['nrestart'], nbhyp])
        neglnlikecand = np.zeros(shape=[KrigInfo['nrestart']])
        if nbhyp <= 1:
            res = minimize_scalar(likelihood.likelihood, bounds=(lbvalue, ubvalue), method='golden')
            best_x = np.array([res.x])
        else:
            lbfgsbbound = np.transpose(np.vstack((lbhyp,ubhyp)))
            if disp == True:
                print("Hyperparam training is repeated for ",KrigInfo['nrestart']," time(s)")
            for ii in range(0,KrigInfo['nrestart']):
                if disp == True:
                    print("hyperparam training attempt number ",ii+1)
                # bestxcand[ii, :], es = cma.fmin2(likelihood.likelihood,xhyp[ii,:],sigmacmaes,{'BoundaryHandler': cma.BoundPenalty,'bounds': [lbhyp.tolist(), ubhyp.tolist()],'scaling_of_variables':scaling,'verb_disp': 0, 'verbose':-9},args=(KrigInfo,num))
                # neglnlikecand[ii] = es.result[1]
                res = minimize(likelihood.likelihood, xhyp[ii, :], method='L-BFGS-B', bounds=lbfgsbbound, args=(KrigInfo, num))
                bestxcand[ii, :] = res.x
                neglnlikecand[ii] = res.fun
                if disp == True:
                    print(" ")
            I = np.argmin(neglnlikecand)
            best_x = bestxcand[I, :]

        KrigInfo["Theta"][num] = best_x
        if disp == True:
            print("Multi Objective, train hyperparam, end.")
            print("Best hyperparameter is ", best_x)
            print("With NegLnLikelihood of ", neglnlikecand[I])
        KrigInfo= likelihood.likelihood(best_x,KrigInfo,retresult="all",num=num)
        U = KrigInfo["U"][num]
        Psi = KrigInfo["U"][num]

    else:
        # SINGLE OBJECTIVE
        KrigInfo["multiobj"] = False

        # Create regression matrix
        KrigInfo["idx"] = polytruncation(KrigInfo["TrendOrder"],KrigInfo["nvar"],1)
        # Standardize X and y
        if standardization == True:
            if standtype.lower() == "default":
                KrigInfo["normtype"] = "default"
                bound = np.vstack((- np.ones(shape=[1, KrigInfo["nvar"]]), np.ones(shape=[1, KrigInfo["nvar"]])))
                if normy == True:
                    KrigInfo["X_norm"], KrigInfo["y_norm"] = standardize(X, Y, type=standtype.lower(), normy=True, range=np.vstack((KrigInfo["lb"][num],KrigInfo["ub"][num])))
                    KrigInfo["norm_y"] = True
                else:
                    KrigInfo["X_norm"] = standardize(X, Y, type=standtype.lower(), range=np.vstack((KrigInfo["lb"][num],KrigInfo["ub"][num])))
                    KrigInfo["norm_y"] = False
            else:
                KrigInfo["normtype"] = "std"
                if normy == True:
                    KrigInfo["X_norm"], KrigInfo["y_norm"], KrigInfo["X_mean"], KrigInfo["y_mean"], \
                    KrigInfo["X_std"], KrigInfo["y_std"] = standardize(X, Y, type=standtype.lower(), normy=True)
                    KrigInfo["norm_y"] = True
                else:
                    KrigInfo["X_norm"], KrigInfo["X_mean"], KrigInfo["X_std"] = standardize(X, Y,type=standtype.lower())
                    KrigInfo["norm_y"] = False
            KrigInfo["standardization"] = True
        else:
            KrigInfo["standardization"] = False
            KrigInfo["norm_y"] = False
        KrigInfo["F"] = compute_regression_mat(KrigInfo["idx"],KrigInfo["X_norm"],bound,np.ones(shape=[KrigInfo["nvar"]]))

        if disp == True:
            print("Single Objective, train hyperparam, begin.")

        # Find optimum value of Theta
        if KrigInfo["nrestart"] <=1:
            xhyp = nbhyp*[0]
        else:
            _,xhyp = sampling('sobol',nbhyp,KrigInfo['nrestart'],result="real",upbound=ubhyp,lobound=lbhyp)

        bestxcand = np.zeros(shape=[KrigInfo['nrestart'],nbhyp])
        neglnlikecand = np.zeros(shape=[KrigInfo['nrestart']])
        if nbhyp <= 1:
            res = minimize_scalar(likelihood.likelihood, bounds=(lbvalue, ubvalue), method='golden')
            best_x = np.array([res.x])
        else:
            if disp == True:
                print("Hyperparam training is repeated for ",KrigInfo['nrestart']," time(s)")
            for ii in range(0,KrigInfo['nrestart']):
                if disp == True:
                    print("hyperparam training attempt number ",ii+1)
                bestxcand[ii, :], es = cma.fmin2(likelihood.likelihood,xhyp[ii,:],sigmacmaes,{'bounds': [lbhyp.tolist(), ubhyp.tolist()],'scaling_of_variables':scaling,'verb_disp': 0, 'verbose':-9},args=(KrigInfo,))
                neglnlikecand[ii] = es.result[1]
                if disp == True:
                    print(" ")
            I = np.argmin(neglnlikecand)
            best_x = bestxcand[I, :]

        if disp == True:
            print("Single Objective, train hyperparam, end.")
            print("Best hyperparameter is ", best_x)
            print("With NegLnLikelihood of ", neglnlikecand[I])
        KrigInfo= likelihood.likelihood(best_x,KrigInfo,retresult="all")
        U = KrigInfo["U"]
        Psi = KrigInfo["Psi"]

    return KrigInfo


def kpls (X,Y,**kwargs):
    globvar.type = "kpls"
    globvar.X = X
    globvar.y = Y

    num = kwargs.get('num', None)  # Means objective function number XX
    ubvalue = kwargs.get('ub', 3)
    lbvalue = kwargs.get('ub', -2)
    n_princomp = kwargs.get('principalcomp', 1)
    standardization = kwargs.get('standardization', False)

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
        globvar.plscoeff = [0] * (num + 1)
        globvar.y[num] = Y

        # Standardize X and y
        if standardization == True:
            globvar.X_norm, globvar.y_norm, globvar.X_mean, globvar.y_mean[num], \
            globvar.X_std, globvar.y_std[num] = standardize(X, Y)
            globvar.X = globvar.X_norm
            globvar.y[num] = globvar.y_norm
            globvar.standardization = True

        # Calculate PLS coeff
        _pls = pls(n_princomp)
        coeff_pls = _pls.fit(X.copy(), Y.copy()).x_rotations_
        globvar.plscoeff[num]= coeff_pls

        print("Multi Objective, train hyperparam, begin.")

        # Find optimum value of Theta
        if n_princomp <= 1:
            res = minimize_scalar(likelihood.likelihood, bounds=(lbvalue, ubvalue), method='golden')
            best_x = np.array([res.x])
            # best_x, MinNegLnLikelihood, _ = uncGA(likelihood.likelihood, lb, ub, opt, disp=True, num=num)
        else:
            best_x, es = cma.fmin2(likelihood.likelihood, n_princomp * [0], 3, options={'popsize': 10})

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
            X = globvar.X
            Y = globvar.y

        # Calculate PLS coeff
        _pls = pls(n_princomp)
        coeff_pls = _pls.fit(X.copy(), Y.copy()).x_rotations_
        globvar.plscoeff = coeff_pls

        print("Single Objective, train hyperparam, begin.")

        # Find optimum value of Theta
        if n_princomp <= 1:
            res = minimize_scalar(likelihood.likelihood, bounds=(lbvalue, ubvalue), method='golden')
            best_x = np.array([res.x])
            # best_x, MinNegLnLikelihood, _ = uncGA(likelihood.likelihood, lb, ub, opt, disp=True)
        else:
            best_x, es = cma.fmin2(likelihood.likelihood, n_princomp * [0], 3, options={'popsize': 10})

        globvar.Theta = best_x
        print("Single Objective, train hyperparam, end.")
        NegLnLike = likelihood.likelihood(best_x)
        U = globvar.U
        Psi = globvar.Psi

    return (NegLnLike, U, Psi)
