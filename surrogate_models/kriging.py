import numpy as np
from copy import deepcopy
from miscellaneous.surrogate_support import likelihood
from miscellaneous.surrogate_support.krigloocv import loocv
from miscellaneous.sampling.samplingplan import sampling
from sklearn.cross_decomposition.pls_ import PLSRegression as pls
from scipy.optimize import minimize_scalar
from miscellaneous.sampling.samplingplan import standardize
from miscellaneous.surrogate_support.trendfunction import polytruncation, compute_regression_mat
from scipy.optimize import minimize, fmin_cobyla
import cma

def kriging (KrigInfo,loocvcalc=False,**kwargs):
    """
    Create Kriging model based on the information from inputs and global variables.
    Inputs:
     - KrigInfo - Containing necessary information to create a Kriging model

    Outputs:
     - MyKrig - Trained kriging model

    Details of KrigInfo:
    REQUIRED PARAMETERS. These parameters need to be specified manually by
    the user. Otherwise, the process cannot continue.
        - KrigInfo['lb'] - Variables' lower bounds.
        - KrigInfo['ub'] - Variables' upper bounds.
        - KrigInfo['nvar'] - Number of variables.
        - KrigInfo['nsamp'] - Number of samples.
        - KrigInfo['X'] - Experimental design.
        - KrigInfo['y'] - Responses of the experimental design.

    EXTRA PARAMETERS. These parameters can be set by the user. If not
    specified, default values will be used (or computed for the experimetntal design and responses)
        - KrigInfo['problem'] - Function name to evaluate responses (No need to specify this if KrigInfo.X and KrigInfo.Y are specified).
        - KrigInfo['nugget'] - Nugget (noise factor). Default: 1e-6
        - KrigInfo['TrendOrder'] - Polynomial trend function order (note that this code uses polynomial chaos expansion). Default: 0 (ordinary Kriging).
        - KrigInfo['kernel'] - Kernel function. Available kernels are 'gaussian', 'exponential','matern32', 'matern52', and 'cubic'. Default: 'Gaussian'.
        - KrigInfo['nrestart'] - Number of restarts for hyperparameters optimization. Default: 1.
        - KrigInfo['LOOCVtype'] - Type of cross validation error. Default: 'rmse'.
        - KrigInfo['optimizer'] - Optimizer for hyperparameters optimization. Default: 'cmaes'.

    **kwargs:
     - (num=) : for multi-objective optimization, stands for objective number XX
     - (ub=) : upper bound value of the domain space, by default is 3
     - (lb=) : lower bound value of the domain space, by default is -3
     - (standardization=) : True or False, normalize sample
     - (standtype=) : default or std, default range (-1 1), std (standard score)
     - (normy=) : True or False, normalize y or not
    """
    KrigInfo["type"] = "kriging"
    eps = np.finfo(float).eps

    num = kwargs.get('num',None) #Means objective function number XX
    disp = kwargs.get('disp',None)
    ubvalue = kwargs.get('ub', 4)
    lbvalue = kwargs.get('lb', -4)
    standardization = kwargs.get('standardization', False)
    standtype = kwargs.get('normtype', "default")
    normy = kwargs.get('normalize_y', True)
    nbhyp = KrigInfo["nvar"]+1
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

    if "optimizer" not in KrigInfo:
        KrigInfo["optimizer"] = "lbfgsb"
        print("The acquisition function is not specified, set to lbfgsb")
    else:
        availoptmzr = ["lbfgsb","cmaes","cobyla"]
        if KrigInfo["optimizer"].lower() not in availoptmzr:
            raise TypeError(KrigInfo["optimizer"]," is not a valid acquisition function.")
        if disp == True:
            print("The acquisition function is specified to ", KrigInfo["optimizer"], " by user")

    #Polynomial Order
    if "TrendOrder" not in KrigInfo:
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
        Y = np.hstack((Y[0],Y[1]))

        # KrigInfo["y"][num]= np.transpose(np.array([Y[:,num]]))

        # Create regression matrix
        KrigInfo["idx"][num] = polytruncation(KrigInfo["TrendOrder"], KrigInfo["nvar"], 1)
        # Standardize X and y
        if standardization == True:
            if standtype.lower()=="default":
                KrigInfo["normtype"] = "default"
                bound = np.vstack((- np.ones(shape=[1, KrigInfo["nvar"]]), np.ones(shape=[1, KrigInfo["nvar"]])))
                if normy == True:
                    KrigInfo["X_norm"], y_normtemp = standardize(X, Y, type=standtype.lower(), normy=True, range=np.vstack((np.hstack((KrigInfo["lb"],np.min(Y,0))),np.hstack((KrigInfo["ub"],np.max(Y,0))))))
                    KrigInfo["norm_y"] = True
                    KrigInfo["y_norm"][num] = y_normtemp[:,num]
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

        if loocvcalc == True:
            KrigInfo["LOOCVerror"][num],KrigInfo["LOOCVpred"][num] = loocv(KrigInfo,errtype="mape",num=num)

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
                    KrigInfo["X_norm"], KrigInfo["y_norm"]= standardize(X, Y, type=standtype.lower(), normy=True, range=np.vstack((np.hstack((KrigInfo["lb"],np.min(Y))),np.hstack((KrigInfo["ub"],np.max(Y))))))
                    KrigInfo["norm_y"] = True
                else:
                    KrigInfo["X_norm"] = standardize(X, Y, type=standtype.lower(), range=np.vstack((KrigInfo["lb"],KrigInfo["ub"])))
                    KrigInfo["norm_y"] = False
            else:
                KrigInfo["normtype"] = "std"
                bound = np.vstack((- np.ones(shape=[1, KrigInfo["nvar"]]), np.ones(shape=[1, KrigInfo["nvar"]])))
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
            if KrigInfo["optimizer"] == "lbfgsb":
                lbfgsbbound = np.transpose(np.vstack((lbhyp, ubhyp)))
            elif KrigInfo["optimizer"] == "cobyla":
                constraints = []
                for i in range(len(ubhyp)):
                    constraints.append(lambda x, Kriginfo, itemp=i: x[itemp] - lbhyp[itemp])
                    constraints.append(lambda x, Kriginfo, itemp=i: ubhyp[itemp] - x[itemp])

            if disp == True:
                print("Hyperparam training is repeated for ",KrigInfo['nrestart']," time(s)")
            for ii in range(0,KrigInfo['nrestart']):
                if disp == True:
                    print("hyperparam training attempt number ",ii+1)
                if KrigInfo["optimizer"] == "cmaes":
                    bestxcand[ii, :], es = cma.fmin2(likelihood.likelihood,xhyp[ii,:],sigmacmaes,{'bounds': [lbhyp.tolist(), ubhyp.tolist()],'scaling_of_variables':scaling,'verb_disp': 0, 'verbose':-9},args=(KrigInfo,))
                    neglnlikecand[ii] = es.result[1]
                elif KrigInfo["optimizer"] == "lbfgsb":
                    res = minimize(likelihood.likelihood, xhyp[ii, :], method='L-BFGS-B', bounds=lbfgsbbound, args=(KrigInfo, num))
                    bestxcand[ii, :] = res.x
                    neglnlikecand[ii] = res.fun
                elif KrigInfo["optimizer"] == "cobyla":
                    res = fmin_cobyla(likelihood.likelihood, xhyp[ii, :], constraints, rhobeg = 0.5, rhoend = 1e-4, args=(KrigInfo,))
                    bestxcand[ii, :] = res
                    neglnlikecand[ii] = likelihood.likelihood(res,KrigInfo)
                if disp == True:
                    print(" ")
            I = np.argmin(neglnlikecand)
            best_x = bestxcand[I, :]

        if disp == True:
            print("Single Objective, train hyperparam, end.")
            print("Best hyperparameter is ", best_x)
            print("With NegLnLikelihood of ", neglnlikecand[I])
        # best_x = np.array([-0.74496822, -0.82209113, -0.21335276136851444]) # inject value best_x for debugging
        KrigInfo= likelihood.likelihood(best_x,KrigInfo,retresult="all")
        U = KrigInfo["U"]
        Psi = KrigInfo["Psi"]

        if loocvcalc == True:
            KrigInfo["LOOCVerror"],KrigInfo["LOOCVpred"] = loocv(KrigInfo,errtype="mape")

    return KrigInfo



def kpls (KrigInfo,**kwargs):
    """
    Create Kriging with Partial Least Square model based on the information from inputs and global variables.
    Inputs:
     - KrigInfo - Containing necessary information to create a Kriging model

    Outputs:
     - MyKrig - Trained kriging model

    Details of KrigInfo:
    REQUIRED PARAMETERS. These parameters need to be specified manually by
    the user. Otherwise, the process cannot continue.
        - KrigInfo['lb'] - Variables' lower bounds.
        - KrigInfo['ub'] - Variables' upper bounds.
        - KrigInfo['nvar'] - Number of variables.
        - KrigInfo['nsamp'] - Number of samples.
        - KrigInfo['X'] - Experimental design.
        - KrigInfo['y'] - Responses of the experimental design.

    EXTRA PARAMETERS. These parameters can be set by the user. If not
    specified, default values will be used (or computed for the experimetntal design and responses)
        - KrigInfo['problem'] - Function name to evaluate responses (No need to specify this if KrigInfo.X and KrigInfo.Y are specified).
        - KrigInfo['nugget'] - Nugget (noise factor). Default: 1e-6
        - KrigInfo['TrendOrder'] - Polynomial trend function order (note that this code uses polynomial chaos expansion). Default: 0 (ordinary Kriging).
        - KrigInfo['kernel'] - Kernel function. Available kernels are 'gaussian', 'exponential','matern32', 'matern52', and 'cubic'. Default: 'Gaussian'.
        - KrigInfo['nrestart'] - Number of restarts for hyperparameters optimization. Default: 1.
        - KrigInfo['LOOCVtype'] - Type of cross validation error. Default: 'rmse'.
        - KrigInfo['optimizer'] - Optimizer for hyperparameters optimization. Default: 'cmaes'.
        - KrigInfo['n_princomp'] - Number of principal component. Default: 1.

    **kwargs:
     - (num=) : for multi-objective optimization, stands for objective number XX
     - (ub=) : upper bound value of the domain space, by default is 3
     - (lb=) : lower bound value of the domain space, by default is -3
     - (standardization=) : True or False, normalize sample
     - (standtype=) : default or std, default range (-1 1), std (standard score)
     - (normy=) : True or False, normalize y or not
    """
    KrigInfo["type"] = "kpls"
    eps = np.finfo(float).eps

    num = kwargs.get('num', None)  # Means objective function number XX
    disp = kwargs.get('disp', None)
    ubvalue = kwargs.get('ub', 6)
    lbvalue = kwargs.get('lb', -6)
    standardization = kwargs.get('standardization', False)
    standtype = kwargs.get('normtype', "default")
    normy = kwargs.get('normalize_y', False)
    nbhyp = KrigInfo["nvar"]+1
    y = KrigInfo["y"]
    X = KrigInfo["X"]
    sigmacmaes = (ubvalue - lbvalue) / 5

    # Set default value for number of principal component
    if "n_princomp" not in KrigInfo:
        KrigInfo["n_princomp"] = 1
        n_princomp = KrigInfo["n_princomp"]+1
    else:
        n_princomp = KrigInfo["n_princomp"]+1

    # upperbound and lowerbound for Theta
    ubtheta = np.zeros(shape=[n_princomp]);
    ubtheta[:] = ubvalue
    lbtheta = np.zeros(shape=[n_princomp]);
    lbtheta[:] = lbvalue
    opt = "min"

    # If conditional
    # Number of restart
    if 'nrestart' not in KrigInfo:
        KrigInfo['nrestart'] = 1

    if "optimizer" not in KrigInfo:
        KrigInfo["optimizer"] = "lbfgsb"
        print("The acquisition function is not specified, set to lbfgsb")
    else:
        availoptmzr = ["lbfgsb", "cmaes", "cobyla"]
        if KrigInfo["optimizer"].lower() not in availoptmzr:
            raise TypeError(KrigInfo["optimizer"], " is not a valid acquisition function.")
        print("The acquisition function is specified to ", KrigInfo["optimizer"], " by user")

    # Polynomial Order
    if "TrendOrder" not in KrigInfo:
        KrigInfo["TrendOrder"] = 0
    elif KrigInfo["TrendOrder"] < 0:
        raise ValueError("The order of the polynomial trend should be a positive value.")

    # Nugget Setting
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
        if KrigInfo["nuggetparam"][0] > KrigInfo["nuggetparam"][1]:
            raise TypeError("The lower bound of the nugget should be lower than the upper bound.")

    # Kernel Setting
    if "kernel" not in KrigInfo:
        KrigInfo["kernel"] = ["gaussian"]
        nkernel = 1
        print("Kernel is not defined, set kernel to gaussian")
    elif type(KrigInfo["kernel"]) is not list:
        nkernel = 1
        KrigInfo["kernel"] = [KrigInfo["kernel"]]
    else:
        nkernel = len(KrigInfo["kernel"])

    lbhyp = lbtheta
    ubhyp = ubtheta
    KrigInfo["nkernel"] = nkernel

    # Overall Hyperparam
    if nnugget == 1 and nkernel == 1:  # Fixed nugget, one kernel function
        pass
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
    elif nnugget > 1 and nkernel == 1:  # Tunable nugget, one kernel function
        lbhyp = np.hstack((lbhyp, KrigInfo["nugget"][0]))
        ubhyp = np.hstack((ubhyp, KrigInfo["nugget"][1]))
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
        scaling[-1] = (KrigInfo["nugget"][1] - KrigInfo["nugget"][0]) / (ubvalue - lbvalue)
    elif nnugget == 1 and nkernel > 1:  # Fixed nugget, multiple kernel functions
        lbhyp = np.hstack((lbhyp, np.zeros(shape=[nkernel]) + eps))
        ubhyp = np.hstack((ubhyp, np.ones(shape=[nkernel])))
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
        scaling[-nkernel:] = 1 / (ubvalue - lbvalue)
    elif nnugget > 1 and nkernel > 1:  # Tunable nugget, multiple kernel functions
        lbhyp = np.hstack((lbhyp, KrigInfo["nugget"][0], np.zeros(shape=[nkernel]) + eps))
        ubhyp = np.hstack((ubhyp, KrigInfo["nugget"][1], np.ones(shape=[nkernel])))
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
        scaling[-nkernel - 1] = (KrigInfo["nugget"][1] - KrigInfo["nugget"][0]) / (ubvalue - lbvalue)
        scaling[-nkernel:] = 1 / (ubvalue - lbvalue)
    KrigInfo["lbhyp"] = lbhyp
    KrigInfo["ubhyp"] = ubhyp

    if num != None:

        # MULTI OBJECTIVE
        KrigInfo["multiobj"] = True
        KrigInfo["num"] = num

        # KrigInfo["y"][num]= np.transpose(np.array([Y[:,num]]))

        # Create regression matrix
        KrigInfo["idx"][num] = polytruncation(KrigInfo["TrendOrder"], KrigInfo["nvar"], 1)
        # Standardize X and y
        if standardization == True:
            if standtype.lower() == "default":
                KrigInfo["normtype"] = "default"
                bound = np.vstack((- np.ones(shape=[1, KrigInfo["nvar"]]), np.ones(shape=[1, KrigInfo["nvar"]])))
                if normy == True:
                    KrigInfo["X_norm"], KrigInfo["y_norm"][num] = standardize(X, y, type=standtype.lower(), normy=True,
                                                                              range=np.vstack(
                                                                                  (KrigInfo["lb"], KrigInfo["ub"])))
                    KrigInfo["norm_y"] = True
                else:
                    KrigInfo["X_norm"] = standardize(X, y, type=standtype.lower(),
                                                     range=np.vstack((KrigInfo["lb"], KrigInfo["ub"])))
                    KrigInfo["norm_y"] = False
            else:
                KrigInfo["normtype"] = "std"
                if normy == True:
                    KrigInfo["X_norm"], KrigInfo["y_norm"][num], KrigInfo["X_mean"], KrigInfo["y_mean"][num], \
                    KrigInfo["X_std"], KrigInfo["y_std"][num] = standardize(X, y, type=standtype.lower(), normy=True)
                    KrigInfo["norm_y"] = True
                else:
                    KrigInfo["X_norm"], KrigInfo["X_mean"], KrigInfo["X_std"] = standardize(X, y,
                                                                                            type=standtype.lower())
                    KrigInfo["norm_y"] = False
            KrigInfo["standardization"] = True
        else:
            KrigInfo["standardization"] = False
            KrigInfo["norm_y"] = False
        KrigInfo["F"][num] = compute_regression_mat(KrigInfo["idx"][num], KrigInfo["X_norm"], bound, np.ones(shape=[KrigInfo["nvar"]]))

        # Calculate PLS coeff
        _pls = pls(n_princomp)
        coeff_pls = _pls.fit(KrigInfo["X_norm"].copy(), y.copy()).x_rotations_
        KrigInfo["plscoeff"][num]= coeff_pls

        if disp == True:
            print("Multi Objective, train hyperparam, begin.")

        # Find optimum value of Theta
        if KrigInfo["nrestart"] <=1:
            xhyp = nbhyp*[0]
        else:
            _,xhyp = sampling('sobol',nbhyp,KrigInfo['nrestart'],result="real",upbound=ubhyp,lobound=lbhyp)

        bestxcand = np.zeros(shape=[KrigInfo['nrestart'], nbhyp])
        neglnlikecand = np.zeros(shape=[KrigInfo['nrestart']])
        if n_princomp == 1:
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
        KrigInfo["idx"] = polytruncation(KrigInfo["TrendOrder"], KrigInfo["nvar"], 1)
        # Standardize X and y
        if standardization == True:
            if standtype.lower() == "default":
                KrigInfo["normtype"] = "default"
                bound = np.vstack((- np.ones(shape=[1, KrigInfo["nvar"]]), np.ones(shape=[1, KrigInfo["nvar"]])))
                if normy == True:
                    KrigInfo["X_norm"], KrigInfo["y_norm"] = standardize(X, y, type=standtype.lower(), normy=True,range=np.vstack((KrigInfo["lb"][num],KrigInfo["ub"][num])))
                    KrigInfo["norm_y"] = True
                else:
                    KrigInfo["X_norm"] = standardize(X, y, type=standtype.lower(), range=np.vstack((KrigInfo["lb"][num], KrigInfo["ub"][num])))
                    KrigInfo["norm_y"] = False
            else:
                KrigInfo["normtype"] = "std"
                if normy == True:
                    KrigInfo["X_norm"], KrigInfo["y_norm"], KrigInfo["X_mean"], KrigInfo["y_mean"], \
                    KrigInfo["X_std"], KrigInfo["y_std"] = standardize(X, y, type=standtype.lower(), normy=True)
                    KrigInfo["norm_y"] = True
                else:
                    KrigInfo["X_norm"], KrigInfo["X_mean"], KrigInfo["X_std"] = standardize(X, y, type=standtype.lower())
                    KrigInfo["norm_y"] = False
            KrigInfo["standardization"] = True
        else:
            KrigInfo["standardization"] = False
            KrigInfo["norm_y"] = False

        KrigInfo["F"] = compute_regression_mat(KrigInfo["idx"], KrigInfo["X_norm"], bound, np.ones(shape=[KrigInfo["nvar"]]))

        if disp == True:
            print("Single Objective, train hyperparam, begin.")

        # Calculate PLS coeff
        _pls = pls(n_princomp)
        coeff_pls = _pls.fit(KrigInfo["X_norm"].copy(), y.copy()).x_rotations_
        KrigInfo["plscoeff"] = coeff_pls

        # Find optimum value of Theta
        if KrigInfo["nrestart"] <=1:
            xhyp = n_princomp*[0]
        else:
            _,xhyp = sampling('sobol',n_princomp,KrigInfo['nrestart'],result="real",upbound=ubhyp,lobound=lbhyp)

        bestxcand = np.zeros(shape=[KrigInfo['nrestart'],n_princomp])
        neglnlikecand = np.zeros(shape=[KrigInfo['nrestart']])
        if n_princomp == 1:
            if disp == True:
                print("Hyperparam training is repeated for ",KrigInfo['nrestart']," time(s)")
            for ii in range(0, KrigInfo['nrestart']):
                if disp == True:
                    print("hyperparam training attempt number ",ii+1)
                res = minimize_scalar(likelihood.likelihood, bounds=(lbvalue, ubvalue), method='golden',args=(KrigInfo,))
                bestxcand[ii, :] = res.x
                neglnlikecand[ii] = res.fun
            I = np.argmin(neglnlikecand)
            best_x = bestxcand[I, :]
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
