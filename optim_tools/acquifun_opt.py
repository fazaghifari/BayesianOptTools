import numpy as np
from miscellaneous.sampling.samplingplan import realval
from miscellaneous.surrogate_support.prediction import prediction
from scipy.optimize import minimize
from optim_tools.ehvi.EHVIcomputation import ehvicalc
import cma

def run_acquifun_opt(BayesInfo, KrigNewInfo, **kwargs):
    """
     Run the optimization of acquisition to find the next sampling point.

    Inputs:
      BayesInfo - A structure containing necessary information for Bayesian optimization.
      KrigNewInfo - A structure containing information of the constructed Kriging of the objective function.
      KrigNewConInfo -A nested structure containing information of the constructed Kriging of the constraint function.

    Output:
      xnext - Suggested next sampling point as discovered by the optimization of the acquisition function
      fnext - Optimized acquisition function

    The available optimizers for the acquisition function are 'sampling+fmincon', 'sampling+cmaes', 'cmaes', 'fmincon'.
    Note that this function runs for both unconstrained and constrained single-objective Bayesian optimization.
    """
    KrigNewConInfo = kwargs.get("KrigNewConInfo",np.array([None]))
    acquifuncopt = BayesInfo["acquifuncopt"]
    acquifunc = BayesInfo["acquifunc"]

    if acquifunc.lower() == "parego":
        acquifunc = BayesInfo["paregoacquifunc"]

    if BayesInfo["acquifunc"].lower() == "lcb":
        KrigNewInfo["sigmalcb"] = BayesInfo["sigmalcb"]
    pass

    if KrigNewConInfo.any() == None:
        probtype = 1 #Unconstrained problem
    else:
        probtype = 2 #Constrained problem
        ncon = len(KrigNewConInfo)

    if acquifuncopt.lower() == "sampling+fmincon":
        nmcs = int(2e5)
        Xrand = realval(KrigNewInfo["lb"], KrigNewInfo["ub"], np.random.rand(nmcs,KrigNewInfo["nvar"]))
        acquifuncval = prediction(Xrand,KrigNewInfo,acquifunc)
        lbfgsbbound = np.array([KrigNewInfo["lb"][0], KrigNewInfo["ub"][0]])
        lbfgsbbound = np.matlib.repmat(lbfgsbbound, len(Xrand[0, :]), 1)
        if probtype == 1:
            I = np.argmin(acquifuncval)
            res = minimize(prediction,Xrand[I,:],method='L-BFGS-B',bounds=lbfgsbbound,args=(KrigNewInfo,acquifunc))
            xnext = res.x
            fnext = res.fun
        elif probtype == 2:
            PoF = np.empty(shape=[nmcs,ncon])
            for jj in range(0,ncon):
                PoF[:,jj] = prediction(Xrand,KrigNewConInfo[jj],"PoF")
            if ncon > 1:
                conacquifuncval = acquifuncval*PoF
            else:
                conacquifuncval = acquifuncval*np.prod(PoF,1)
            I = np.argmin(conacquifuncval)
            res = minimize(constrainedacqfun,Xrand[I,:],method='L-BFGS-B',bounds=lbfgsbbound,args=(KrigNewInfo,KrigNewConInfo,acquifunc))
            xnext = res.x
            fnext = res.fun
            pass

    elif acquifuncopt.lower() == "sampling+cmaes":
        nmcs = int(2e5)
        Xrand = realval(KrigNewInfo["lb"], KrigNewInfo["ub"], np.random.rand(nmcs, KrigNewInfo["nvar"]))
        acquifuncval = prediction(Xrand, KrigNewInfo, acquifunc)
        sigmacmaestemp = (KrigNewInfo["ub"] - KrigNewInfo["lb"])/5
        sigmacmaes = np.max(sigmacmaestemp)
        scaling = sigmacmaestemp/sigmacmaes
        if probtype == 1:
            I = np.argmin(acquifuncval)
            xnext,es = cma.fmin2(prediction,Xrand[I,:],sigmacmaes,{'BoundaryHandler': cma.BoundPenalty,'bounds': [KrigNewInfo["lb"].tolist(), KrigNewInfo["ub"].tolist()], 'scaling_of_variables':scaling,'verb_disp': 0,'verbose': -9},args=(KrigNewInfo,acquifunc))
            fnext = es.result[1]
        elif probtype == 2:
            PoF = np.empty(shape=[nmcs, ncon])
            for jj in range(0, ncon):
                PoF[:, jj] = prediction(Xrand, KrigNewConInfo[jj], "PoF")
            if ncon > 1:
                conacquifuncval = acquifuncval*PoF
            else:
                conacquifuncval = acquifuncval*np.prod(PoF,1)
            I = np.argmin(conacquifuncval)
            xnext,es = cma.fmin2(constrainedacqfun,Xrand[I,:],sigmacmaes,{'BoundaryHandler': cma.BoundPenalty,'bounds': [KrigNewInfo["lb"].tolist(), KrigNewInfo["ub"].tolist()],'verbose': -9},args=(KrigNewInfo,KrigNewConInfo,acquifunc))
            fnext = es.result[1]

    elif acquifuncopt.lower() == "cmaes":
        Xrand = realval(KrigNewInfo["lb"], KrigNewInfo["ub"], np.random.rand(BayesInfo["nrestart"], KrigNewInfo["nvar"]))
        xnextcand = np.zeros(shape=[BayesInfo["nrestart"],KrigNewInfo["nvar"]])
        fnextcand = np.zeros(shape=[BayesInfo["nrestart"]])
        sigmacmaestemp = (KrigNewInfo["ub"] - KrigNewInfo["lb"]) / 4
        sigmacmaes = np.max(sigmacmaestemp)
        for im in range(0,BayesInfo["nrestart"]):
            if probtype == 1:
                xnextcand[im,:],es = cma.fmin2(prediction,Xrand[im,:],sigmacmaes,{'BoundaryHandler': cma.BoundPenalty,'bounds': [KrigNewInfo["lb"].tolist(), KrigNewInfo["ub"].tolist()], 'verb_disp': 0,'verbose': -9},args=(KrigNewInfo,acquifunc))
                fnextcand[im] = es.result[1]
            elif probtype == 2:
                xnextcand[im,:],es = cma.fmin2(constrainedacqfun,Xrand[im,:],sigmacmaes,{'BoundaryHandler': cma.BoundPenalty,'bounds': [KrigNewInfo["lb"].tolist(), KrigNewInfo["ub"].tolist()]},args=(KrigNewInfo,KrigNewConInfo,acquifunc))
                fnextcand[im] = es.result[1]
        I = np.argmin(fnextcand)
        xnext = xnextcand[I,:]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == "fmincon":
        Xrand = realval(KrigNewInfo["lb"], KrigNewInfo["ub"], np.random.rand(BayesInfo["nrestart"], KrigNewInfo["nvar"]))
        xnextcand = np.zeros(shape=[BayesInfo["nrestart"], KrigNewInfo["nvar"]])
        fnextcand = np.zeros(shape=[BayesInfo["nrestart"]])
        lbfgsbbound = np.vstack((KrigNewInfo["lb"],KrigNewInfo["ub"])).transpose()
        for im in range(0,BayesInfo["nrestart"]):
            if probtype == 1:
                res = minimize(prediction,Xrand[im,:],method='L-BFGS-B',bounds=lbfgsbbound,args=(KrigNewInfo,acquifunc))
                xnextcand[im,:] = res.x
                fnextcand[im] = res.fun
            elif probtype == 2:
                res = minimize(constrainedacqfun,Xrand[im,:],method='L-BFGS-B',bounds=lbfgsbbound,args=(KrigNewInfo,KrigNewConInfo,acquifunc))
                xnextcand[im, :] = res.x
                fnextcand[im] = res.fun
        I = np.argmin(fnextcand)
        test = prediction(np.array([10,15]),KrigNewInfo,"EI")
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    return (xnext,fnext)

def constrainedacqfun(x,KrigNewInfo,KrigNewConInfo,acquifunc):
    #Calculate unconstrained acquisition function
    acquifuncval = prediction(x,KrigNewInfo,acquifunc)

    #Calculate probability of feasibility for each constraint
    ncon = len(KrigNewConInfo)
    PoF = np.zeros(shape=[ncon])
    for ii in range(0,ncon):
        PoF[ii] = prediction(x,KrigNewConInfo[ii],'PoF')

    if ncon > 1:
        conacquifuncval = acquifuncval * PoF
    else:
        conacquifuncval = acquifuncval * np.prod(PoF, 1)

    pass

    return conacquifuncval

def run_multi_acquifun_opt(BayesMultiInfo, KrigNewMultiInfo, ypar, **kwargs):
    """
    Run the optimization of multi-objective acquisition function to find the next sampling point.

    Inputs:
      BayesMultiInfo - A structure containing necessary information for Bayesian optimization.
      KrigNewMultiInfo - A structure containing information of the constructed Kriging of the objective function.
      KrigNewConInfo -A nested structure containing information of the constructed Kriging of the constraint function.

    Output:
      xnext - Suggested next sampling point as discovered by the optimization of the acquisition function
      fnext - Optimized acquisition function

    The available optimizers for the acquisition function are 'sampling+fmincon', 'sampling+cmaes', 'cmaes', 'fmincon'.
    Note that this function runs for both unconstrained and constrained single-objective Bayesian optimization.
    """
    KrigNewConInfo = kwargs.get("KrigConInfo",np.array([None]))
    acquifuncopt = BayesMultiInfo["acquifuncopt"]
    acquifunc = BayesMultiInfo["acquifunc"]

    if KrigNewConInfo.all() == None:
        probtype = 1 # Unconstrained problem
    else:
        probtype = 2 # Constrained problem
        ncon = len(KrigNewConInfo)

    if acquifunc.lower() == "ehvi":
        acqufunhandle = ehvicalc
    else:
        pass

    if acquifuncopt.lower() == "cmaes":
        Xrand = realval(KrigNewMultiInfo["lb"], KrigNewMultiInfo["ub"], np.random.rand(BayesMultiInfo["nrestart"], KrigNewMultiInfo["nvar"]))
        xnextcand = np.zeros(shape=[BayesMultiInfo["nrestart"], KrigNewMultiInfo["nvar"]])
        fnextcand = np.zeros(shape=[BayesMultiInfo["nrestart"]])
        sigmacmaes = (KrigNewMultiInfo["ub"] - KrigNewMultiInfo["lb"]) / 6
        for im in range(0,BayesMultiInfo["nrestart"]):
            if probtype == 1:# For unconstrained problem
                xnextcand[im,:],es = cma.fmin2(acqufunhandle,Xrand[im,:],sigmacmaes,{'BoundaryHandler': cma.BoundPenalty,'bounds': [KrigNewMultiInfo["lb"].tolist(), KrigNewMultiInfo["ub"].tolist()],'verb_disp': 0,'verbose': -9},args=(ypar,BayesMultiInfo,KrigNewMultiInfo))
                fnextcand[im] = es.result[1]
            elif probtype == 2: # For constrained problem (on progress)
                xnextcand[im,:],es = cma.fmin2(acqufunhandle,Xrand[im,:],sigmacmaes,{'BoundaryHandler': cma.BoundPenalty,'bounds': [KrigNewMultiInfo["lb"].tolist(), KrigNewMultiInfo["ub"].tolist()],'verb_disp': 0,'verbose': -9},args=(ypar,BayesMultiInfo,KrigNewMultiInfo))
                fnextcand[im] = es.result[1]
        I = np.argmin(fnextcand)
        xnext = xnextcand[I,:]
        fnext = fnextcand[I]
    elif acquifuncopt.lower() == "fmincon":
        Xrand = realval(KrigNewMultiInfo["lb"], KrigNewMultiInfo["ub"], np.random.rand(BayesMultiInfo["nrestart"], KrigNewMultiInfo["nvar"]))
        xnextcand = np.zeros(shape=[BayesMultiInfo["nrestart"], KrigNewMultiInfo["nvar"]])
        fnextcand = np.zeros(shape=[BayesMultiInfo["nrestart"]])
        lbfgsbbound = np.hstack((KrigNewMultiInfo["lb"].reshape(-1, 1), KrigNewMultiInfo["ub"].reshape(-1, 1)))
        for im in range(0,BayesMultiInfo["nrestart"]):
            if probtype == 1:# For unconstrained problem
                res = minimize(acqufunhandle,Xrand[im,:],method='L-BFGS-B',bounds=lbfgsbbound,args=(ypar,BayesMultiInfo,KrigNewMultiInfo))
                xnextcand[im,:] = res.x
                fnextcand[im] = res.fun
            elif probtype == 2: # For constrained problem (on progress)
                res = minimize(acqufunhandle,Xrand[im,:],method='L-BFGS-B',bounds=lbfgsbbound,args=(ypar,BayesMultiInfo,KrigNewMultiInfo))
                xnextcand[im, :] = res.x
                fnextcand[im] = res.fun
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    return (xnext,fnext)
