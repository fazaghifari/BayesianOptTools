import numpy as np
from misc.sampling.samplingplan import realval
from scipy.optimize import minimize, fmin_cobyla
from optim_tools.ehvi.EHVIcomputation import ehvicalc
from misc.constfunc import sweepdiffcheck,FoongConst
import cma

def run_single_opt(krigobj,optInfo):
    pass


def run_multi_opt(kriglist, moboInfo, ypar, constlist=None):
    """
    Run the optimization of multi-objective acquisition function to find the next sampling point.

    Args:
      kriglist (list): A list containing Kriging instances.
      moboInfo (dict): A structure containing necessary information for Bayesian optimization.
      ypar (nparray): Array contains the current non-dominated solutions.

    Returns:
      xnext - Suggested next sampling point as discovered by the optimization of the acquisition function
      fnext - Optimized acquisition function

    The available optimizers for the acquisition function are 'sampling+fmincon', 'sampling+cmaes', 'cmaes', 'fmincon'.
    Note that this function runs for both unconstrained and constrained single-objective Bayesian optimization.
    """
    acquifuncopt = moboInfo["acquifuncopt"]
    acquifunc = moboInfo["acquifunc"]

    if acquifunc.lower() == 'ehvi':
        acqufunhandle = ehvicalc
    else:
        pass

    if acquifuncopt.lower() == 'cmaes':
        Xrand = realval(kriglist[0].KrigInfo["lb"], kriglist[0].KrigInfo["ub"],
                        np.random.rand(moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]))
        xnextcand = np.zeros(shape=[moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[moboInfo["nrestart"]])
        sigmacmaes = 1  # np.mean((KrigNewMultiInfo["ub"] - KrigNewMultiInfo["lb"]) / 6)
        for im in range(0, moboInfo["nrestart"]):
            if constlist is None:  # For unconstrained problem
                xnextcand[im, :], es = cma.fmin2(acqufunhandle, Xrand[im, :], sigmacmaes,
                                                 {'verb_disp': 0,'verbose': -9},
                                                 args=(ypar, moboInfo, kriglist))
                fnextcand[im] = es.result[1]
            else:  # For constrained problem
                xnextcand[im, :], es = cma.fmin2(constfunhandle, Xrand[im, :], sigmacmaes,
                                                 {'verb_disp': 0, 'verbose': -9},
                                                 args=(ypar, moboInfo, kriglist))
                fnextcand[im] = es.result[1]
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == 'lbfgsb':
        Xrand = realval(kriglist[0].KrigInfo["lb"], kriglist[0].KrigInfo["ub"],
                        np.random.rand(moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]))
        xnextcand = np.zeros(shape=[moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[moboInfo["nrestart"]])
        lbfgsbbound = np.hstack((kriglist[0].KrigInfo["lb"].reshape(-1, 1), kriglist[0].KrigInfo["ub"].reshape(-1, 1)))
        for im in range(0,moboInfo["nrestart"]):
            if constlist is None:  # For unconstrained problem
                res = minimize(acqufunhandle,Xrand[im,:],method='L-BFGS-B',bounds=lbfgsbbound,args=(ypar,moboInfo,
                                                                                                    kriglist))
                xnextcand[im,:] = res.x
                fnextcand[im] = res.fun
            else:  # For constrained problem (on progress)
                res = minimize(constfunhandle,Xrand[im,:],method='L-BFGS-B',bounds=lbfgsbbound,args=(ypar, moboInfo, kriglist))
                xnextcand[im, :] = res.x
                fnextcand[im] = res.fun
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == 'cobyla':
        Xrand = realval(kriglist[0].KrigInfo["lb"], kriglist[0].KrigInfo["ub"],
                        np.random.rand(moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]))
        xnextcand = np.zeros(shape=[moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[moboInfo["nrestart"]])
        optimbound = []
        for i in range(len(kriglist[0].KrigInfo["ubhyp"])):
            # params aa and bb are not used, just to avoid error in Cobyla optimizer
            optimbound.append(lambda x, Kriginfo, aa, bb, itemp=i: x[itemp] - kriglist[0].KrigInfo["lbhyp"][itemp])
            optimbound.append(lambda x, Kriginfo, aa, bb, itemp=i: kriglist[0].KrigInfo["ubhyp"][itemp] - x[itemp])
        for im in range(0, moboInfo["nrestart"]):
            if constlist is None:  # For unconstrained problem
                res = fmin_cobyla(acqufunhandle, Xrand[im,:], optimbound,
                                  rhobeg=0.5, rhoend=1e-4, args=(ypar, moboInfo, kriglist))
                xnextcand[im, :] = res
                fnextcand[im] = acqufunhandle(ypar, moboInfo, kriglist)
            else:
                res = fmin_cobyla(constfunhandle, Xrand[im, :], optimbound,
                                  rhobeg=0.5, rhoend=1e-4, args=(ypar, moboInfo, kriglist))
                xnextcand[im, :] = res
                fnextcand[im] = constfunhandle(res, ypar, moboInfo, kriglist)

    return (xnext,fnext)

def constfunhandle():
    pass
