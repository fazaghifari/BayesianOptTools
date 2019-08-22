import numpy as np
from misc.sampling.samplingplan import realval
from scipy.optimize import minimize, fmin_cobyla
from optim_tools.ehvi.EHVIcomputation import ehvicalc
from misc.constfunc import sweepdiffcheck,FoongConst
import cma


def run_single_opt(krigobj, moboInfo, krigconstlist=None, cheapconstlist=None):
    """
   Run the optimization of multi-objective acquisition function to find the next sampling point.

   Args:
     krigobj (object): Kriging object.
     moboInfo (dict): A structure containing necessary information for Bayesian optimization.
     krigconstlist (list): List of Kriging object for constraints. Defaults to None.
     cheapconstlist (list): List of constraints function. Defaults to None.
            Expected output of the constraint functions is 1 if the constraint is satisfied and 0 if not.
            The constraint functions MUST have an input of x (the decision variable to be evaluated)

   Returns:
     xnext (nparray): Suggested next sampling point as discovered by the optimization of the acquisition function
     fnext (nparray): Optimized acquisition function

   The available optimizers for the acquisition function are 'cmaes', 'lbfgsb', 'cobyla'.
   Note that this function runs for both unconstrained and constrained single-objective Bayesian optimization.
   """
    acquifuncopt = moboInfo["acquifuncopt"]
    acquifunc = moboInfo["acquifunc"]

    if acquifunc.lower() == 'parego':
        acquifunc = moboInfo['paregoacquifunc']
    else:
        pass

    if acquifuncopt.lower() == 'cmaes':
        Xrand = realval(krigobj.KrigInfo["lb"], krigobj.KrigInfo["ub"],
                        np.random.rand(moboInfo["nrestart"], krigobj.KrigInfo["nvar"]))
        xnextcand = np.zeros(shape=[moboInfo["nrestart"], krigobj.KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[moboInfo["nrestart"]])
        sigmacmaes = 1  # np.mean((KrigNewMultiInfo["ub"] - KrigNewMultiInfo["lb"]) / 6)
        for im in range(0, moboInfo["nrestart"]):
            if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
                xnextcand[im, :], es = cma.fmin2(krigobj.predict, Xrand[im, :], sigmacmaes,
                                                 {'verb_disp': 0,'verbose': -9},
                                                 args=(acquifunc))
                fnextcand[im] = es.result[1]
            else:  # For constrained problem
                xnextcand[im, :], es = cma.fmin2(singleconstfun, Xrand[im, :], sigmacmaes,
                                                 {'verb_disp': 0, 'verbose': -9},
                                                 args=(krigobj, acquifunc, krigconstlist,cheapconstlist))
                fnextcand[im] = es.result[1]
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == 'lbfgsb':
        Xrand = realval(krigobj.KrigInfo["lb"], krigobj.KrigInfo["ub"],
                        np.random.rand(moboInfo["nrestart"], krigobj.KrigInfo["nvar"]))
        xnextcand = np.zeros(shape=[moboInfo["nrestart"], krigobj.KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[moboInfo["nrestart"]])
        lbfgsbbound = np.hstack((krigobj.KrigInfo["lb"].reshape(-1, 1), krigobj.KrigInfo["ub"].reshape(-1, 1)))
        for im in range(0,moboInfo["nrestart"]):
            if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
                res = minimize(krigobj.predict,Xrand[im,:] ,method='L-BFGS-B', bounds=lbfgsbbound, args=(acquifunc))
                xnextcand[im,:] = res.x
                fnextcand[im] = res.fun
            else:  # For constrained problem (on progress)
                res = minimize(singleconstfun,Xrand[im,:], method='L-BFGS-B', bounds=lbfgsbbound,
                               args=(krigobj, acquifunc, krigconstlist,cheapconstlist))
                xnextcand[im, :] = res.x
                fnextcand[im] = res.fun
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == 'cobyla':
        Xrand = realval(krigobj.KrigInfo["lb"], krigobj.KrigInfo["ub"],
                        np.random.rand(moboInfo["nrestart"], krigobj.KrigInfo["nvar"]))
        xnextcand = np.zeros(shape=[moboInfo["nrestart"], krigobj.KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[moboInfo["nrestart"]])
        optimbound = []
        for i in range(len(krigobj.KrigInfo["ubhyp"])):
            optimbound.append(lambda x, Kriginfo, aa, bb, itemp=i: x[itemp] - krigobj.KrigInfo["lbhyp"][itemp])
            optimbound.append(lambda x, Kriginfo, aa, bb, itemp=i: krigobj.KrigInfo["ubhyp"][itemp] - x[itemp])
        for im in range(0, moboInfo["nrestart"]):
            if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
                res = fmin_cobyla(krigobj.predict, Xrand[im,:], optimbound,
                                  rhobeg=0.5, rhoend=1e-4, args=(acquifunc))
                xnextcand[im, :] = res
                fnextcand[im] = krigobj.predict(res, acquifunc)
            else:
                res = fmin_cobyla(singleconstfun, Xrand[im, :], optimbound,
                                  rhobeg=0.5, rhoend=1e-4, args=(krigobj, acquifunc, krigconstlist,cheapconstlist))
                xnextcand[im, :] = res
                fnextcand[im] = singleconstfun(res, krigobj, acquifunc, krigconstlist,cheapconstlist)
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    return (xnext,fnext)


def run_multi_opt(kriglist, moboInfo, ypar, krigconstlist=None, cheapconstlist=None):
    """
    Run the optimization of multi-objective acquisition function to find the next sampling point.

    Args:
      kriglist (list): A list containing Kriging instances.
      moboInfo (dict): A structure containing necessary information for Bayesian optimization.
      ypar (nparray): Array contains the current non-dominated solutions.
      krigconstlist (list): List of Kriging object for constraints. Defaults to None.
      cheapconstlist (list): List of constraints function. Defaults to None.
            Expected output of the constraint functions is 1 if the constraint is satisfied and 0 if not.
            The constraint functions MUST have an input of x (the decision variable to be evaluated)

    Returns:
      xnext (nparray): Suggested next sampling point as discovered by the optimization of the acquisition function
      fnext (nparray): Optimized acquisition function

    The available optimizers for the acquisition function are 'cmaes', 'lbfgsb', 'cobyla'.
    Note that this function runs for both unconstrained and constrained single-objective Bayesian optimization.
    """
    acquifuncopt = moboInfo["acquifuncopt"]
    acquifunc = moboInfo["acquifunc"]

    if acquifunc.lower() == 'ehvi':
        acqufunhandle = ehvicalc
    else:
        raise ValueError("Acquisition function handle is not available")

    if acquifuncopt.lower() == 'cmaes':
        Xrand = realval(kriglist[0].KrigInfo["lb"], kriglist[0].KrigInfo["ub"],
                        np.random.rand(moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]))
        xnextcand = np.zeros(shape=[moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[moboInfo["nrestart"]])
        sigmacmaes = 1  # np.mean((KrigNewMultiInfo["ub"] - KrigNewMultiInfo["lb"]) / 6)
        for im in range(0, moboInfo["nrestart"]):
            if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
                xnextcand[im, :], es = cma.fmin2(acqufunhandle, Xrand[im, :], sigmacmaes,
                                                 {'verb_disp': 0, 'verbose': -9},
                                                 args=(ypar, moboInfo, kriglist))
                fnextcand[im] = es.result[1]
            else:  # For constrained problem
                xnextcand[im, :], es = cma.fmin2(multiconstfun, Xrand[im, :], sigmacmaes,
                                                 {'verb_disp': 0, 'verbose': -9},
                                                 args=(ypar, kriglist, moboInfo, krigconstlist, cheapconstlist))
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
            if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
                res = minimize(acqufunhandle,Xrand[im,:],method='L-BFGS-B',bounds=lbfgsbbound,args=(ypar,moboInfo,
                                                                                                    kriglist))
                xnextcand[im,:] = res.x
                fnextcand[im] = res.fun
            else:  # For constrained problem (on progress)
                res = minimize(multiconstfun,Xrand[im,:],method='L-BFGS-B',bounds=lbfgsbbound,
                               args=(ypar, kriglist, moboInfo, krigconstlist, cheapconstlist))
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
            optimbound.append(lambda x, Kriginfo, aa, bb, itemp=i: x[itemp] - kriglist[0].KrigInfo["lbhyp"][itemp])
            optimbound.append(lambda x, Kriginfo, aa, bb, itemp=i: kriglist[0].KrigInfo["ubhyp"][itemp] - x[itemp])
        for im in range(0, moboInfo["nrestart"]):
            if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
                res = fmin_cobyla(acqufunhandle, Xrand[im,:], optimbound,
                                  rhobeg=0.5, rhoend=1e-4, args=(ypar, moboInfo, kriglist))
                xnextcand[im, :] = res
                fnextcand[im] = acqufunhandle(res, ypar, moboInfo, kriglist)
            else:
                res = fmin_cobyla(multiconstfun, Xrand[im, :], optimbound,
                                  rhobeg=0.5, rhoend=1e-4, args=(ypar, kriglist, moboInfo, krigconstlist, cheapconstlist))
                xnextcand[im, :] = res
                fnextcand[im] = multiconstfun(res,ypar, kriglist, moboInfo, krigconstlist, cheapconstlist)
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    return xnext,fnext


def singleconstfun(x, krigobj, acquifunc, krigconstlist=None, cheapconstlist=None):
    """
    Calculate the single objective acquisition function value

    Args:
        x (nparray): Decision variable to be evaluated.
        krigobj (object): The kriging object.
        acquifunc (str): Acquisition function metric.
        krigconstlist (list): List of Kriging object for constraints. Defaults to None.
        cheapconstlist (list): List of constraints function. Defaults to None.
            Expected output of the constraint functions is 1 if the constraint is satisfied and 0 if not.
            The constraint functions MUST have an input of x (the decision variable to be evaluated)

    Returns:
        fx (float): The acquisition function value.
    """
    # Calculate unconstrained acquisition function
    acquifuncval = krigobj.predict(x, acquifunc)

    if krigconstlist is not None:

        # Change to list if the type is not list
        if type(krigconstlist) is not list:
            krigconstlist = [krigconstlist]
        else:
            pass

        nkrigcon = len(krigconstlist)

        pof = np.zeros(shape=[nkrigcon])
        for ii in range(nkrigcon):
            pof[ii] = krigconstlist[ii].predict(x, 'PoF')
        pof = np.prod(pof)

    else:
        pof = 1

    if cheapconstlist is None:
        pass

    else:
        # Change to list if the type is not list
        if type(cheapconstlist) is not list:
            cheapconstlist = [cheapconstlist]
        else:
            pass

        coeff = np.zeros(shape=[len(cheapconstlist)])
        for jj in range(len(cheapconstlist)):
            coeff[jj] = cheapconstlist[jj](x)
        coeff = np.prod(coeff)

    fx = pof*coeff*acquifuncval

    return fx


def multiconstfun(x, ypar, kriglist, moboInfo, krigconstlist=None, cheapconstlist=None):
    """
    Calculate the single objective acquisition function value

    Args:
        x (nparray): Decision variable to be evaluated.
        ypar (nparray): Array contains the current non-dominated solutions.
        kriglist (list): List of Kriging object.
        moboInfo (dict): A structure containing necessary information for Bayesian optimization.
        krigconstlist (list): List of Kriging object for constraints. Defaults to None.
        cheapconstlist (list): List of constraints function. Defaults to None.
            Expected output of the constraint functions is 1 if the constraint is satisfied and 0 if not.
            The constraint functions MUST have an input of x (the decision variable to be evaluated)

    Returns:
        fx (float): The acquisition function value.
    """
    acquifunc = moboInfo['acquifunc']
    if acquifunc.lower() == 'ehvi':
        acqufunhandle = ehvicalc
    else:
        pass

    if krigconstlist is not None:

        # Change to list if the type is not list
        if type(krigconstlist) is not list:
            krigconstlist = [krigconstlist]
        else:
            pass

        nkrigcon = len(krigconstlist)

        pof = np.zeros(shape=[nkrigcon])
        for ii in range(nkrigcon):
            pof[ii] = krigconstlist[ii].predict(x, 'PoF')
        pof = np.prod(pof)

    else:
        pof = 1

    if cheapconstlist is None:
        pass

    else:
        # Change to list if the type is not list
        if type(cheapconstlist) is not list:
            cheapconstlist = [cheapconstlist]
        else:
            pass

        coeff = np.zeros(shape=[len(cheapconstlist)])
        for jj in range(len(cheapconstlist)):
            coeff[jj] = cheapconstlist[jj](x)
        coeff = np.prod(coeff)

    metric = acqufunhandle(x, ypar, moboInfo, kriglist)

    fx = pof*coeff*metric

    return fx
