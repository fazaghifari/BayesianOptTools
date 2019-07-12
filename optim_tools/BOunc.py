import numpy as np
from copy import deepcopy
from optim_tools.acquifun_opt import run_acquifun_opt, run_multi_acquifun_opt
from optim_tools.parego import paregopre
from surrogate_models.kriging import kriging, kpls
from miscellaneous.surrogate_support.prediction import prediction
from testcase.analyticalfcn.cases import evaluate
from miscellaneous.surrogate_support.initinfo import initkriginfo,copymultiKrigInfo
from miscellaneous.sampling.samplingplan import standardize
from miscellaneous.surrogate_support.likelihood import likelihood
from miscellaneous.surrogate_support.trendfunction import polytruncation, compute_regression_mat
from optim_tools import searchpareto
import math
import scipy.io as sio

def sobounc(BayesInfo,KrigInfoBayes,auto=True,**kwargs):
    """
    Perform unconstrained single-objective Bayesian optimization

    Inputs:
        BayesInfo - Structure containing necessary information for Bayesian optimization.
        KrigInfoBayes - Structure containing information of the constructed initial Kriging of the objective function.

    Outputs:
        xbest - Best solution observed after optimization.
        ybest - Best response after optimization.
        yhist - History of best solution observed.
        KrigNewInfo - Structure containing information of final Kriging after optimization.
    """

    krigfun = kwargs.get('krigtype',kriging)
    norm_y = kwargs.get('normalize_y', True)
    # Check necessary parameters
    if "nup" not in BayesInfo:
        if auto is True:
            raise ValueError("Number of updates for Bayesian optimization, BayesInfo['nup'], is not specified")
        else:
            BayesInfo["nup"] = 1
            print("Number of updates for Bayesian optimization has been set to 1")

    # Set default values
    if "stalliteration" not in BayesInfo:
        if auto is True:
            BayesInfo["stalliteration"] = int(np.floor(BayesInfo["nsamp"]/2))
            print("The number of stall iteration is not specified, set to nsamp/2.")
        else:
            BayesInfo["stalliteration"] = 1
            print("Number of stall iteration for Bayesian optimization has been set to 1")
    else:
        if auto is True:
            print("The number of stall iteration is specified to ", BayesInfo["stalliteration"]," by user")
        else:
            BayesInfo["stalliteration"] = 1
            print("Number of stall iteration for Bayesian optimization has been set to 1")

    if "acquifunc" not in BayesInfo:
        BayesInfo["acquifunc"] = "EI"
        print("The acquisition function is not specified, set to EI")
    else:
        availacqfun = ["ei","pred","lcb","poi","ebe"]
        if BayesInfo["acquifunc"].lower() not in availacqfun:
            raise TypeError(BayesInfo["acquifunc"]," is not a valid acquisition function.")
        print("The acquisition function is specified to ", BayesInfo["acquifunc"], " by user")
        if BayesInfo["acquifunc"].lower() == "lcb":
            if "sigmalcb" not in BayesInfo:
                BayesInfo["sigmalcb"] = 3
                print("The sigma for lower confidence bound is not specified, set to 3.")
            else:
                print("The sigma for lower confidence bound is specified to ", BayesInfo["sigmalcb"], " by user")

    if "acquifuncopt" not in BayesInfo:
        BayesInfo["acquifuncopt"] = "sampling+cmaes"
        print("The acquisition function optimizer is not specified, set to sampling+cmaes.")
    else:
        availacqfunopt = ["sampling+cmaes","sampling+fmincon","cmaes","fmincon"]
        if BayesInfo["acquifuncopt"].lower() not in availacqfunopt:
            raise TypeError(BayesInfo["acquifuncopt"]," is not a valid acquisition function optimizer.")
        print("The acquisition function optimizer is specified to ", BayesInfo["acquifuncopt"], " by user")

    if "nrestart" not in BayesInfo:
        BayesInfo["nrestart"] = 1
        print("The number of restart for acquisition function optimization is not specified, setting BayesInfo.nrestart to 1.")
    else:
        if BayesInfo["nrestart"] < 1:
            raise ValueError("BayesInfo['nrestart'] should be at least one")
        print("The number of restart for acquisition function optimization is specified to ", BayesInfo["nrestart"], " by user")


    # RUN BAYESIAN OPTIMIZATION
    nup = 0;
    KrigNewInfo = deepcopy(KrigInfoBayes)
    yhist = np.array([np.min(KrigNewInfo["y"])])
    istall = 0
    print("Begin Bayesian optimization process.")
    if auto is True:
        print("Iteration: ",nup,", F-count: ",np.size(KrigNewInfo["X"],0)," Best f(x): ",yhist[nup]," Stall counter: ", istall)
    else:
        pass

    while nup <= BayesInfo["nup"]:
        # Perform one iteration of single-objective Bayesian optimization
        xnext,_ = run_acquifun_opt(BayesInfo,KrigNewInfo)

        # Break Loop if auto is False
        if auto == False:
            break
        else:
            pass

        # Evaluate new sample
        ynext = eval('evaluate(xnext,KrigNewInfo["problem"])')

        # Give treatment to failed solutions, Reference : "Forrester, A. I., SÃ³bester, A., & Keane, A. J. (2006). Optimization with missing data.
        # Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 462(2067), 935-945."
        if math.isnan(ynext) ==  True:
            SSqr, y_hat = prediction(xnext,KrigNewInfo,["SSqr","pred"])
            ynext = y_hat+SSqr

        # Enrich the experimental design
        KrigNewInfo["X"] = np.vstack((KrigNewInfo["X"],xnext))
        KrigNewInfo["y"] = np.vstack((KrigNewInfo["y"],ynext))
        #Re-Create Kriging model
        KrigNewInfo = krigfun(KrigNewInfo,standardization=True,normalize_y=norm_y)

        nup = nup+1
        yhist = np.vstack((yhist,np.min(KrigNewInfo["y"])))

        #Check stall iteration
        if yhist[nup,0] == yhist[nup-1,0]:
            istall = istall + 1;
            if istall == BayesInfo["stalliteration"]:
                break
        else:
            istall = 0

        print("Iteration: ", nup, ", F-count: ", np.size(KrigNewInfo["X"], 0),", x: ", xnext,", f(x): ", ynext, ", Best f(x): ", yhist[nup], ", Stall counter: ", istall)

    print("Optimization finished, now creating the final outputs.")

    I = np.argmin(KrigNewInfo["y"])
    ybest = KrigNewInfo["y"][I,:]
    if auto == True:
        xbest = KrigNewInfo["X"][I, :]
        Xout = xbest
        print("The best feasible value is ", ybest)
    else:
        Xout = xnext
        print("Suggested next sample: ",xnext,", F-count: ",np.size(KrigNewInfo["X"],0))
        print("The current best value is ", ybest)

    return (Xout, ybest, yhist, KrigNewInfo)

def mobounc(BayesMultiInfo,KrigInfoBayesMulti,auto=True,multiupdate=0,**kwargs):
    """
    Perform unconstrained multi-objective Bayesian optimization

    Inputs:
      BayesMultiInfo - Structure containing necessary information for multi-objective Bayesian optimization.
      KrigInfoBayesMulti - Nested Structure containing information of the constructed initial Kriging of the objective function.

    Outputs:
      Xbest - Matrix of final non-dominated solutions observed after optimization.
      Ybest - Matrix of responses of final non-dominated solutions after optimization.
      KrigNewMultiInfo - Nested structure containing information of final Kriging models after optimization.
    """
    norm_y = kwargs.get('normalize_y', True)
    # Check necessary parameters
    if "nup" not in BayesMultiInfo:
        if auto is True:
            raise ValueError("Number of updates for Bayesian optimization, BayesMultiInfo['nup'], is not specified")
        else:
            BayesMultiInfo["nup"] = 1
            print("Number of updates for Bayesian optimization has been set to 1")
    else:
        if auto == True:
            pass
        else:
            BayesMultiInfo["nup"] = 1
            print("Manual mode is active, number of updates for Bayesian optimization is forced to 1")

    # Set default values
    if "acquifunc" not in BayesMultiInfo:
        BayesMultiInfo["acquifunc"] = "EHVI"
        print("The acquisition function is not specified, set to EHVI")
    else:
        availacqfun = ["ehvi","parego"]
        if BayesMultiInfo["acquifunc"].lower() not in availacqfun:
            raise TypeError(BayesMultiInfo["acquifunc"]," is not a valid acquisition function.")
        else:
            print("The acquisition function is specified to ", BayesMultiInfo["acquifunc"], " by user")

    # Set necessary params for multiobjective acquisition function
    if BayesMultiInfo["acquifunc"].lower() == "ehvi":
        BayesMultiInfo["krignum"] = np.size(KrigInfoBayesMulti["y"],0)
        if "refpoint" not in BayesMultiInfo:
            BayesMultiInfo["refpointtype"] = 'dynamic'
    elif BayesMultiInfo["acquifunc"].lower() == "parego":
        BayesMultiInfo["krignum"] = 1
        if "paregoacquifunc" not in BayesMultiInfo:
            BayesMultiInfo["paregoacquifunc"] = "EI"

    # If BayesMultiInfo.acquifuncopt (optimizer for the acquisition function) is not specified set to 'sampling+cmaes'
    if "acquifuncopt" not in BayesMultiInfo:
        BayesMultiInfo["acquifuncopt"] = "sampling+cmaes"
        print("The acquisition function optimizer is not specified, set to sampling+cmaes.")
    else:
        availableacqoptimizer = ['sampling+cmaes','sampling+fmincon','cmaes','fmincon']
        if BayesMultiInfo["acquifuncopt"].lower() not in availableacqoptimizer:
            raise TypeError(BayesMultiInfo["acquifuncopt"], " is not a valid acquisition function optimizer.")
        else:
            print("The acquisition function optimizer is specified to ", BayesMultiInfo["acquifuncopt"], " by user")

    if "nrestart" not in BayesMultiInfo:
        BayesMultiInfo["nrestart"] = 1
        print("The number of restart for acquisition function optimization is not specified, setting BayesInfo.nrestart to 1.")
    else:
        if BayesMultiInfo["nrestart"] < 1:
            raise ValueError("BayesInfo['nrestart'] should be at least one")
        print("The number of restart for acquisition function optimization is specified to ", BayesMultiInfo["nrestart"], " by user")

    if "filename" not in BayesMultiInfo:
        BayesMultiInfo["filename"] = "temporarydata.mat"
        print("The file name for saving the results is not specified, set the name to temporarydata.mat")
    else:
        print("The file name for saving the results is not specified, set the name to ",BayesMultiInfo["filename"] )


    ### RUN BAYESIAN OPTIMIZATION ###
    nup = 0
    KrigNewMultiInfo = KrigInfoBayesMulti # Initialize Kriging Model

    Xall = KrigNewMultiInfo["X"]
    yall = np.zeros(shape=[np.size(KrigNewMultiInfo["y"][0],axis=0),len(KrigNewMultiInfo["y"])])
    for ii in range(0,len(KrigNewMultiInfo["y"])):
        yall[:,ii] = KrigNewMultiInfo["y"][ii][:,0]
    ypar,_ = searchpareto.paretopoint(yall)

    print("Begin multi-objective Bayesian optimization process.")
    if auto ==  True:
        print("Iteration: ", nup,", F-count: ",np.size(KrigNewMultiInfo["X"],0)," Maximum number of updates): ",BayesMultiInfo["nup"])
    else:
        pass

    KrigScalarizedInfo = copymultiKrigInfo(KrigInfoBayesMulti, 0)
    if BayesMultiInfo["krignum"] == 1:
        KrigScalarizedInfo["y"] = paregopre(yall)
        KrigScalarizedInfo = kriging(KrigScalarizedInfo, standardization=True, normalize_y=norm_y)

    while nup <= BayesMultiInfo["nup"]:

        # Iteratively update the reference point for hypervolume computation if EHVI is used as the acquisition function
        if "refpointtype" in BayesMultiInfo:
            if BayesMultiInfo["refpointtype"].lower() == "dynamic":
                BayesMultiInfo["refpoint"] = np.max(yall,0)+(np.max(yall,0)-np.min(yall,0))*2

        # Perform one iteration of multi-objective Bayesian optimization
        if BayesMultiInfo["krignum"] == 1:
            xnext,ehvinext = run_acquifun_opt(BayesMultiInfo,KrigScalarizedInfo)
        else:
            xnext,ehvinext = run_multi_acquifun_opt(BayesMultiInfo,KrigNewMultiInfo,ypar)

        # perform multi update
        if multiupdate == 0 or multiupdate == 1:
            pass
        else:
            yalltemp = deepcopy(yall)
            Xalltemp = deepcopy(Xall)
            yprednext = np.zeros(shape=[len(KrigNewMultiInfo["y"])])
            KrigNewMultiInfotemp = deepcopy(KrigNewMultiInfo)
            Xalltemp, yalltemp = simultpred(multiupdate, KrigNewMultiInfotemp, BayesMultiInfo, KrigScalarizedInfo,
                                            yprednext, xnext, yalltemp, Xalltemp)
            xnext = Xalltemp[-multiupdate:]
            ynextpredicted = yalltemp[-multiupdate:]

        # Break Loop if auto is False
        if auto == False:
            break
        else:
            pass

        # Evaluate new sample
        if np.ndim(xnext) == 1:
            ynext = eval('evaluate(xnext,KrigInfoBayesMulti["problem"])')
        else:
            ynext = np.zeros(shape=[np.size(xnext,0),len(KrigNewMultiInfo["y"])])
            for ii in range(0,np.size(xnext,0)):
                ynext[ii,:] = eval('evaluate(xnext[ii,:],KrigInfoBayesMulti["problem"])')

        # Give treatment to failed solutions, Reference : "Forrester, A. I., SÃ³bester, A., & Keane, A. J. (2006). Optimization with missing data.
        # Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 462(2067), 935-945."
        if math.isnan(ynext.any()) == True:
            for jj in range(0,len(KrigInfoBayesMulti["y"])):
                if BayesMultiInfo.krignum == 1:
                    KrigNewMultiInfo = kriging(KrigNewMultiInfo,standardization=True,num=jj)
                SSqr, y_hat = prediction(xnext, KrigNewMultiInfo, ["SSqr", "pred"],num=jj)
                ynext[0,jj] = y_hat + SSqr

        # Enrich experimental design
        KrigNewMultiInfo["X"] = np.vstack((KrigNewMultiInfo["X"],xnext))
        for jj in range(0, len(KrigInfoBayesMulti["y"])):
            if np.ndim(ynext) == 1:
                ynext = np.array([ynext])
            KrigNewMultiInfo["y"][jj] = np.vstack((KrigNewMultiInfo["y"][jj],ynext[:,jj].reshape(-1,1)))
            # Re-create Kriging models if multiple Kriging methods are used.
        if BayesMultiInfo["krignum"] > 1:
            for jj in range(0, len(KrigInfoBayesMulti["y"])):
                KrigNewMultiInfo = kriging(KrigNewMultiInfo,standardization=True,normalize_y=norm_y,num=jj)
        else:
            pass

        yall = np.vstack((yall,ynext))
        Xall = np.vstack((Xall,xnext))
        ypar,_ = searchpareto.paretopoint(yall) # Recompute non-dominated solutions

        # Pre-process solution for ParEGO
        if BayesMultiInfo["krignum"] == 1:
            KrigScalarizedInfo["X"] = np.vstack((KrigScalarizedInfo["X"],xnext))
            KrigScalarizedInfo["y"] = paregopre(yall)
            KrigScalarizedInfo = kriging(KrigScalarizedInfo,standardization=True)

        nup = nup+1 # Update the number of iterations
        # Show the optimization progress.
        print("Iteration: ",nup,", F-count: ",np.size(KrigNewMultiInfo["X"],0)," Maximum number of updates): ",BayesMultiInfo["nup"])

        # Save results
        ybest,I = searchpareto.paretopoint(yall)
        I = I.astype(int)
        Xbest = Xall[I,:]
        sio.savemat(BayesMultiInfo["filename"],{"xbest":Xbest,"ybest":ybest,"KrigNewMultiInfo":KrigNewMultiInfo})

    if BayesMultiInfo["krignum"] == 1:
        print("ParEGO finished. Now retraining the Kriging models for each individual models.")
        for jj in range (0, len(KrigInfoBayesMulti["y"])):
            KrigNewMultiInfo = kriging(KrigNewMultiInfo, standardization=True, num=jj)

    print("Optimization finished, now creating the final outputs.")

    ybest,I = searchpareto.paretopoint(yall)
    I = I.astype(int)

    if auto == True:
        Xbest = Xall[I, :]
        Xout = Xbest
        yout = ybest
    else:
        Xout = xnext
        yout = ynextpredicted

    return (Xout, yout, KrigNewMultiInfo)

def simultpred(multiupdate,KrigNewMultiInfotemp,BayesMultiInfo,KrigScalarizedInfo,yprednext,xnext,yalltemp,Xalltemp):
    for ii in range(0, multiupdate):
        KrigNewMultiInfotemp["X"] = np.vstack((KrigNewMultiInfotemp["X"], xnext))
        bound = np.vstack((- np.ones(shape=[1, KrigNewMultiInfotemp["nvar"]]),
                           np.ones(shape=[1, KrigNewMultiInfotemp["nvar"]])))

        for jj in range(0, len(KrigNewMultiInfotemp["y"])):
            yprednext[jj] = prediction(xnext, KrigNewMultiInfotemp, ["pred"], num=jj)
            KrigNewMultiInfotemp["y"][jj] = np.vstack((KrigNewMultiInfotemp["y"][jj], yprednext[jj]))

        Y = KrigNewMultiInfotemp["y"]
        Y = np.hstack((Y[0], Y[1]))
        KrigNewMultiInfotemp["X_norm"], y_normtemp = standardize(KrigNewMultiInfotemp["X"], Y, type="default", normy= True,
                                                     range=np.vstack((np.hstack((KrigNewMultiInfotemp["lb"],np.min(Y,0))),
                                                                      np.hstack((KrigNewMultiInfotemp["ub"],np.max(Y,0))))))

        KrigNewMultiInfotemp["y_norm"][0] = y_normtemp[:,0]
        KrigNewMultiInfotemp["y_norm"][1] = y_normtemp[:,1]
        for jj in range(0, len(KrigNewMultiInfotemp["y"])):
            KrigNewMultiInfotemp["F"][jj] = compute_regression_mat(KrigNewMultiInfotemp["idx"][jj],
                                                                   KrigNewMultiInfotemp["X_norm"], bound,
                                                                   np.ones(
                                                                       shape=[KrigNewMultiInfotemp["nvar"]]))
        for jj in range(0, len(KrigNewMultiInfotemp["y"])):
            KrigNewMultiInfotemp["num"] = jj
            xinput = np.hstack((KrigNewMultiInfotemp["Theta"][jj],np.log10(KrigNewMultiInfotemp["SigmaSqr"][jj]) ))
            KrigNewMultiInfotemp = likelihood(xinput, KrigNewMultiInfotemp, retresult="all")
            
        yalltemp = np.vstack((yalltemp, yprednext))
        Xalltemp = np.vstack((Xalltemp, xnext))
        ypartemp, _ = searchpareto.paretopoint(yalltemp)
        if BayesMultiInfo["krignum"] == 1:
            xnext, ehvinext = run_acquifun_opt(BayesMultiInfo, KrigScalarizedInfo)
        else:
            xnext, ehvinext = run_multi_acquifun_opt(BayesMultiInfo, KrigNewMultiInfotemp, ypartemp)

    return (Xalltemp,yalltemp)
