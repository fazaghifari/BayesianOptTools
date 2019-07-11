import numpy as np
from miscellaneous.sampling.samplingplan import realval
from miscellaneous.surrogate_support.prediction import prediction
from surrogate_models.kriging import kriging,kpls
from testcase.RA.testcase import evaluate
import matplotlib.pyplot as plt
from scipy.stats import mode
import time

def akmcs(KrigInfo,akmcsInfo):

    # If conditional set default
    if "init_samp" not in akmcsInfo:
        raise ValueError('akmcsInfo["init_samp"] must be defined')
    else:
        pass

    if "maxupdate" not in akmcsInfo:
        akmcsInfo["maxupdate"] = 120
        print("Maximum update is set to ", akmcsInfo["maxupdate"])
    else:
        print("Maximum update is set to ", akmcsInfo["maxupdate"]," by user.")

    if "problem" not in akmcsInfo:
        raise ValueError('akmcsInfo["problem"] must be defined')
    else:
        pass

    if "krigtype" not in akmcsInfo:
        akmcsInfo["krigtype"] = "kriging"
        print("Kriging type is set to ", akmcsInfo["krigtype"])
    else:
        availmod = ["kriging", "kpls"]
        if akmcsInfo["krigtype"].lower() not in availmod:
            raise TypeError(akmcsInfo["krigtype"], " is not a valid acquisition function.")
        print("Kriging type is set to ", akmcsInfo["krigtype"], " by user")

    # Get important variables
    init_samp = akmcsInfo["init_samp"]
    maxupdate = akmcsInfo["maxupdate"]

    # Run Kriging
    if akmcsInfo["krigtype"].lower() == "kriging":
        t = time.time()
        KrigInfo = kriging(KrigInfo, standardization=True, normtype="std", normalize_y=False, disp=True, loocvcalc=True)
        elapsed = time.time() - t
        print("elapsed time for train Kriging model: ", elapsed, "s")
        print("LOOCV Error Kriging : ", KrigInfo["LOOCVerror"], " % (MAPE)")
    elif akmcsInfo["krigtype"].lower() == "kpls":
        t = time.time()
        KrigInfo = kpls(KrigInfo, standardization=True, normtype="std", normalize_y=False, disp=True, loocvcalc=True)
        elapsed = time.time() - t
        print("elapsed time for train Kriging model: ", elapsed, "s")
        print("LOOCV Error Kriging : ", KrigInfo["LOOCVerror"], " % (MAPE)")
    else:
        raise ValueError("Kriging Model not available.")

    # Predict all Monte Carlo Samples
    Gx,sigmaG = prediction(init_samp, KrigInfo, ["pred","s"])

    # Calculate Prob of Failure
    Pf = pf(Gx,init_samp)

    # Calculate Learning Function U
    updateX = np.zeros(shape=[2])
    minUiter = np.zeros(shape=[1])
    Uval, xnew, Uall = LFU(init_samp, Gx, sigmaG)

    # plotting
    # plotting(Uall, sigmaG, init_samp, KrigInfo["X"], "initial")

    while 1 > 0 :

        for i in range(0,maxupdate):
            Uval,xnew,Uall = LFU(init_samp, Gx, sigmaG)

            if i == 0:
                updateX = np.array([xnew])
                minUiter = np.array([Uval])
            else:
                updateX = np.vstack((updateX,xnew))
                minUiter = np.vstack((minUiter,Uval))

            ynew = evaluate(xnew, type=akmcsInfo["problem"])
            KrigInfo["y"] = np.vstack((KrigInfo["y"],ynew))
            KrigInfo["X"] = np.vstack((KrigInfo["X"],xnew))
            KrigInfo["nsamp"] = KrigInfo["nsamp"]+1

            if akmcsInfo["krigtype"].lower() == "kriging":
                KrigInfo = kriging(KrigInfo, standardization=True, normtype="std", normalize_y=False, disp=False)
            elif akmcsInfo["krigtype"].lower() == "kpls":
                KrigInfo = kpls(KrigInfo, standardization=True, normtype="std", normalize_y=False, disp=False)
            else:
                pass

            Gx, sigmaG = prediction(init_samp, KrigInfo, ["pred", "s"])
            Pf = pf(Gx, init_samp)
            cov = COVPf(Pf, init_samp)
            print(f"done iter no : {i+1}, Pf : {Pf}")

            if i >= 190:
                #plotting
                plotting(Uall,sigmaG,init_samp, KrigInfo["X"],i+1, updateX=updateX)

            if Uval >= 2 and i >= 15:
                break
            else:
                pass

        if cov <= 0.05:
            break
        else:
            pass
        break

    return minUiter,updateX

def pf(Gx, init_samp):
    nGless = len([i for i in Gx if i <= 0])
    nsamp = np.size(init_samp, axis=0)
    Pf = nGless / nsamp
    return Pf

def COVPf (Pf, init_samp):
    nmc = np.size(init_samp, axis=0)
    if Pf == 0:
        cov = 1000
    else:
        cov = np.sqrt((1-Pf)/(Pf*nmc))
    return cov

def LFU(initsamp,Gx,sigmaG):
    U = abs(Gx) / sigmaG.reshape(-1,1)
    minU = np.min(U)
    minUloc = np.argmin(U)
    xmin = initsamp[minUloc,:]
    return (minU,xmin,U)

def mcpopgen(lb=None,ub=None,n_order=6,n_coeff=1,type="random",sigma=1,ndim=2):
    nmc = int(n_coeff*10**n_order)
    if type.lower()== "gaussian":
        pop = sigma*np.random.randn(nmc,ndim)
    elif type.lower() == "lognormal":
        pop = np.random.lognormal(0,sigma)
    elif type.lower()== "random":
        if lb.any() == None or ub.any() == None:
            raise ValueError("type 'random' is selected, please input lower bound and upper bound value")
        else:
            pop = realval(lb, ub, np.random.rand(nmc,len(lb)))
    else:
        raise ValueError("Monte Carlo sampling type not supported")
    return pop

def initkrigpop(mcsamp,nsamp):
    rand_index = np.random.choice(np.size(mcsamp, 0), nsamp)
    krigsamp = mcsamp[rand_index]
    return krigsamp

def plotting(Ux,sigmaG,init_samp,krigsamp,updateno,updateX=np.array([None])):
    # Plotting
    yeval1 = np.reshape(sigmaG, (250, 250))
    Uxeval = np.reshape(Ux, (250, 250))
    x1eval = np.reshape(init_samp[:, 0], (250, 250))
    x2eval = np.reshape(init_samp[:, 1], (250, 250))
    mins = np.min(sigmaG)
    maxs = np.max(sigmaG)
    modeU = mode(Uxeval,axis=None)

    plt.figure()
    v = np.linspace(mins, maxs, 20, endpoint=True)
    plt.contourf(x1eval, x2eval, yeval1, v, cmap='jet')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.title(f's Value of Trained SigmaSqr Kriging no.{updateno}')
    plt.colorbar(ticks=v)
    if updateX.any() == None:
        plt.scatter(krigsamp[:, 0], krigsamp[:, 1], marker='^', c='dimgrey')
    else:
        plt.scatter(krigsamp[:, 0], krigsamp[:, 1], marker='^', c='dimgrey')
        plt.scatter(updateX[:, 0], updateX[:, 1], marker='x', c='k')

    plt.figure()
    vv = np.linspace(0, modeU[0][0]+3, 20, endpoint=True)
    plt.contourf(x1eval, x2eval, Uxeval, vv, cmap='jet')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.title(f'Uval Value of Trained SigmaSqr Kriging no.{updateno}')
    plt.colorbar(ticks=vv)
    if updateX.any() == None:
        plt.scatter(krigsamp[:, 0], krigsamp[:, 1], marker='^', c='dimgrey')
    else:
        plt.scatter(krigsamp[:, 0], krigsamp[:, 1], marker='^', c='dimgrey')
        plt.scatter(updateX[:, 0], updateX[:, 1], marker='x', c='k')