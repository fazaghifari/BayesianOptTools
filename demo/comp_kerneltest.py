import sys
sys.path.insert(0, "..")
import numpy as np
from reliability_analysis.akmcs import AKMCS,mcpopgen
from testcase.RA.testcase import evaluate
from surrogate_models.kriging_model import Kriging
from surrogate_models.kpls_model import KPLS
from surrogate_models.supports.initinfo import initkriginfo
import time


def generate_krig(init_samp,n_krigsamp,nvar,problem,runmc=False):

    # Monte Carlo Sampling
    init_krigsamp = np.loadtxt('../innout/bridge3_krig.csv', delimiter=',')
    ykrig = evaluate(init_krigsamp, type=problem)
    print(np.count_nonzero(ykrig <= 0))

    lb = (np.min(init_samp, axis=0))
    ub = (np.max(init_samp, axis=0))

    # Set Kriging Info
    KrigInfo = initkriginfo("single")
    KrigInfo["X"] = init_krigsamp
    KrigInfo["y"] = ykrig
    KrigInfo["nvar"] = nvar
    KrigInfo["nsamp"] = n_krigsamp
    KrigInfo["nrestart"] = 5
    KrigInfo["ub"] = ub
    KrigInfo["lb"] = lb
    KrigInfo["kernel"] = ["exponential"]
    KrigInfo["nkernel"] = len(KrigInfo["kernel"])
    # KrigInfo["n_princomp"] = 4
    KrigInfo["optimizer"] = "lbfgsb"

    #trainkrig
    t = time.time()
    krigobj = Kriging(KrigInfo, standardization=True, standtype='default', normy=False, trainvar=False)
    krigobj.train(parallel=False)
    loocverr, _ = krigobj.loocvcalc()
    elapsed = time.time() - t
    print("elapsed time for train Kriging model: ", elapsed, "s")
    print("LOOCV error of Kriging model: ", loocverr, "%")

    return krigobj


def krigsamp():
    E12 = mcpopgen(type="lognormal", ndim=2, n_order=1, n_coeff=1.2, stddev=2.1e10, mean=2.1e11)
    A1 = mcpopgen(type="lognormal", ndim=1, n_order=1, n_coeff=1.2, stddev=2e-4, mean=2e-3)
    A2 = mcpopgen(type="lognormal", ndim=1, n_order=1, n_coeff=1.2, stddev=1e-4, mean=1e-3)
    P = mcpopgen(type="gumbel", ndim=6, n_order=1, n_coeff=1.2, stddev=7.5e3, mean=5e4)
    all = np.hstack((E12, A1, A2, P))
    return all

def predictkrig(krigobj, mcs_samp):

    # Evaluate output
    yeval = krigobj.predict(mcs_samp,'pred')
    yact = np.loadtxt('../innout/bridge3_gx.csv', delimiter=',').reshape(-1,1)
    res = np.hstack((yeval, yact))
    neval = 3e5

    # Evaluate RMSE
    subs = np.transpose((yact - yeval))
    subs1 = np.transpose((yact - yeval) / yact)
    RMSE = np.sqrt(np.sum(subs ** 2) / neval)
    RMSRE = np.sqrt(np.sum(subs1 ** 2) / neval)
    MAPE = 100 * np.sum(abs(subs1)) / neval
    print("RMSE = ", RMSE)
    print("RMSRE = ", RMSRE)
    print("MAPE = ", MAPE, "%")


if __name__ == '__main__':
    init_samp = np.loadtxt('../innout/bridge3.csv', delimiter=',')
    for i in range(1):
        print("--"*25)
        print("loop no.",i+1)
        print("--" * 25)
        nvar = 10
        n_krigsamp = 30
        problem = 'bridge'
        filename = "bridge2_krig"+str(i+1)+".csv"

        krigobj = generate_krig(init_samp,n_krigsamp,nvar,problem)
        predictkrig(krigobj,init_samp)