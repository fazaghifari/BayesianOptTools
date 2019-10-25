import sys
sys.path.insert(0, "..")
import numpy as np
from reliability_analysis.akmcs import AKMCS,mcpopgen
from testcase.RA.testcase import evaluate
from surrogate_models.kriging_model import Kriging
from surrogate_models.kpls_model import KPLS
from surrogate_models.supports.initinfo import initkriginfo
import matplotlib.pyplot as plt
import time


def generate_krig(init_samp, n_krigsamp, nvar,problem):

    # Monte Carlo Sampling
    init_krigsamp = krigsamp()
    print("Evaluating Kriging Sample")
    ykrig = evaluate(init_krigsamp,problem)
    print(np.count_nonzero(ykrig <= 0))

    lb = (np.min(init_samp, axis=0))
    ub = (np.max(init_samp, axis=0))

    Pfreal = None

    # Set Kriging Info
    KrigInfo = initkriginfo("single")
    KrigInfo["X"] = init_krigsamp
    KrigInfo["y"] = ykrig
    KrigInfo["nvar"] = nvar
    KrigInfo["nsamp"] = n_krigsamp
    KrigInfo["nrestart"] = 5
    KrigInfo["ub"] = ub
    KrigInfo["lb"] = lb
    KrigInfo["nkernel"] = len(KrigInfo["kernel"])
    KrigInfo["n_princomp"] = 1
    KrigInfo["optimizer"] = "lbfgsb"

    #trainkrig
    drm = None
    t = time.time()
    krigobj = KPLS(KrigInfo, standardization=True, standtype='default', normy=False, trainvar=False)
    krigobj.train(parallel=False)
    loocverr, _ = krigobj.loocvcalc()
    elapsed = time.time() - t
    print("elapsed time for train Kriging model: ", elapsed, "s")
    print("LOOCV error of Kriging model: ", loocverr, "%")

    return krigobj,loocverr,drm

def krigsamp():
    all = mcpopgen(type="normal", ndim=53, n_order=1, n_coeff=5)
    return all

def pred(krigobj, init_samp, problem, drmmodel=None):

    nsamp = np.size(init_samp,axis=0)
    Gx = np.zeros(shape=[nsamp, 1])
    if nsamp < 10000:
        Gx = krigobj.predict(init_samp, ['pred'])
    else:
        run_times = int(np.ceil(nsamp / 10000))
        for i in range(run_times):
            start = i * 10000
            stop = (i + 1) * 10000
            if i != (run_times - 1):
                Gx[start:stop, :]=  krigobj.predict(init_samp[start:stop, :], ['pred'], drmmodel=drmmodel)
            else:
                Gx[start:, :] = krigobj.predict(init_samp[start:, :], ['pred'], drmmodel=drmmodel)

    init_samp_G = np.loadtxt('../innout/out/heatcond22.csv', delimiter=',').reshape(-1,1)

    subs = np.transpose((init_samp_G - Gx))
    subs1 = np.transpose((init_samp_G - Gx) / init_samp_G)
    RMSE = np.sqrt(np.sum(subs ** 2) / nsamp)
    RMSRE = np.sqrt(np.sum(subs1 ** 2) / nsamp)
    MAPE = 100 * np.sum(abs(subs1)) / nsamp
    print("RMSE = ", RMSE)
    print("MAPE = ", MAPE, "%")
    return MAPE


if __name__ == '__main__':
    init_samp = np.loadtxt('../innout/in/heat_samp2.csv', delimiter=',')
    dic = dict()

    for i in range(20):
        print("--"*25)
        print("loop no.",i+1)
        print("--" * 25)
        nvar = 53
        n_krigsamp = 50
        problem = 'heatcond'

        krigobj,loocverr,drm= generate_krig(init_samp,n_krigsamp,nvar,problem)
        MAPE = pred(krigobj,init_samp,problem,drmmodel=drm)
