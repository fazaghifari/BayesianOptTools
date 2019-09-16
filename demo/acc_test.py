import sys
sys.path.insert(0, "..")
import numpy as np
from reliability_analysis.akmcs import AKMCS,mcpopgen
from testcase.RA.testcase import evaluate
from surrogate_models.kriging_model import Kriging
from surrogate_models.kpls_model import KPLS
from surrogate_models.supports.initinfo import initkriginfo
import time


def generate_krig(init_samp,n_krigsamp,nvar,problem):

    # Monte Carlo Sampling
    t1 = time.time()
    init_krigsamp = mcpopgen(type="lognormal",ndim=nvar,n_order=1,n_coeff=7, stddev=0.2, mean=1)
    ykrig = evaluate(init_krigsamp, type=problem)
    t2 = time.time()
    print("50 samp eval", t2 - t1)

    init_samp_G = evaluate(init_samp, type=problem)
    total_samp = np.hstack((init_samp, init_samp_G)).transpose()
    positive_samp = total_samp[:, total_samp[nvar] >= 0]
    positive_samp = positive_samp.transpose()
    nsamp = np.size(init_samp, 0)
    npos = np.size(positive_samp, 0)
    Pfreal = 1 - npos / nsamp

    lb = np.floor(np.min(init_samp)) * np.ones(shape=[nvar])
    ub = np.ceil(np.max(init_samp)) * np.ones(shape=[nvar])

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
    # KrigInfo["n_princomp"] = 1
    KrigInfo["optimizer"] = "lbfgsb"

    #trainkrig
    t = time.time()
    krigobj = Kriging(KrigInfo, standardization=True, standtype='default', normy=False, trainvar=False)
    krigobj.train(parallel=False)
    loocverr, _ = krigobj.loocvcalc()
    elapsed = time.time() - t
    print("elapsed time for train Kriging model: ", elapsed, "s")
    print("LOOCV error of Kriging model: ", loocverr, "%")

    return krigobj,Pfreal


def pred(krigobj, init_samp, problem):

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
                Gx[start:stop, :]=  krigobj.predict(init_samp[start:stop, :], ['pred'])
            else:
                Gx[start:, :] = krigobj.predict(init_samp[start:, :], ['pred'])

    init_samp_G = evaluate(init_samp, type=problem)

    subs = np.transpose((init_samp_G - Gx))
    subs1 = np.transpose((init_samp_G - Gx) / init_samp_G)
    RMSE = np.sqrt(np.sum(subs ** 2) / nsamp)
    RMSRE = np.sqrt(np.sum(subs1 ** 2) / nsamp)
    MAPE = 100 * np.sum(abs(subs1)) / nsamp
    print("RMSE = ", RMSE)
    print("MAPE = ", MAPE, "%")


if __name__ == '__main__':
    init_samp = np.loadtxt('../innout/lognormal100.csv', delimiter=',')
    for i in range(5):
        print("--"*25)
        print("loop no.",i+1)
        print("--" * 25)
        nvar = 100
        n_krigsamp = 70
        problem = 'hidimenra'
        filename = "krig_"+str(i+1)+".csv"

        krigobj,Pfreal = generate_krig(init_samp,n_krigsamp,nvar,problem)
        pred(krigobj,init_samp,problem)
        print(Pfreal)