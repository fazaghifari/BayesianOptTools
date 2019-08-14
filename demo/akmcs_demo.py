import numpy as np
from reliability_analysis.akmcs import AKMCS,mcpopgen
from testcase.RA.testcase import evaluate
from surrogate_models.kriging_model import Kriging
from surrogate_models.kpls_model import KPLS
from surrogate_models.supports.initinfo import initkriginfo
import time


def generate_krig(init_samp,n_krigsamp,nvar,problem):

    # Monte Carlo Sampling
    init_krigsamp = mcpopgen(type="lognormal", stddev=0.2, ndim=nvar, n_order=1, n_coeff=5, mean=1)
    ykrig = evaluate(init_krigsamp, type=problem)

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
    KrigInfo["kernel"] = ["gaussian"]
    KrigInfo["nugget"] = -6
    KrigInfo["nkernel"] = len(KrigInfo["kernel"])
    KrigInfo["LOOCVerror"] = 0
    KrigInfo["LOOCVpred"] = 0
    KrigInfo["n_princomp"] = 2
    KrigInfo["optimizer"] = "cobyla"

    #trainkrig
    t = time.time()
    krigobj = KPLS(KrigInfo, standardization=True, standtype='default', normy=False, trainvar=True, disp='INFO')
    krigobj.train(parallel=False)
    loocverr, _ = krigobj.loocvcalc()
    elapsed = time.time() - t
    print("elapsed time for train Kriging model: ", elapsed, "s")
    print("LOOCV error of Kriging model: ", loocverr, "%")

    return krigobj,Pfreal

def run_akmcs(krigobj,init_samp,problem):
    akmcsInfo = dict()
    akmcsInfo["init_samp"] = init_samp
    akmcsInfo["maxupdate"] = 50
    akmcsInfo["problem"] = problem
    akmcsInfo["krigtype"] = "kpls"
    t = time.time()
    akmcsobj = AKMCS(krigobj,akmcsInfo)
    akmcsobj.run()
    elapsed = time.time() - t
    print("elapsed time is : ", elapsed, "s")

if __name__ == '__main__':
    nvar = 40
    n_krigsamp = 50
    problem = 'hidimenra'
    init_samp = mcpopgen(type="lognormal", stddev=0.2, ndim=nvar, n_order=5, n_coeff=3, mean=1)

    krigobj,Pfreal = generate_krig(init_samp,n_krigsamp,nvar,problem)
    run_akmcs(krigobj,init_samp,problem)
    print(Pfreal)