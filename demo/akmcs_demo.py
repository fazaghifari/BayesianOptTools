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
    init_krigsamp = mcpopgen(type="normal",ndim=nvar,n_order=1,n_coeff=5)
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
    KrigInfo["nkernel"] = len(KrigInfo["kernel"])
    KrigInfo["n_princomp"] = 4
    KrigInfo["optimizer"] = "lbfgsb"

    #trainkrig
    t = time.time()
    krigobj = KPLS(KrigInfo, standardization=True, standtype='default', normy=False, trainvar=False)
    krigobj.train(parallel=False)
    loocverr, _ = krigobj.loocvcalc()
    elapsed = time.time() - t
    print("elapsed time for train Kriging model: ", elapsed, "s")
    print("LOOCV error of Kriging model: ", loocverr, "%")

    return krigobj,Pfreal

def run_akmcs(krigobj,init_samp,problem,filename):

    # Define AKMCS Information
    akmcsInfo = dict()
    akmcsInfo["init_samp"] = init_samp
    akmcsInfo["maxupdate"] = 50
    akmcsInfo["problem"] = problem

    # Run AKMCS
    t = time.time()
    akmcsobj = AKMCS(krigobj,akmcsInfo)
    akmcsobj.run(savedatato=filename)
    elapsed = time.time() - t
    print("elapsed time is : ", elapsed, "s")

if __name__ == '__main__':
    init_samp = np.loadtxt('../innout/akmcssampnorm.csv', delimiter=',')
    for i in range(5):
        print("--"*25)
        print("loop no.",i+1)
        print("--" * 25)
        nvar = 50
        n_krigsamp = 50
        problem = 'hidimenra2'
        filename = "kpls4_"+str(i+1)+".csv"

        krigobj,Pfreal = generate_krig(init_samp,n_krigsamp,nvar,problem)
        run_akmcs(krigobj,init_samp,problem,filename)
        print(Pfreal)