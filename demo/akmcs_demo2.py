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
    init_krigsamp = krigsamp()
    ykrig = evaluate(init_krigsamp, type=problem)
    print(np.count_nonzero(ykrig <= 0))

    Pfreal = None
    if runmc is True:
        Pfreal = evalmc(init_samp,problem)

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

    return krigobj,Pfreal


def krigsamp():
    E12 = mcpopgen(type="lognormal", ndim=2, n_order=1, n_coeff=1.2, stddev=2.1e10, mean=2.1e11)
    A1 = mcpopgen(type="lognormal", ndim=1, n_order=1, n_coeff=1.2, stddev=2e-4, mean=2e-3)
    A2 = mcpopgen(type="lognormal", ndim=1, n_order=1, n_coeff=1.2, stddev=1e-4, mean=1e-3)
    P = mcpopgen(type="gumbel", ndim=6, n_order=1, n_coeff=1.2, stddev=7.5e3, mean=5e4)
    all = np.hstack((E12, A1, A2, P))
    return all


def evalmc(init_samp, problem):
    init_samp_G = evaluate(init_samp, type=problem)
    total_samp = np.hstack((init_samp, init_samp_G)).transpose()
    positive_samp = total_samp[:, total_samp[nvar] >= 0]
    positive_samp = positive_samp.transpose()
    nsamp = np.size(init_samp, 0)
    npos = np.size(positive_samp, 0)
    Pfreal = 1 - npos / nsamp
    np.savetxt('../innout/bridge3_gx.csv',init_samp_G,delimiter=',')

    return Pfreal


def run_akmcs(krigobj,init_samp,problem,filename):

    # Define AKMCS Information
    akmcsInfo = dict()
    akmcsInfo["init_samp"] = init_samp
    akmcsInfo["maxupdate"] = 70
    akmcsInfo["problem"] = problem

    # Run AKMCS
    t = time.time()
    akmcsobj = AKMCS(krigobj,akmcsInfo)
    akmcsobj.run(savedatato=filename)
    elapsed = time.time() - t
    print("elapsed time is : ", elapsed, "s")

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

        # krigobj,Pfreal = generate_krig(init_samp,n_krigsamp,nvar,problem)
        # run_akmcs(krigobj,init_samp,problem,filename)
        t1 = time.time()
        Pfreal = evalmc(init_samp,problem)
        t2 = time.time()
        print('Runtime:',t2-t1)
        print(Pfreal)