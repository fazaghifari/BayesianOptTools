import sys
sys.path.insert(0, "..")
import numpy as np
from reliability_analysis.akmcs import AKMCS,mcpopgen
from testcase.RA.testcase import evaluate
from surrogate_models.kriging_model import Kriging
from surrogate_models.kpls_model import KPLS
from surrogate_models.kkpca_model import KKPCA
from sensitivity_analysis.sobol_ind import SobolIndices as SobolI
from surrogate_models.supports.initinfo import initkriginfo
import time


def generate_krig(init_samp, n_krigsamp, nvar,problem):

    # Monte Carlo Sampling
    init_krigsamp = krigsamp()
    ykrig = evaluate(init_krigsamp, type=problem)
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
    KrigInfo["n_princomp"] = 4
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
    E12 = mcpopgen(type="lognormal", ndim=2, n_order=1, n_coeff=3, stddev=2.1e10, mean=2.1e11)
    A1 = mcpopgen(type="lognormal", ndim=1, n_order=1, n_coeff=3, stddev=2e-4, mean=2e-3)
    A2 = mcpopgen(type="lognormal", ndim=1, n_order=1, n_coeff=3, stddev=1e-4, mean=1e-3)
    P = mcpopgen(type="gumbel", ndim=6, n_order=1, n_coeff=3, stddev=7.5e3, mean=5e4)
    all = np.hstack((E12, A1, A2, P))
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

    init_samp_G = np.loadtxt('../innout/out/bridge3_gx.csv', delimiter=',').reshape(-1,1)

    subs = np.transpose((init_samp_G - Gx))
    subs1 = np.transpose((init_samp_G - Gx) / init_samp_G)
    RMSE = np.sqrt(np.sum(subs ** 2) / nsamp)
    RMSRE = np.sqrt(np.sum(subs1 ** 2) / nsamp)
    MAPE = 100 * np.sum(abs(subs1)) / nsamp
    print("RMSE = ", RMSE)
    print("MAPE = ", MAPE, "%")
    mean = np.mean(Gx)
    stdev = np.std(Gx)
    return MAPE, RMSE, mean, stdev

def sensitivity(krigobj,init_samp,nvar):
    lb = (np.min(init_samp, axis=0))
    ub = (np.max(init_samp, axis=0))
    lb = np.hstack((lb,lb))
    ub = np.hstack((ub,ub))
    testSA = SobolI(nvar, krigobj, None, ub, lb)
    result = testSA.analyze(True, True, True)
    for key in result.keys():
        print(key+':')
        if type(result[key]) is not dict:
            print(result[key])
        else:
            pass
            # for subkey in result[key].keys():
            #     print(subkey+':', result[key][subkey])

    return result

if __name__ == '__main__':
    init_samp = np.loadtxt('../innout/in/bridge3.csv', delimiter=',')

    for i in range(50):
        print("--"*25)
        print("loop no.",i+1)
        print("--" * 25)
        nvar = 10
        n_krigsamp = 30
        problem = 'bridge'

        # Create Kriging model
        t = time.time()
        krigobj, loocverr, drm = generate_krig(init_samp, n_krigsamp, nvar, problem)
        ktime = time.time() - t
        # Predict and UQ
        MAPE, RMSE, mean, stdev = pred(krigobj, init_samp, problem, drmmodel=drm)
        # Sensitivity Analysis
        t1 = time.time()
        result = sensitivity(krigobj, init_samp, nvar)
        SAtime = time.time() - t1
        print("time: ", SAtime, " s")

        # Create UQ and Acc test output file
        temparray = np.array([krigobj.KrigInfo['NegLnLike'], loocverr, RMSE, MAPE, mean, stdev, SAtime, ktime])
        if i == 0:
            totaldata = temparray[:]
        else:
            totaldata = np.vstack((totaldata, temparray))

        # np.savetxt('../innout/out/bridge/acctest_bridge_60samp_KPLS1.csv', totaldata, fmt='%10.5f', delimiter=',',
        #            header='Neglnlike,LOOCV Error,RMSE,MAPE,Mean,Std Dev,SA time,Krig time')

        # Create SA output file
        mylist = []
        for ii in range(10):
            mylist.append("S" + str(ii + 1) + ", ")
        for ii in range(10):
            mylist.append("St" + str(ii + 1) + ", ")
        for label in result['second'].keys():
            mylist.append(label+", ")
        SAhead = ""
        for header in mylist:
            SAhead += header
        second = np.fromiter(result['second'].values(), dtype=float)
        saresult = np.array([np.hstack((result['first'], result['total'], second))])
        if i == 0:
            sadata = saresult[:]
        else:
            sadata = np.vstack((sadata, saresult))
        # np.savetxt('../innout/out/bridge/acctest_bridge_60samp_KPLS1_SA.csv', sadata, fmt='%10.5f', delimiter=',',
        #            header=SAhead)