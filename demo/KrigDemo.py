import sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt
from surrogate_models.kriging_model import Kriging
from surrogate_models.kpls_model import KPLS
from misc.sampling.samplingplan import sampling,realval,standardize
from testcase.analyticalfcn.cases import evaluate
from surrogate_models.supports.initinfo import initkriginfo
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def generate_kriging():
    # Initialization
    KrigInfo = dict()
    kernel = ["gaussian"]
    # Sampling
    nsample = 25
    nvar = 2
    ub = np.array([10, 15])
    lb = np.array([-5, 0])
    nup = 3
    sampoption = "halton"
    samplenorm, sample = sampling(sampoption, nvar, nsample, result="real", upbound=ub, lobound=lb)
    X = sample
    # Evaluate sample
    y1 = evaluate(X, "branin")

    # Initialize KrigInfo
    KrigInfo = initkriginfo("single")
    # Set KrigInfo
    KrigInfo["X"] = X
    KrigInfo["y"] = y1
    KrigInfo["nvar"] = nvar
    KrigInfo["problem"] = "branin"
    KrigInfo["nsamp"] = nsample
    KrigInfo["nrestart"] = 7
    KrigInfo["ub"] = ub
    KrigInfo["lb"] = lb
    KrigInfo["kernel"] = kernel
    KrigInfo["TrendOrder"] = 0
    KrigInfo["nugget"] = -6
    KrigInfo["n_princomp"] = 1
    KrigInfo["optimizer"] = "slsqp"

    # Run Kriging
    t = time.time()
    krigobj = Kriging(KrigInfo,standardization=True,standtype='default',normy=False,trainvar=False)
    krigobj.train(parallel=False)
    loocverr, _ = krigobj.loocvcalc()
    elapsed = time.time() - t
    print("elapsed time for train Kriging model: ", elapsed, "s")
    print("LOOCV error of Kriging model: ", loocverr, "%")

    return krigobj


def predictkrig(krigobj):
    nsample = 25
    nvar = 2
    ub = np.array([10, 15])
    lb = np.array([-5, 0])
    nup = 3
    sampoption = "halton"

    # Test Kriging Output
    neval = 10000
    samplenormout, sampleeval = sampling(sampoption, nvar, neval, result="real", upbound=ub, lobound=lb)
    xx = np.linspace(-5, 10, 100)
    yy = np.linspace(0, 15, 100)
    Xevalx, Xevaly = np.meshgrid(xx, yy)
    Xeval = np.zeros(shape=[neval, 2])
    Xeval[:, 0] = np.reshape(Xevalx, (neval))
    Xeval[:, 1] = np.reshape(Xevaly, (neval))

    # Evaluate output
    yeval = np.zeros(shape=[neval, 1])
    yact = np.zeros(shape=[neval, 1])
    yeval = krigobj.predict(Xeval,'pred')
    yact = evaluate(Xeval, "branin")
    hasil = np.hstack((yeval, yact))

    # Evaluate RMSE
    subs = np.transpose((yact - yeval))
    subs1 = np.transpose((yact - yeval) / yact)
    RMSE = np.sqrt(np.sum(subs ** 2) / neval)
    RMSRE = np.sqrt(np.sum(subs1 ** 2) / neval)
    MAPE = 100 * np.sum(abs(subs1)) / neval
    print("RMSE = ", RMSE)
    print("RMSRE = ", RMSRE)
    print("MAPE = ", MAPE, "%")

    yeval1 = np.reshape(yeval, (100, 100))
    x1eval = np.reshape(Xeval[:, 0], (100, 100))
    x2eval = np.reshape(Xeval[:, 1], (100, 100))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x1eval, x2eval, yeval1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

if __name__ == '__main__':
    krigingobj1 = generate_kriging()
    predictkrig(krigingobj1)