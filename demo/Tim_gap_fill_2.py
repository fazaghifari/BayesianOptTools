import sys
sys.path.insert(0, "..")
import numpy as np
from copy import deepcopy
from surrogate_models.kriging_model import Kriging
from surrogate_models.supports.initinfo import initkriginfo
from reliability_analysis.akmcs import AKMCS,mcpopgen
from misc.sampling.samplingplan import realval,sampling
from misc.constfunc import sweepdiffcheck
import pandas as pd
from matplotlib import pyplot as plt

class Fillgap:
    def __init__(self, init_samp):
        self.init_samp = init_samp
        self.nsamp = np.size(self.init_samp,axis=0)
        self.cd = np.zeros(shape=[self.nsamp, 1])
        self.noise = np.zeros(shape=[self.nsamp, 1])
        self.cl = np.zeros(shape=[self.nsamp, 1])
        self.solnlist = [self.cd,self.noise,self.cl]

    def createkrig(self):
        df = pd.read_csv('../innout/opt_data(1).csv', sep=',', index_col='Name')
        data = df.values
        X = data[:, 0:6].astype(float)
        y = data[:, 7:9].astype(float)
        self.y = y
        cldat = data[:, 6].astype(float)

        # define variables
        lb = np.min(X, axis=0)
        ub = np.max(X, axis=0)

        # Set Const Kriging
        KrigConstInfo = initkriginfo("single")
        KrigConstInfo["X"] = X
        KrigConstInfo["y"] = cldat.reshape(-1, 1)
        KrigConstInfo["nrestart"] = 5
        KrigConstInfo["ub"] = ub
        KrigConstInfo["lb"] = lb
        KrigConstInfo["optimizer"] = "lbfgsb"
        KrigConstInfo['limit'] = 0.15

        # Set Kriging Info
        KrigMultiInfo1 = initkriginfo("single")
        KrigMultiInfo1["X"] = X
        KrigMultiInfo1["y"] = y[:, 0].reshape(-1, 1)
        KrigMultiInfo1["nrestart"] = 7
        KrigMultiInfo1["ub"] = ub
        KrigMultiInfo1["lb"] = lb
        KrigMultiInfo1["optimizer"] = "slsqp"

        KrigMultiInfo2 = deepcopy(KrigMultiInfo1)
        KrigMultiInfo2['y'] = y[:, 1].reshape(-1, 1)

        self.krigobj1 = Kriging(KrigMultiInfo1, standardization=True, standtype='default', normy=False, trainvar=False)
        self.krigobj1.train(parallel=False)
        loocverr1, _ = self.krigobj1.loocvcalc()

        self.krigobj2 = Kriging(KrigMultiInfo2, standardization=True, standtype='default', normy=False, trainvar=False)
        self.krigobj2.train(parallel=False)
        loocverr2, _ = self.krigobj2.loocvcalc()

        self.krigconst = Kriging(KrigConstInfo, standardization=True, standtype='default', normy=False, trainvar=False)
        self.krigconst.train(parallel=False)
        loocverr3, _ = self.krigconst.loocvcalc()

        self.kriglist = [self.krigobj1,self.krigobj2,self.krigconst]

    def evaluate(self):
        for idx, krigobj in enumerate(self.kriglist):
            self.solnlist[idx] = krigobj.predict(self.init_samp, ['pred'])

        return self.solnlist

if __name__ == '__main__':
    init_samp,_ = sampling('sobol',6,20)
    ub = np.array([9.9583e-02, 4.425e-03, 7.3833e+01, 1.475e+01, 1.094167e-01, 9.933e-02])
    lb = np.array([5.04167e-02, -4.425e-03, -6.3833e+01, -1.475e+01, 4.29167e-02, 2.067e-02])
    init_samp = realval(lb,ub,init_samp)

    fill = Fillgap(init_samp)
    fill.createkrig()
    values = fill.evaluate()

    alldata = init_samp[:,:]
    for item in values:
        alldata = np.hstack((alldata,item))

    np.savetxt("../innout/Tim_fill.csv",alldata,delimiter=',',header="x,z,le_sweep,dihedral,root_chord,root_tc,CD,"
                                                                         "dB(A),CL", comments="")

    plt.scatter(fill.y[:, 0], fill.y[:, 1], label='initial samples')
    plt.scatter(alldata[:,6], alldata[:,7], label='nondom soln')
    plt.xlabel("CD")
    plt.ylabel("dB(A)")
    plt.show()