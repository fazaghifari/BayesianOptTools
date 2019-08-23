import numpy as np
from surrogate_models.kriging_model import Kriging
from surrogate_models.supports.initinfo import initkriginfo
from copy import deepcopy
from optim_tools.MOBO import MOBO
import time
from matplotlib import pyplot as plt
from misc.constfunc import sweepdiffcheck
import pandas as pd

class Problem:

    def __init__(self, X, y, cldat):
        self.X = X
        self.y = y
        self.cldat = cldat

    def createkrig(self):
        # define variables
        lb = np.min(self.X, axis=0)
        ub = np.max(self.X, axis=0)

        # Set Const Kriging
        KrigConstInfo = initkriginfo("single")
        KrigConstInfo["X"] = self.X
        KrigConstInfo["y"] = self.cldat.reshape(-1, 1)
        KrigConstInfo["nrestart"] = 5
        KrigConstInfo["ub"] = ub
        KrigConstInfo["lb"] = lb
        KrigConstInfo["optimizer"] = "lbfgsb"
        KrigConstInfo['limittype'] = '>='
        KrigConstInfo['limit'] = 0.15

        # Set Kriging Info
        KrigMultiInfo1 = initkriginfo("single")
        KrigMultiInfo1["X"] = self.X
        KrigMultiInfo1["y"] = self.y[:, 0].reshape(-1, 1)
        KrigMultiInfo1["nrestart"] = 7
        KrigMultiInfo1["ub"] = ub
        KrigMultiInfo1["lb"] = lb
        KrigMultiInfo1["optimizer"] = "lbfgsb"

        KrigMultiInfo2 = deepcopy(KrigMultiInfo1)
        KrigMultiInfo2['y'] = self.y[:, 1].reshape(-1, 1)

        self.krigobj1 = Kriging(KrigMultiInfo1, standardization=True, standtype='default', normy=False, trainvar=False)
        self.krigobj1.train(parallel=False)
        loocverr1, _ = self.krigobj1.loocvcalc()

        self.krigobj2 = Kriging(KrigMultiInfo2, standardization=True, standtype='default', normy=False, trainvar=False)
        self.krigobj2.train(parallel=False)
        loocverr2, _ = self.krigobj2.loocvcalc()

        self.krigconst = Kriging(KrigConstInfo, standardization=True, standtype='default', normy=False, trainvar=False)
        self.krigconst.train(parallel=False)
        loocverr3, _ = self.krigconst.loocvcalc()

        self.kriglist = [self.krigobj1, self.krigobj2]
        self.expconst = [self.krigconst]

    def update_sample(self):
        moboInfo = dict()
        moboInfo["nup"] = 1
        moboInfo["nrestart"] = 10
        moboInfo["acquifunc"] = "ehvi"
        moboInfo["acquifuncopt"] = "lbfgsb"
        moboInfo["refpoint"] = np.array([0.06, 83])
        cheapconstlist = [self.geomconst]
        mobo = MOBO(moboInfo,self.kriglist,autoupdate=False,multiupdate=5,savedata=False,expconst=self.expconst,
                    chpconst=cheapconstlist)
        xupdate, yupdate, metricall = mobo.run(disp=True)

        return xupdate, yupdate, metricall

    def geomconst(self,vars):
        # constraint 'geomconst' should have input of the design variables
        vars = np.array(vars)
        tip_angle = sweepdiffcheck.sweep_diff(vars[2], vars[4], 0.00165529)
        stat = sweepdiffcheck.min_angle_violated(tip_angle, 7)
        return stat

if __name__ == '__main__':
    df = pd.read_csv('../innout/opt_data_60.csv', sep=',', index_col='Name')
    data = df.values
    X = data[:, 0:6].astype(float)
    y = data[:, 7:9].astype(float)
    cldat = data[:, 6].astype(float)

    optim = Problem(X,y,cldat)
    optim.createkrig()
    xupdate, yupdate, metricall = optim.update_sample()
    clpred = optim.krigconst.predict(xupdate,['pred'])

    totalupdate = np.hstack((xupdate,clpred,yupdate,metricall))
    np.savetxt("../innout/Timnext.csv", totalupdate, delimiter=",",
               header="x,z,le_sweep,dihedral,root_chord,root_tc,CL,CD,dB(A),metric", comments="")

    plt.scatter(y[:, 0], y[:, 1], label='initial samples')
    plt.scatter(yupdate[:, 0], yupdate[:, 1], label='predicted next samples')
    plt.ylabel('dB(A)')
    plt.xlabel('CD')
    plt.legend()
    plt.show()
