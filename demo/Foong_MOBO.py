import sys
sys.path.insert(0, "..")
import numpy as np
from surrogate_models.kriging_model import Kriging
from surrogate_models.supports.initinfo import initkriginfo
from copy import deepcopy
from optim_tools.MOBO import MOBO
import time
from matplotlib import pyplot as plt
from misc.constfunc.FoongConst import foongconst
import pandas as pd

class Problem:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def createkrig(self):
        # define variables
        lb = np.array([0.1, 3, 3.1])
        ub = np.array([0.9, 11.41, 60.95])

        # Set kriging Info
        KrigMultiInfo1 = initkriginfo("single")
        KrigMultiInfo1["X"] = self.X
        KrigMultiInfo1["y"] = self.y[:, 0].reshape(-1, 1)
        KrigMultiInfo1["nrestart"] = 7
        KrigMultiInfo1["ub"] = ub
        KrigMultiInfo1["lb"] = lb
        KrigMultiInfo1["optimizer"] = "lbfgsb"

        KrigMultiInfo2 = deepcopy(KrigMultiInfo1)
        KrigMultiInfo2['y'] = self.y[:, 1].reshape(-1, 1)

        # train kriging
        self.krigobj1 = Kriging(KrigMultiInfo1, standardization=True, standtype='default', normy=False, trainvar=False)
        self.krigobj1.train(parallel=False)
        loocverr1, _ = self.krigobj1.loocvcalc()
        print(f"LOOCV Error Kriging 1: {loocverr1}% (MAPE)")

        self.krigobj2 = Kriging(KrigMultiInfo2, standardization=True, standtype='default', normy=False, trainvar=False)
        self.krigobj2.train(parallel=False)
        loocverr2, _ = self.krigobj2.loocvcalc()
        print(f"LOOCV Error Kriging 2: {loocverr2}% (MAPE)")

        self.kriglist = [self.krigobj1, self.krigobj2]

    def update_sample(self):
        moboInfo = dict()
        moboInfo["nup"] = 1
        moboInfo["nrestart"] = 10
        moboInfo["acquifunc"] = "ehvi"
        moboInfo["acquifuncopt"] = "lbfgsb"
        cheapconstlist = [self.geomconst]
        mobo = MOBO(moboInfo, self.kriglist, autoupdate=False, multiupdate=5, savedata=False, chpconst=cheapconstlist)
        xupdate, yupdate, metricall = mobo.run(disp=True)

        return xupdate, yupdate, metricall

    def geomconst(self,vars):
        # constraint 'geomconst' should have input of the design variables
        vars = np.array(vars)
        stat = foongconst(vars)
        return stat

if __name__ == '__main__':
    df = pd.read_csv('../innout/foongtestcase_ini.csv', sep=',', index_col='Model')
    data = df.values
    X = data[:, 0:3].astype(float)
    y = data[:, 3:].astype(float)

    optim = Problem(X,y)
    optim.createkrig()
    xupdate, yupdate, metricall = optim.update_sample()

    totalupdate = np.hstack((xupdate,yupdate,metricall))
    np.savetxt("../innout/Foongnextdata.csv", totalupdate, delimiter=",",
               header="n, theta, beta, CD, OP, metric")

    plt.scatter(y[:, 0], y[:, 1])
    plt.scatter(yupdate[:, 0], yupdate[:, 1])
    plt.ylabel('OP')
    plt.xlabel('CD')
    plt.show()