import sys
sys.path.insert(0, "..")
import numpy as np
from surrogate_models.kriging_model import Kriging
from surrogate_models.supports.initinfo import initkriginfo
from copy import deepcopy
from optim_tools.MOBO import MOBO
import time
from matplotlib import pyplot as plt
from misc.constfunc import sweepdiffcheck
from misc.constfunc import constraints_check
import pandas as pd

class Problem:

    def __init__(self, X, y, cldat,area_2):
        self.X = X
        self.y = y
        self.cldat = cldat
        self.area2 = area_2

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

        # KrigAreaInfo = initkriginfo("single")
        # KrigAreaInfo["X"] =  np.delete(self.X, [3,5,10], axis=1)
        # KrigAreaInfo["y"] = self.area2.reshape(-1,1)
        # KrigAreaInfo["nrestart"] = 5
        # KrigAreaInfo["ub"] = np.max(KrigAreaInfo["X"], axis=0)
        # KrigAreaInfo["lb"] = np.min(KrigAreaInfo["X"], axis=0)
        # KrigAreaInfo["optimizer"] = "cobyla"

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
        loocverrCD, _ = self.krigobj1.loocvcalc()

        self.krigobj2 = Kriging(KrigMultiInfo2, standardization=True, standtype='default', normy=False, trainvar=False)
        self.krigobj2.train(parallel=False)
        loocverrNOISE, _ = self.krigobj2.loocvcalc()

        self.krigconst = Kriging(KrigConstInfo, standardization=True, standtype='default', normy=False, trainvar=False)
        self.krigconst.train(parallel=False)
        loocverrCL, _ = self.krigconst.loocvcalc()

        print('LOOCV CD: ', loocverrCD)
        print('LOOCV Noise: ', loocverrNOISE)
        print('LOOCV CL: ', loocverrCL)

        # # Create Kriging for Area (uncomment if needed)
        # self.krigarea = Kriging(KrigAreaInfo, standardization=True, standtype='default', normy=False, trainvar=False)
        # self.krigarea.train(parallel=False)
        # loocverrAREA, _ = self.krigarea.loocvcalc()

        self.kriglist = [self.krigobj1, self.krigobj2]
        self.expconst = [self.krigconst]

    def update_sample(self):
        moboInfo = dict()
        moboInfo["nup"] = 1
        moboInfo["nrestart"] = 10
        moboInfo["acquifunc"] = "ehvi"
        moboInfo["acquifuncopt"] = "ga"
        moboInfo["refpoint"] = np.array([0.06, 83])
        cheapconstlist = [self.geomconst]
        mobo = MOBO(moboInfo,self.kriglist,autoupdate=False,multiupdate=5,savedata=False,expconst=self.expconst,
                    chpconst=cheapconstlist)
        xupdate, yupdate, metricall = mobo.run(disp=True)

        return xupdate, yupdate, metricall

    def geomconst(self,vars):
        # constraint 'geomconst' should have input of the design variables
        # vars = np.array(vars)
        # proj_area_1, area_1, proj_area_2, area_2 = constraints_check.calc_areas(vars[6],vars[4],vars[3],vars[7],vars[9],
        #                                                                         total_proj_area=0.00165529)
        # s1_min = 0.3 * 0.00165529
        # s1_max = 0.9 * 0.00165529
        # s1_satisfied = constraints_check.min_max_satisfied(proj_area_1, min_val=s1_min, max_val=s1_max,disp=False)
        # tip_angle = constraints_check.triangular_tip_angle(vars[8], vars[7], area_2)
        # tip_satisfied = constraints_check.min_max_satisfied(tip_angle, 7,disp=False)
        # stat = s1_satisfied & tip_satisfied
        # return stat

        vars = np.array(vars)
        tip_angle = sweepdiffcheck.sweep_diff(vars[2], vars[4], 0.00165529)
        stat = sweepdiffcheck.min_angle_violated(tip_angle, 7)
        return stat

if __name__ == '__main__':
    df = pd.read_csv('../innout/tim/In/opt_data_AS_old_all4.csv', sep=',', index_col='Name')
    data = df.values
    X = data[:, 0:6].astype(float)
    y = data[:, 7:9].astype(float)
    cldat = data[:, 6].astype(float)
    area_2 = None#data[:, 11].astype(float)

    t = time.time()
    optim = Problem(X,y,cldat,area_2)
    optim.createkrig()
    xupdate, yupdate, metricall = optim.update_sample()
    clpred = optim.krigconst.predict(xupdate,['pred'])
    elapsed = time.time() - t
    # _,_,_,area_2pred = constraints_check.calc_areas(xupdate[:,6],xupdate[:,4],xupdate[:,3],xupdate[:,7],xupdate[:,9],
    #                                                 total_proj_area=0.00165529)
    print('Time required:', elapsed)

    # cycle = np.array(["opt03_AT"]*5).reshape(-1,1)
    # totalupdate = np.hstack((xupdate,area_2pred.reshape(-1,1),cycle,clpred,yupdate,metricall))
    # np.savetxt("../innout/tim/Out/nextpoints3_AT_LBF.csv", totalupdate, delimiter=",",
    #            header="x,z,le_sweep_1,dihedral_1,chord_1,tc_1,proj_span_1,chord_2,le_sweep_2,dihedral_2,tc_2,area_2,cycle,"
    #                   "CL,CD,dB(A),metric", comments="", fmt="%s")

    totalupdate = np.hstack((xupdate, clpred, yupdate, metricall))
    np.savetxt("../innout/tim/Out/nextpoints_oldAS_all5.csv", totalupdate, delimiter=",",
               header="x,z,le_sweep,dihedral,root_chord,root_tc,CL,CD,dB(A),metric", comments="")

    plt.scatter(y[cldat > 0.15, 0], y[cldat > 0.15, 1], c='#1f77b4',label='initial feasible samples')
    plt.scatter(y[cldat <= 0.15, 0], y[cldat <= 0.15, 1],marker='x',c='k',label='initial infeasible samples')
    plt.scatter(yupdate[:, 0], yupdate[:, 1], c='#ff7f0e',label='predicted next samples')
    plt.ylabel('dB(A)')
    plt.xlabel('CD')
    plt.legend()
    plt.show()
