from platypus import NSGAII, Problem, Real, nondominated
import numpy as np
from copy import deepcopy
from surrogate_models.kriging_model import Kriging
from surrogate_models.supports.initinfo import initkriginfo
from misc.constfunc import sweepdiffcheck
import pandas as pd
from matplotlib import pyplot as plt


class Hispeedplane(Problem):

    def __init__(self):
        super(Hispeedplane, self).__init__(6, 2, 2)
        self.types[:] = [Real(5.04167e-02, 9.9583e-02), Real(-4.425e-03, 4.425e-03), Real(-6.3833e+01,7.3833e+01),
                         Real(-1.475e+01,1.475e+01), Real(4.29167e-02,1.094167e-01), Real(2.067e-02,9.933e-02)]
        self.constraints[:] = ">=0.15"
        self.krigconst = object
        self.krigobj1 = object
        self.krigobj2 = object
        self.createkrig()

    def createkrig(self):
        df = pd.read_csv('../innout/opt_data.csv', sep=',', index_col='Name')
        data = df.values
        X = data[:, 0:6].astype(float)
        y = data[:, 7:9].astype(float)
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

    def geomconst(self,vars):
        vars = np.array(vars)
        tip_angle = sweepdiffcheck.sweep_diff(vars[2], vars[4], 0.00165529)
        stat = sweepdiffcheck.min_angle_violated(tip_angle, 7)
        return stat  # return 1 or 0, 1 is larger than 0.15 then the constraint is satisfied

    def evaluate(self, solution):
        vars = np.array(solution.variables)
        solution.objectives[:] = [self.krigobj1.predict(vars,'pred'), self.krigobj2.predict(vars,'pred')]
        solution.constraints[:] = [self.krigconst.predict(vars,'pred'), self.geomconst(vars)]


if __name__ == '__main__':
    prob1 = Hispeedplane()
    algorithm = NSGAII(prob1)
    algorithm.run(1000)

    nondominated_solutions = nondominated(algorithm.result)

    df = pd.read_csv('../innout/opt_data.csv', sep=',', index_col='Name')
    data = df.values
    X = data[:, 0:6].astype(float)
    y = data[:, 7:9].astype(float)

    nondom1 = np.array([s.objectives[0] for s in nondominated_solutions]).reshape(-1, 1)
    nondom2 = np.array([s.objectives[1] for s in nondominated_solutions]).reshape(-1, 1)
    var = [s.variables for s in nondominated_solutions]
    con = [s.constraint_violation for s in nondominated_solutions]
    var = np.array(var)
    nondom = np.hstack((nondom1, nondom2))
    predCL = prob1.krigconst.predict(var, 'pred')
    total = np.hstack((var, predCL))
    total = np.hstack((total, nondom))
    np.savetxt("../innout/Timnsga2next.csv", total, delimiter=",", header="x,z,le_sweep,dihedral,root_chord,root_tc,CL,"
                                                                         "CD,dB(A)", comments="")
    plt.scatter(y[:, 0], y[:, 1], label='initial samples')
    plt.scatter(nondom1, nondom2, label='nondom soln')
    plt.xlabel("CD")
    plt.ylabel("dB(A)")
    plt.show()
