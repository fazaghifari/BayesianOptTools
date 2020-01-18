import sys
sys.path.insert(0, "..")
from platypus import NSGAII, Problem, Real, nondominated
import numpy as np
from copy import deepcopy
from surrogate_models.kriging_model import Kriging
from surrogate_models.supports.initinfo import initkriginfo
from misc.constfunc import sweepdiffcheck
from misc.constfunc import constraints_check
import pandas as pd
from matplotlib import pyplot as plt


class Hispeedplane(Problem):

    def __init__(self):
        super(Hispeedplane, self).__init__(11, 2, 2)
        self.types[:] = [Real(5.04167e-02, 9.9583e-02), Real(-4.425e-03, 4.425e-03), Real(-6.3833e+01,7.3833e+01),
                         Real(-2.9864e+01,2.905e+01), Real(3.0181e-02,1.094167e-01), Real(2.067e-02,9.982e-02),
                         Real(1.0113122e-02,4.382353e-02), Real(2.1201357e-02,1.0988e-01),Real(-6.468326e+01,7.3833e+01),
                         Real(-4.35746e+01,4.15385e+01), Real(2.066667e-02,9.934e-02)]
        self.constraints[:] = ">=0.15"
        self.krigconst = object
        self.krigobj1 = object
        self.krigobj2 = object
        self.createkrig()

    def createkrig(self):
        df = pd.read_csv('../innout/tim/opt_data_ASAT1.csv', sep=',', index_col='code')
        data = df.values
        X = data[:, 0:11].astype(float)
        y = data[:, 14:16].astype(float)
        cldat = data[:, 13].astype(float)
        area_2 = data[:, 11].astype(float)

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
        KrigMultiInfo1["optimizer"] = "lbfgsb"

        KrigMultiInfo2 = deepcopy(KrigMultiInfo1)
        KrigMultiInfo2['y'] = y[:, 1].reshape(-1, 1)

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

    def geomconst(self,vars):
        # This constraint function should return 1 if the constraint is satisfied and 0 if not.
        vars = np.array(vars)
        proj_area_1, area_1, proj_area_2, area_2 = constraints_check.calc_areas(vars[6], vars[4], vars[3], vars[7],
                                                                                vars[9],
                                                                                total_proj_area=0.00165529)
        s1_min = 0.3 * 0.00165529
        s1_max = 0.9 * 0.00165529
        s1_satisfied = constraints_check.min_max_satisfied(proj_area_1, min_val=s1_min, max_val=s1_max, disp=False)
        tip_angle = constraints_check.triangular_tip_angle(vars[8], vars[7], area_2)
        tip_satisfied = constraints_check.min_max_satisfied(tip_angle, 7, disp=False)
        stat = s1_satisfied & tip_satisfied
        return stat

    def evaluate(self, solution):
        vars = np.array(solution.variables)
        solution.objectives[:] = [self.krigobj1.predict(vars,'pred'), self.krigobj2.predict(vars,'pred')]
        solution.constraints[:] = [self.krigconst.predict(vars,'pred'), self.geomconst(vars)]


if __name__ == '__main__':
    prob1 = Hispeedplane()
    algorithm = NSGAII(prob1)
    algorithm.run(5000)

    nondominated_solutions = nondominated(algorithm.result)

    df = pd.read_csv('../innout/tim/opt_data_ASAT1.csv', sep=',', index_col='code')
    data = df.values
    X = data[:, 0:11].astype(float)
    y = data[:, 14:16].astype(float)
    cldat = data[:, 13].astype(float)

    nondom1 = np.array([s.objectives[0] for s in nondominated_solutions]).reshape(-1, 1)
    nondom2 = np.array([s.objectives[1] for s in nondominated_solutions]).reshape(-1, 1)
    var = [s.variables for s in nondominated_solutions]
    con = [s.constraint_violation for s in nondominated_solutions]
    var = np.array(var)
    nondom = np.hstack((nondom1, nondom2))
    predCL = prob1.krigconst.predict(var, 'pred')
    _, _, _, area_2pred = constraints_check.calc_areas(var[:, 6], var[:, 4], var[:, 3], var[:, 7],
                                                       var[:, 9],
                                                       total_proj_area=0.00165529)
    total = np.hstack((var, area_2pred.reshape(-1,1), predCL,nondom))
    np.savetxt("../innout/tim/Timnsga2next.csv", total, delimiter=",",
               header="x,z,le_sweep_1,dihedral_1,chord_1,tc_1,proj_span_1,chord_2,le_sweep_2,dihedral_2,tc_2,area_2,"
                      "CL,CD,dB(A)", comments="")

    plt.scatter(y[cldat > 0.15, 0], y[cldat > 0.15, 1], c='#1f77b4', label='initial feasible samples')
    plt.scatter(y[cldat <= 0.15, 0], y[cldat <= 0.15, 1], marker='x', c='k', label='initial infeasible samples')
    plt.scatter(nondom1, nondom2, c='#ff7f0e', label='predicted next samples')
    plt.xlabel("CD")
    plt.ylabel("dB(A)")
    plt.show()
