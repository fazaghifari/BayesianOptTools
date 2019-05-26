import numpy as np
from miscellaneous.surrogate_support.prediction import prediction
from optim_tools.ehvi.exi2d import exi2d
from testcase.analyticalfcn.cases import evaluate

def ehvicalc(x,ypar,BayesMultiInfo,KrigNewMultiInfo):
    """
    ModelInfoKR{i} = Model Information of objective i
    ObjectiveInfoKR{i} = Objective Information of objective i

    Input :
        - x : Design variables
        - ypar: Current Pareto front
        - BayesMultiInfo: Structure(Dictionary) containing necessary information for multiobjective Bayesian optimization.
        - KrigMultiInfo: Structure(Dictionary) containing necessary information for multiobjective Kriging.
    """

    X = KrigNewMultiInfo["X"]
    nobj = len(KrigNewMultiInfo["y"])
    nsamp = np.size(X,0)
    YO = np.zeros(shape=[nsamp,nobj])
    RefP = BayesMultiInfo["refpoint"]

    # prediction of each objective
    pred = np.zeros(shape=[nobj])
    SSqr = np.zeros(shape=[nobj])
    for ii in range(0,nobj):
        pred[ii],SSqr[ii] = prediction(x, KrigNewMultiInfo, ["pred","SSqr"], num=ii)

    # Compute (negative of) hypervolume
    HV = -1 * exi2d(ypar,RefP,pred,SSqr)

    if HV == 0: # give penalty to HV, to avoid error in CMA-ES when in an iteration produce all HV = 0
        HV = np.random.uniform(np.finfo("float").tiny, np.finfo("float").tiny * 100)

    return HV
