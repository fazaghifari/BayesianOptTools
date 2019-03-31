from miscellaneous.surrogate_support.trendfunction import polytruncation,compute_regression_mat,legendre
import numpy as np
from testcase.analyticalfcn.cases import evaluate
from surrogate_models.kriging import ordinarykrig,kpls
from miscellaneous.surrogate_support.initinfo import initkriginfo
from miscellaneous.sampling.samplingplan import sampling,realval,standardize
from testcase.analyticalfcn.cases import evaluate,branin
from optim_tools.BOunc import sobounc
import time

# EI update still encounter some error when using CMA-ES and
#Initialization
KrigInfo = dict()
BayesInfo = dict()
kernel = ["gaussian","matern32"]
# Sampling
nsample = 10
nvar = 2
ub = np.array([10,15])
lb = np.array([-5,0])
nup = 20
sampoption = "rlh"
samplenorm,sample = sampling(sampoption,nvar,nsample,result="real",upbound=ub,lobound=lb)
X = sample
#Evaluate sample
y = evaluate(X,"branin")

#Set Kriging Info
KrigInfo = initkriginfo("single")
KrigInfo["X"] = X
KrigInfo["y"] = y
KrigInfo["nvar"] = nvar
KrigInfo["problem"] = "branin"
KrigInfo["nsamp"]= nsample
KrigInfo["nrestart"] = 5
KrigInfo["ub"]= ub
KrigInfo["lb"]= lb
KrigInfo["kernel"] = kernel
KrigInfo["nugget"] = -6

#Set Bayesian Optimization info
BayesInfo["nup"] = nup
BayesInfo["stalliteration"] = 20
BayesInfo["nrestart"] = 10
BayesInfo["acquifunc"] = "EI"
BayesInfo["acquifuncopt"] = "cmaes"

#Run Kriging
t = time.time()
MyKrig = ordinarykrig(KrigInfo,standardization=True,normtype="default",normalize_y=False,disp=True)
elapsed = time.time() - t
print("elapsed time for train Kriging model: ", elapsed,"s")

#Run Bayesian Opt
xbest,ybest,yhist,KrigNewInfo = sobounc(BayesInfo,MyKrig)

print("The best feasible value is ",ybest)