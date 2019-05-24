from miscellaneous.surrogate_support.trendfunction import polytruncation,compute_regression_mat,legendre
import numpy as np
from testcase.analyticalfcn.cases import evaluate
from surrogate_models.kriging import kriging,kpls
from miscellaneous.surrogate_support.initinfo import initkriginfo
from miscellaneous.sampling.samplingplan import sampling,realval,standardize
from testcase.analyticalfcn.cases import evaluate,branin
from optim_tools.BOunc import sobounc
from matplotlib import pyplot as plt
import time

# EI update still encounter some error when using CMA-ES and
#Initialization
KrigInfo = dict()
BayesInfo = dict()
kernel = ["gaussian"]
# Sampling
nsample = 20
nvar = 2
ub = np.array([5,5])
lb = np.array([-5,-5])
nup = 35
sampoption = "rlh"
samplenorm,sample = sampling(sampoption,nvar,nsample,result="real",upbound=ub,lobound=lb)
X = sample
#Evaluate sample
y = evaluate(X,"styblinski")

#Set Kriging Info
KrigInfo = initkriginfo("single")
KrigInfo["X"] = X
KrigInfo["y"] = y
KrigInfo["nvar"] = nvar
KrigInfo["problem"] = "styblinski"
KrigInfo["nsamp"]= nsample
KrigInfo["nrestart"] = 5
KrigInfo["ub"]= ub
KrigInfo["lb"]= lb
KrigInfo["kernel"] = kernel
KrigInfo["nugget"] = -6

#Set Bayesian Optimization info
BayesInfo["nup"] = nup
BayesInfo["stalliteration"] = 40
BayesInfo["nrestart"] = 10
BayesInfo["acquifunc"] = "EI"
BayesInfo["acquifuncopt"] = "fmincon"

#Run Kriging
t = time.time()
MyKrig = kriging(KrigInfo,standardization=True,normtype="default",normalize_y=True,disp=True)
elapsed = time.time() - t
print("elapsed time for train Kriging model: ", elapsed,"s")

#Run Bayesian Opt
xbest,ybest,yhist,KrigNewInfo = sobounc(BayesInfo,MyKrig,krigtype=kriging,normalize_y=True)

print("The best feasible value is ",ybest)

plt.scatter(KrigNewInfo['X'][0:nsample,0],KrigNewInfo['X'][0:nsample,1])
plt.scatter(KrigNewInfo['X'][nsample:nsample+nup,0],KrigNewInfo['X'][nsample:nsample+nup,1])
plt.show()