from miscellaneous.surrogate_support.trendfunction import polytruncation,compute_regression_mat,legendre
import numpy as np
from testcase.analyticalfcn.cases import evaluate
from surrogate_models.kriging import kriging,kpls
from miscellaneous.surrogate_support.initinfo import initkriginfo
from miscellaneous.sampling.samplingplan import sampling,realval,standardize
from testcase.analyticalfcn.cases import schaffer,evaluate
from optim_tools.BOunc import mobounc
import time
from matplotlib import pyplot as plt

nvar = 4
nsample = 20
nobj = 2
KrigMultiInfo = dict()
BayesMultiInfo = dict()
kernel = ["gaussian","matern32"]

# Construct Kriging for multiple objective functions
lb = -1 * np.ones(shape=[nvar])
ub =  np.ones(shape=[nvar])
sampoption = "rlh"
samplenorm,sample = sampling(sampoption,nvar,nsample,result="real",upbound=ub,lobound=lb)
X = sample
y = schaffer(X)

#Set Kriging Info
KrigMultiInfo = initkriginfo("multi",objective=2)
KrigMultiInfo["X"] = X
KrigMultiInfo["y"][0] = np.transpose([y[:,0]])
KrigMultiInfo["y"][1] = np.transpose([y[:,1]])
KrigMultiInfo["nvar"] = nvar
KrigMultiInfo["problem"] = "schaffer"
KrigMultiInfo["nsamp"]= nsample
KrigMultiInfo["nrestart"] = 7
KrigMultiInfo["ub"]= ub
KrigMultiInfo["lb"]= lb
KrigMultiInfo["kernel"] = kernel
KrigMultiInfo["nugget"] = -6
KrigMultiInfo["nkernel"] = len(KrigMultiInfo["kernel"])

#Set Bayesian Optimization info
BayesMultiInfo["nup"] = 10
BayesMultiInfo["nrestart"] = 10
BayesMultiInfo["acquifunc"] = "ehvi"
BayesMultiInfo["acquifuncopt"] = "fmincon"

#Create Kriging
myKrig = [0]*2
for kk in range(0,2):
    myKrig[kk] = kriging(KrigMultiInfo,standardization=True,normtype="default",normalize_y=False,disp=True,num=kk)

#Run Bayesian Optimization
xbest, ybest, KrigNewMultiInfo = mobounc(BayesMultiInfo,KrigMultiInfo)

plt.scatter(y[:,0],y[:,1])
plt.scatter(ybest[:,0],ybest[:,1])
plt.show()