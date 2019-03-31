import numpy as np
import matplotlib.pyplot as plt
import globvar
from surrogate_models.kriging import ordinarykrig,kpls
from miscellaneous.surrogate_support.prediction import prediction
from miscellaneous.sampling.samplingplan import sampling,realval,standardize
from testcase.analyticalfcn.cases import evaluate,branin
from miscellaneous.surrogate_support.initinfo import initkriginfo
from optim_tools.GAv1 import uncGA
import time

#Initialization
KrigInfo = dict()
kernel = ["gaussian"]
# Sampling
nsample = 20
nvar = 2
ub = np.array([10,15])
lb = np.array([-5,0])
nup = 3
sampoption = "halton"
samplenorm,sample = sampling(sampoption,nvar,nsample,result="real",upbound=ub,lobound=lb)
X = sample
#Evaluate sample
y1 = evaluate(X,"branin")
y2 = evaluate(X,"styblinski")

#Initialize KrigInfo
KrigInfo = initkriginfo("multi",2)
#Set KrigInfo
KrigInfo["X"] = X
KrigInfo["y"][0] = y1
KrigInfo["y"][1] = y2
KrigInfo["nvar"] = nvar
KrigInfo["problem"] = "branin"
KrigInfo["nsamp"]= nsample
KrigInfo["nrestart"] = 5
KrigInfo["ub"]= ub
KrigInfo["lb"]= lb
KrigInfo["kernel"] = kernel
KrigInfo["nugget"] = -6

#Run Kriging
t = time.time()
myKrig = [0]*2
for kk in range(0,2):
    myKrig[kk] = ordinarykrig(KrigInfo,standardization=True,normtype="default",normalize_y=False,disp=True,num=kk)
elapsed = time.time() - t
print("elapsed time for train Kriging model: ", elapsed,"s")

#Test Kriging Output
neval = 10000
samplenormout,sampleeval = sampling(sampoption,nvar,neval,result="real",upbound=ub,lobound=lb)
Xeval = sampleeval

#Evaluate output
yeval = np.zeros(shape=[neval,1])
yact = np.zeros(shape=[neval,1])
yeval= prediction(Xeval,KrigInfo,"pred",num=1)
for ii in range(0,neval):
    yact[ii,0]= branin(Xeval[ii,:])

hasil = np.hstack((yeval,yact))

#Evaluate RMSE
subs = np.transpose((yact-yeval))
RMSE = np.sqrt(np.sum(subs**2)/neval)
print("RMSE = ",RMSE)
print(KrigInfo)
