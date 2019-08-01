import numpy as np
from surrogate_models.kriging import kriging,kpls
from miscellaneous.surrogate_support.initinfo import initkriginfo
from optim_tools.BOunc import mobounc
import time
from matplotlib import pyplot as plt
import pandas as pd

#import data
df=pd.read_csv('PotsawatData.csv', sep=',',index_col='Model')
data = df.values
X = data[:,0:3]
y = data[:,3:]

#define variables
nvar = np.size(X,1)
nsample = np.size(X,0)
nobj = np.size(y,1)
KrigMultiInfo = dict()
BayesMultiInfo = dict()
ConstInfo = dict()
kernel = ["gaussian"]
lb = np.array([0.1,3,3.1])
ub = np.array([0.9,11.41,60.95])
tabulatedconst = np.loadtxt("PotsawatConst.csv",delimiter=",")

#Set Kriging Info
KrigMultiInfo = initkriginfo("multi",objective=2)
KrigMultiInfo["X"] = X
KrigMultiInfo["y"][0] = y[:,0].reshape(-1,1)
KrigMultiInfo["y"][1] = y[:,1].reshape(-1,1)
KrigMultiInfo["nvar"] = nvar
KrigMultiInfo["nsamp"]= nsample
KrigMultiInfo["nrestart"] = 7
KrigMultiInfo["ub"]= ub
KrigMultiInfo["lb"]= lb
KrigMultiInfo["kernel"] = kernel
KrigMultiInfo["nugget"] = -6
KrigMultiInfo["nkernel"] = len(KrigMultiInfo["kernel"])
KrigMultiInfo["LOOCVerror"] = [0] * nobj
KrigMultiInfo["LOOCVpred"] = [0] * nobj

#Set Bayesian Optimization info
BayesMultiInfo["nup"] = 1
BayesMultiInfo["nrestart"] = 10
BayesMultiInfo["acquifunc"] = "ehvi"
BayesMultiInfo["acquifuncopt"] = "fmincon"
ConstInfo["constraint"] = tabulatedconst
ConstInfo["consttype"] = "tabulated"

#Create Kriging
myKrig = [0]*2
for kk in range(0,2):
    myKrig[kk] = kriging(KrigMultiInfo,standardization=True,normtype="default",disp=True,num=kk,loocvcalc=True, normalize_y=True)
    print("LOOCV Error Kriging ",kk,": ",KrigMultiInfo["LOOCVerror"][kk]," % (MAPE)")

#Run Bayesian Optimization
xnext, ynext, KrigNewMultiInfo = mobounc(BayesMultiInfo,KrigMultiInfo, auto=False, multiupdate=3, normalize_y=True, ConstraintInfo=ConstInfo)
print("Suggested next sample: ",xnext,", F-count: ",np.size(KrigMultiInfo["X"],0))
np.savetxt("nextsamp.csv", xnext, delimiter=",")
np.savetxt("predictedynext.csv", ynext, delimiter=",")

plt.scatter(y[:,0],y[:,1])
plt.scatter(ynext[:,0],ynext[:,1])
plt.ylabel('dB (C)')
plt.xlabel('CD')
plt.show()