import numpy as np
import matplotlib.pyplot as plt
import globvar
from surrogate_models.kriging import ordinarykrig,kpls
from miscellaneous.surrogate_support.prediction import prediction
from miscellaneous.sampling.samplingplan import sampling,realval,standardize
from testcase.analyticalfcn.twodtestcase import evaluate,branin
from miscellaneous.surrogate_support.EIpred import eipred
from optim_tools.GAv1 import uncGA
import time

# #Testcase Umich
# xt = np.array([[0., 1., 2., 3., 4.]])
# yt = np.array([[0., 1., 1.5, 0.5, 1.0]])
# xt = np.array([[0., 0.6, 1., 1.4, 2., 2.5, 3., 3.5, 4.]])
# yt = np.array([[0., 0.6, 1., 1.3, 1.5, 1.05, 0.5, 0.75, 1.0]])
# xt = np.transpose(xt)
# yt = np.transpose(yt)
# ndim = np.size(xt,1)
#
# NegLnLike, U, Psi = ordinarykrig(xt,yt,ndim)
# num = 100
# x = np.linspace(0., 4., num)
# y = np.zeros(shape=[num,1])
# for i in range(0,num):
#     y[i,0],_,_ = prediction(x[i])
#
# print(NegLnLike)
# plt.plot(xt, yt, 'o')
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(['Training data', 'Prediction'])
# plt.show()

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
y = evaluate(X,"branin")

#Set global variables
KrigInfo["X"] = X
KrigInfo["y"] = y
KrigInfo["nvar"] = nvar
KrigInfo["nsamp"]= nsample
KrigInfo["nrestart"] = 5
KrigInfo["ub"]= ub
KrigInfo["lb"]= lb
KrigInfo["kernel"] = kernel
KrigInfo["nugget"] = -6

#Run Kriging
t = time.time()
KrigInfo = ordinarykrig(KrigInfo,standardization=True,normtype="default",normalize_y=False)
elapsed = time.time() - t
print("elapsed time for train Kriging model: ", elapsed,"s")

#Perform kriging with EI
# globvar.Option = 'NegLogExpImp'
# for i in range(0,nup):
#     print("UPDATE NUMBER ", i+1)
#     print("Find Best EI position, Running . . .")
#     best_EI_pos,best_EI,_ = uncGA(eipred,lb,ub,"min")
#     print("Best EI Position is: ",best_EI_pos)
#     yEI,_,_ = branin(best_EI_pos)
#     X = np.vstack((X,best_EI_pos))
#     y = np.vstack((y,yEI))
#     #Re-Run Kriging
#     NegLnLike, U, Psi = ordinarykrig(X, y, nvar)


#Test Kriging Output
neval = 10000
samplenormout,sampleeval = sampling(sampoption,nvar,neval,result="real",upbound=ub,lobound=lb)
Xeval = sampleeval

#Evaluate output
yeval = np.zeros(shape=[neval,1])
yact = np.zeros(shape=[neval,1])
yeval= prediction(Xeval,KrigInfo,"pred")
for ii in range(0,neval):
    yact[ii,0]= branin(Xeval[ii,:])

hasil = np.hstack((yeval,yact))

#Evaluate RMSE
subs = np.transpose((yact-yeval))
RMSE = np.sqrt(np.sum(subs**2)/neval)
print("RMSE = ",RMSE)

