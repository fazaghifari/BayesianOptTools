import numpy as np
import matplotlib.pyplot as plt
from surrogate_models.kriging import kriging,kpls
from miscellaneous.surrogate_support.prediction import prediction
from miscellaneous.sampling.samplingplan import sampling,realval,standardize
from testcase.analyticalfcn.cases import evaluate,branin
from miscellaneous.surrogate_support.initinfo import initkriginfo
from optim_tools.GAv1 import uncGA
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

#Initialize KrigInfo
KrigInfo = initkriginfo("single")
#Set KrigInfo
KrigInfo["X"] = X
KrigInfo["y"] = y1
KrigInfo["nvar"] = nvar
KrigInfo["problem"] = "branin"
KrigInfo["nsamp"]= nsample
KrigInfo["nrestart"] = 5
KrigInfo["ub"]= ub
KrigInfo["lb"]= lb
KrigInfo["kernel"] = kernel
KrigInfo["TrendOrder"] = 0
KrigInfo["nugget"] = -6

#Run Kriging
t = time.time()
myKrig = kpls(KrigInfo,standardization=True,normtype="default",normalize_y=False,disp=True)
elapsed = time.time() - t
print("elapsed time for train Kriging model: ", elapsed,"s")

#Test Kriging Output
neval = 2500
samplenormout,sampleeval = sampling(sampoption,nvar,neval,result="real",upbound=ub,lobound=lb)
xx = np.linspace(-5, 10, 50)
yy = np.linspace(0, 15, 50)
Xevalx, Xevaly= np.meshgrid(xx, yy)
Xeval = np.zeros(shape=[neval,2])
Xeval[:,0] = np.reshape(Xevalx,(neval))
Xeval[:,1] = np.reshape(Xevaly,(neval))

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

yeval1 = np.reshape(yeval,(50,50))
x1eval = np.reshape(Xeval[:,0],(50,50))
x2eval = np.reshape(Xeval[:,1],(50,50))
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x1eval, x2eval, yeval1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()
