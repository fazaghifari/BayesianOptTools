import numpy as np
import RA.akmcs as AKMCS
from matplotlib import pyplot as plt
from testcase.RA.testcase import evaluate
from miscellaneous.surrogate_support.initinfo import initkriginfo
import time

nvar = 2
n_krigsamp = 20
problem = "styblinski"

# Monte Carlo Sampling
init_samp = AKMCS.mcpopgen(type="gaussian",sigma=1.5,ndim=nvar)
init_krigsamp = 1.5*np.random.randn(n_krigsamp,2)
ykrig = evaluate(init_krigsamp, type= problem)

init_samp_G = evaluate(init_samp, type= problem)
total_samp = np.hstack((init_samp,init_samp_G)).transpose()
positive_samp = total_samp[:,total_samp[2]>= 0]
positive_samp = positive_samp.transpose()
nsamp = np.size(init_samp,0)
npos = np.size(positive_samp,0)
Pfreal = 1 - npos/nsamp # 0.002247 for fourbranches

lb = np.floor(np.min(init_samp))*np.ones(shape=[nvar])
ub = np.ceil(np.max(init_samp))*np.ones(shape=[nvar])

# Set Kriging Info
KrigInfo = initkriginfo("single")
KrigInfo["X"] = init_krigsamp
KrigInfo["y"] = ykrig
KrigInfo["nvar"] = nvar
KrigInfo["nsamp"]= n_krigsamp
KrigInfo["nrestart"] = 5
KrigInfo["ub"]= ub
KrigInfo["lb"]= lb
KrigInfo["kernel"] = ["gaussian"]
KrigInfo["nugget"] = -6
KrigInfo["nkernel"] = len(KrigInfo["kernel"])
KrigInfo["LOOCVerror"] = 0
KrigInfo["LOOCVpred"] = 0

# Run AK-MCS
akmcsInfo = dict()
akmcsInfo["init_samp"] = init_samp
akmcsInfo["maxupdate"] = 70
akmcsInfo["problem"] = problem
t = time.time()
LFUval, updatedX = AKMCS.akmcs(KrigInfo,akmcsInfo)
print(f"real Pf : {Pfreal}")
elapsed = time.time() - t
print("elapsed time is : ", elapsed, "s")
print(LFUval)

plt.scatter(init_samp[:,0],init_samp[:,1],s=2,c='c',label='Monte Carlo Sampling')
plt.scatter(positive_samp[:,0],positive_samp[:,1],s=2,c='y',label='G(x)>0')
plt.scatter(init_krigsamp[:,0],init_krigsamp[:,1],s=15, c='r',label='Initial Samples')
plt.scatter(updatedX[:,0],updatedX[:,1],marker='x',c='k',s=15,label='Update')
plt.ylabel('x2')
plt.xlabel('x1')
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.legend()
plt.show()