import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surrogate_models.supports.errperf import errperf
from copy import deepcopy

nvar = 10
firstorder = [0,0,0,0,0]
totalorder = [0,0,0,0,0]
firsterror = [0,0,0,0,0]
totalerror = [0,0,0,0,0]
real_SA = pd.read_csv("../innout/out/bridge/acctest_bridge_real_SA.csv")
real_first, real_total = real_SA.iloc[:, :nvar], real_SA.iloc[:, nvar:]
for i in range(5):
    if i == 4:
        name = "../innout/out/bridge/acctest_bridge_60samp_OK_SA.csv"
    else:
        name = "../innout/out/bridge/acctest_bridge_60samp_KPLS" + str(i + 1) + "_SA.csv"
    temporary = pd.read_csv(name)
    firstorder[i], totalorder[i] = temporary.iloc[:, :nvar], temporary.iloc[:, nvar:2*nvar]
    temperrfirst = np.zeros(50)
    temperrtot = deepcopy(temperrfirst)
    for ii in range(50):
        temperrfirst[ii] = np.sum(errperf(np.array(real_first.iloc[0,:]),np.array(firstorder[i].iloc[ii,:]),'ae'))
        temperrtot[ii] = np.sum(errperf(np.array(real_total.iloc[0,:]),np.array(totalorder[i].iloc[ii,:]),'ae'))
    firsterror[i] = temperrfirst[:]
    totalerror[i] = temperrtot[:]

meanpointprops = dict(marker='D', markeredgecolor='g',
                      markerfacecolor='g')
red_square = dict(markeredgecolor='r',markerfacecolor='r', marker='X')
label_list = ['KPLS1','KPLS2','KPLS3','KPLS4','OK']
plt.figure(1, figsize=[10,9])
plt.boxplot(firsterror, vert=False, flierprops=red_square, showmeans=True, meanprops=meanpointprops, labels=label_list)
plt.xlabel('First Indices Error',fontsize=20)
plt.xscale('log')
plt.xlim([1e-3,100])
# plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
plt.tick_params(axis='both', which='both', labelsize=20)
plt.grid(which='both',axis='both',linestyle='--')

plt.figure(2, figsize=[10,9])
plt.boxplot(totalerror, vert=False, flierprops=red_square, showmeans=True, meanprops=meanpointprops, labels=label_list)
plt.xlabel('Total Indices Error',fontsize=20)
plt.xscale('log')
plt.xlim([1e-3,100])
# plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
plt.tick_params(axis='both', which='both', labelsize=20)
plt.grid(which='both',axis='both',linestyle='--')

plt.show()