import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from testcase.RA.testcase import evaluate
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

init_samp = np.loadtxt('../innout/in/bridge3.csv', delimiter=',')
init_samp_G = np.loadtxt('../innout/out/bridge3_gx.csv', delimiter=',')
iqr = np.subtract(*np.percentile(init_samp_G, [75, 25]))
kplslist = [0,0,0,0,0]
rrmselist = [0,0,0,0,0]
mapelist = [0,0,0,0,0]
meanlist = [0,0,0,0,0]
stddevlist = [0,0,0,0,0]
meanreal = np.mean(init_samp_G)
stddevreal = np.std(init_samp_G)

for i in range(5):
    if i == 4:
        name = "../innout/out/bridge/acctest_bridge_OK.csv"
    else:
        name = "../innout/out/bridge/acctest_bridge_KPLS"+str(i+1)+".csv"
    kplslist[i] = pd.read_csv(name)
    meanlist[i] = kplslist[i].Mean
    stddevlist[i] = kplslist[i]['Std Dev']
    rrmselist[i] = kplslist[i].RMSE / iqr
    mapelist[i] = kplslist[i].MAPE / 100

meanpointprops = dict(marker='D', markeredgecolor='g',
                      markerfacecolor='g')
red_square = dict(markeredgecolor='r',markerfacecolor='r', marker='X')
label_list = ['KPLS1','KPLS2','KPLS3','KPLS4','OK']
plt.figure(1, figsize=[10,9])
plt.boxplot(rrmselist, vert=False, flierprops=red_square, showmeans=True, meanprops=meanpointprops, labels=label_list)
plt.xlabel('Normalized RMSE',fontsize=20)
plt.xscale('log')
plt.xlim([1e-2,1])
# plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
plt.tick_params(axis='both', which='both', labelsize=16)
plt.grid(which='both',axis='both',linestyle='--')

plt.figure(2, figsize=[10,9])
plt.boxplot(mapelist, vert=False, flierprops=red_square, showmeans=True, meanprops=meanpointprops, labels=label_list)
plt.xlabel('Mean Absolute Relative Error',fontsize=20)
plt.xscale('log')
plt.xlim([1e-2,1])
# plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
plt.tick_params(axis='both', which='both', labelsize=16)
plt.grid(which='both',axis='both',linestyle='--')

plt.figure(3, figsize=[10,9])
plt.boxplot(meanlist, vert=False, flierprops=red_square, showmeans=True, meanprops=meanpointprops, labels=label_list)
plt.axvline(x=meanreal)
plt.xlabel('$\mu f(x)$',fontsize=20)
plt.xscale('log')
plt.xlim([2e-2,4e-2])
# plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
plt.tick_params(axis='both', which='both', labelsize=16)
plt.grid(which='both',axis='both',linestyle='--')

plt.figure(4, figsize=[10,9])
plt.boxplot(stddevlist, vert=False, flierprops=red_square, showmeans=True, meanprops=meanpointprops, labels=label_list)
plt.axvline(x=stddevreal)
plt.xlabel('$\sigma f(x)$',fontsize=20)
plt.xscale('log')
plt.xlim([6e-3,1e-2])
# plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
plt.tick_params(axis='both', which='both', labelsize=16)
plt.grid(which='both',axis='both',linestyle='--')

plt.show()
