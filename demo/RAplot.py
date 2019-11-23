import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../innout/out/radata/heatcond.csv')
kpls1 = data['KPLS1']
kpls2 = data['KPLS2']
kpls3 = data['KPLS3']
kpls4 = data['KPLS4']
ok = data['OK']
iter = data['iter'] + 1
realval = [0.041]*iter.size

plt.figure(0, figsize=[10,9])
plt.plot(iter,realval,'k',label='real',linewidth=2)
plt.plot(iter,kpls1,'g-.',label='KPLS1',linewidth=2)
plt.plot(iter,kpls2,'r--',label='KPLS2',linewidth=2)
plt.plot(iter,kpls3,color='magenta',label='KPLS3',linewidth=2)
plt.plot(iter,kpls4,'y-',label='KPLS4',linewidth=2)
plt.plot(iter,ok,'b:',label='OK',linewidth=2)
plt.xlabel('No. Update',fontsize=18)
plt.ylabel('Prob of Failure',fontsize=18)
plt.tick_params(axis='both', which='both', labelsize=14)
plt.legend(loc=1,prop={'size': 13})
plt.xlim([0,iter.size])
plt.ylim([0.02,0.06])
plt.grid()
plt.show()