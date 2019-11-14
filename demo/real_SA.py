import sys
sys.path.insert(0, "..")
import numpy as np
from reliability_analysis.akmcs import AKMCS,mcpopgen
from testcase.RA.testcase import evaluate
from surrogate_models.supports.initinfo import initkriginfo
from sensitivity_analysis.sobol_ind import SobolIndices as SobolI
import matplotlib.pyplot as plt
import time

def sensitivity(krigobj,init_samp,nvar,second=False):
    lb = (np.min(init_samp, axis=0))
    ub = (np.max(init_samp, axis=0))
    lb = np.hstack((lb,lb))
    ub = np.hstack((ub,ub))
    testSA = SobolI(nvar, krigobj, 'hidimenra', ub, lb)
    result = testSA.analyze(True, True, second)
    for key in result.keys():
        print(key+':')
        if type(result[key]) is not dict:
            print(result[key])
        else:
            for subkey in result[key].keys():
                print(subkey+':', result[key][subkey])

    return result

if __name__ == '__main__':
    nvar = 100
    init_samp = np.loadtxt('../innout/in/lognormal100.csv', delimiter=',')
    result = sensitivity(None, init_samp, nvar)

    i=0
    mylist = []
    for ii in range(100):
        mylist.append("S" + str(ii + 1) + ", ")
    for ii in range(99):
        mylist.append("St" + str(ii + 1) + ", ")
    mylist.append("St" + str(100))
    SAhead = ""
    for header in mylist:
        SAhead += header
    saresult = np.array([np.hstack((result['first'], result['total']))])
    sadata = saresult[:]
    np.savetxt('../innout/out/100/acctest_analytic100_REAL_SA.csv', sadata, fmt='%10.5f', delimiter=',',
               header=SAhead)