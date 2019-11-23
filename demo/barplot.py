import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surrogate_models.supports.errperf import errperf
from copy import deepcopy

nvar = 40
firstorder = [0,0,0,0,0]
totalorder = [0,0,0,0,0]
firsterror = [0,0,0,0,0]
totalerror = [0,0,0,0,0]
real_SA = pd.read_csv("../innout/out/40/acctest_analytic40_real_SA.csv")
real_first, real_total = real_SA.iloc[:, :nvar], real_SA.iloc[:, nvar:]
for i in range(5):
    if i == 4:
        name = "../innout/out/40/acctest_analytic40_OK_SA.csv"
    else:
        name = "../innout/out/40/acctest_analytic40_KPLS" + str(i + 1) + "_SA.csv"
    temporary = pd.read_csv(name)
    firstorder[i], totalorder[i] = temporary.iloc[:, :nvar], temporary.iloc[:, nvar:2*nvar]
    temperrfirst = np.zeros(50)
    temperrtot = deepcopy(temperrfirst)
    labels = list(range(1,41))
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    plt.figure(i+1)
    plt.bar(x-width/2, firstorder[i].mean(0), width, label='Predict')
    plt.bar(x + width / 2, real_first.iloc[0,:], width, label='Real')
    plt.xticks(x)
    plt.legend()

plt.show()