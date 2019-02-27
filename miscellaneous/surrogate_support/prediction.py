import numpy as np
from numpy.linalg import solve as mldivide
from miscellaneous.surrogate_support.kernel import calckernel
from miscellaneous.sampling.samplingplan import standardize
from miscellaneous.surrogate_support.trendfunction import compute_regression_mat
from math import erf

def prediction (x,KrigInfo,predtype,**kwargs):
    """
    Calculates expected improvement (for optimization), SSqr, Kriging prediction, and prediction from regression function

    Information: This function is a modification from "Forrester, A., Sobester, A., & Keane, A. (2008). Engineering design via surrogate modelling:  a practical guide. John Wiley & Sons."

    Inputs:
      XP - Prediction site (will be normalized to [-1,1])
      KrigInfo - A structure containing necessary information of a constructed Kriging model
      predictiontype - The output as defined by the user
          'pred' - for Kriging prediction.
          'SSqr' - for Kriging prediction error.
          'trend' - for computed trend function.
          'EI' - for expected improvement.

    Information used in KrigInfo for krigprediction
      KrigInfo.Xnorm - (nsamp x nvar) matrix of normalized experimental design.
      KrigInfo.Y - (nsamp x 1) vector of responses.
      KrigInfo.PHI - (nsamp x nind) matrix of regression function.
      KrigInfo.idx - (nind x nvar) matrix consisting of polynomial index for regression function.
      KrigInfo.kernel - Type of kernel function.
      KrigInfo.wgkf - (1 x nkrnl) vector of weights for kernel functions.
      KrigInfo.U - Choleski factorisation of correlation matrix.
      KrigInfo.xparam - Hyperparameters of the Kriging model.
      KrigInfo.BE - Coefficients of regression function
      KrigInfo.SigmaSqr - SigmaSqr (Kriging variance) of the Kriging model

    Outputs:
      output - The output as defined by the user (see 'predictiontype').
      SSqr - Kriging prediction error at the prediction site.
      y_hat - Kriging prediction.
      fpc - Kriging trend function.

    Information:
     - Note that the output can be vectorized.

    Author: Pramudita Satria Palar(pramsatriapalar@gmail.com, pramsp@ftmd.itb.ac.id)
    """
    num = kwargs.get('num',None) # Num means Objective Function number XX
    nvar = KrigInfo["nvar"]
    kernel = KrigInfo["kernel"]
    nkernel = KrigInfo["nkernel"]
    wgkf = KrigInfo["wgkf"]
    idx = KrigInfo["idx"]
    p = 2  # from reference

    if KrigInfo["multiobj"] == True:
        num = KrigInfo["num"]

    if num == None:
        if KrigInfo["standardization"] == False:
            X = KrigInfo["X"]
            y = KrigInfo["y"]
        else:
            X = KrigInfo["X_norm"]
            if "y_norm" in KrigInfo:
                y = KrigInfo["y_norm"]
            else:
                y = KrigInfo["y"]
        theta = 10 ** KrigInfo["Theta"]
        U = KrigInfo["U"]
        PHI = KrigInfo["F"]
        BE = KrigInfo["BE"]
        SigmaSqr = KrigInfo["SigmaSqr"]
        if KrigInfo["type"].lower() == "kpls":
            plscoeff = KrigInfo["plscoeff"]
    else:
        if KrigInfo["standardization"] == False:
            X = KrigInfo["X"][num]
            y = KrigInfo["y"][num]
        else:
            X = KrigInfo["X_norm"][num]
            if "y_norm" in KrigInfo:
                y = KrigInfo["y_norm"][num]
            else:
                y = KrigInfo["y"][num]
        theta = 10**KrigInfo["Theta"][num]
        U = KrigInfo["U"][num]
        PHI = KrigInfo["F"][num]
        BE = KrigInfo["BE"][num]
        SigmaSqr = KrigInfo["SigmaSqr"][num]
        if KrigInfo["type"].lower() == "kpls":
            plscoeff = KrigInfo["plscoeff"][num]

    if KrigInfo["standardization"] == True:
        if KrigInfo["normtype"] == "default":
            x = standardize(x, 0, type=KrigInfo["normtype"], range=np.vstack((KrigInfo["lb"],KrigInfo["ub"])))
        elif KrigInfo["normtype"] == "std":
            x = (x - KrigInfo["X_mean"]) / KrigInfo["X_std"]


    #Calculate number of sample points
    n = np.ma.size(X, axis=0)
    npred = np.size(x,axis=0)

    #Construct regression matrix for prediction
    bound = np.vstack((- np.ones(shape=[1, KrigInfo["nvar"]]), np.ones(shape=[1, KrigInfo["nvar"]])))
    PC = compute_regression_mat(idx,x,bound,np.ones(shape=[KrigInfo["nvar"]]))
    fpc = np.dot(PC,BE)

    #initialise psi to vector ones
    psi = np.ones((n,1),float)
    PsiComp = np.zeros(shape=[n, npred, nvar])

    #fill psi vector
    if KrigInfo["type"].lower() == "kriging":
        for ii in range(0,nkernel):
            PsiComp[:,:,ii] = wgkf[ii]*calckernel(X,x,theta,nvar,type=kernel[ii])
        psi = np.sum(PsiComp,2)
        # for i in range (0,n):
        #     psi[i]= np.exp(-1*np.sum(theta*abs(X[i,:]-x)**p))

    elif KrigInfo["type"].lower() == "kpls":
        for i in range(0, n):
            psi[i] = np.exp(-1 * np.sum(theta * np.dot(((X[i, :] - x) ** p), (plscoeff ** p))))

    #calculate prediction
    f = fpc + np.dot(np.transpose(psi), mldivide(U,mldivide(np.transpose(U),(y - PHI*BE))))
    if num == None:
        if KrigInfo["norm_y"] == True:
            f = (KrigInfo["y_mean"] + KrigInfo["y_std"]*f)
    else:
        if KrigInfo["norm_y"] == True:
            f = (KrigInfo["y_mean"][num] + KrigInfo["y_std"][num]*f)

    #compute sigma-squared error
    dummy1 = mldivide(U,mldivide(np.transpose(U),psi))
    dummy2 = mldivide(U, mldivide(np.transpose(U), PHI))
    term1 = (1 - np.sum(np.transpose(psi)*np.transpose(dummy1),1))
    ux = (np.dot(np.transpose(PHI),dummy1))-np.transpose(PC)
    term2 = ux*(mldivide(np.dot(np.transpose(PHI),dummy2),ux))
    SSqr = np.dot(SigmaSqr,(term1+term2))
    s = (abs(SSqr))**0.5

    #Switch prediction type
    if predtype.lower() == "pred":
        output = f
    elif predtype.lower() == "ssqr":
        output = SSqr
    elif predtype.lower() == "fpc":
        output = fpc
    elif predtype.lower() == "lcb":
        output = f - np.dot(KrigInfo["sigmalcb"],SSqr)
    elif predtype.lower() == "ebe":
        output = -SSqr
    elif predtype.lower() == "ei":
        yBest = np.min(y)
        if SSqr == 0:
            ExpImp = 0
        else:
            EITermOne = (yBest - f) * (0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((yBest - f) / s)))
            EITermTwo = s * (1 / np.sqrt(2 * np.pi)) * np.exp(-(1 / 2) * (((yBest - f) ** 2) / SSqr))
            ExpImp = np.log10(EITermOne + EITermTwo + 1e-10)
        output = -ExpImp
    elif predtype.lower() == "poi":
        ProbImp = 0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((np.min(y) - f) / s))
        output = -ProbImp
    elif predtype.lower() == "pof":
        ProbFeas = 0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((KrigInfo["limit"] - (-f)) / s))
        output = ProbFeas

    return output