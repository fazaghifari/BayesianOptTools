import numpy as np
from numpy.linalg import solve as mldivide
from miscellaneous.surrogate_support.kernel import calckernel
from miscellaneous.sampling.samplingplan import standardize
from miscellaneous.surrogate_support.trendfunction import compute_regression_mat
from scipy.special import erf

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
    p = 2  # from reference
    realmin = np.finfo(float).tiny


    if x.ndim == 1:
        x =np.array([x])
    if KrigInfo["multiobj"] == True and "num" in KrigInfo and "wgkf" in KrigInfo and "idx" in KrigInfo:
        wgkf = KrigInfo["wgkf"][num]
        idx = KrigInfo["idx"][num]

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
        wgkf = KrigInfo["wgkf"]
        idx = KrigInfo["idx"]
        SigmaSqr = KrigInfo["SigmaSqr"]
        if KrigInfo["type"].lower() == "kpls":
            plscoeff = KrigInfo["plscoeff"]

    else:
        if KrigInfo["standardization"] == False:
            X = KrigInfo["X"]
            y = KrigInfo["y"][num]
        else:
            X = KrigInfo["X_norm"]
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

    if type(x) is float or type(x) is int:
        x = np.array([x])
    if "n_princomp" in KrigInfo:
        nvar = KrigInfo["n_princomp"]

    if KrigInfo["standardization"] == True:
        if KrigInfo["normtype"] == "default":
            if num == None:
                x = standardize(x, 0, type=KrigInfo["normtype"], range=np.vstack((KrigInfo["lb"],KrigInfo["ub"])))
            else:
                x = standardize(x, 0, type=KrigInfo["normtype"], range=np.vstack((KrigInfo["lb"], KrigInfo["ub"])))
        elif KrigInfo["normtype"] == "std":
            x = (x - KrigInfo["X_mean"]) / KrigInfo["X_std"]

    if type(predtype) is not list:
        npredtype = 1
        predtype = [predtype]
    else:
        npredtype = len(predtype)

    #Calculate number of sample points
    n = np.ma.size(X, axis=0)
    npred = np.size(x,axis=0)

    #Construct regression matrix for prediction
    bound = np.vstack((- np.ones(shape=[1, KrigInfo["nvar"]]), np.ones(shape=[1, KrigInfo["nvar"]])))
    PC = compute_regression_mat(idx,x,bound,np.ones(shape=[KrigInfo["nvar"]]))
    fpc = np.dot(PC,BE)

    #initialise psi to vector ones
    psi = np.ones((n,1),float)
    PsiComp = np.zeros(shape=[n, npred, nkernel])

    #fill psi vector
    if KrigInfo["type"].lower() == "kriging":
        for ii in range(0,nkernel):
            PsiComp[:,:,ii] = wgkf[ii]*calckernel(X,x,theta,nvar,type=kernel[ii])
        psi = np.sum(PsiComp,2)
        # for i in range (0,n):
        #     psi[i]= np.exp(-1*np.sum(theta*abs(X[i,:]-x)**p))

    elif KrigInfo["type"].lower() == "kpls":
        nvar = KrigInfo["nvar"]
        for ii in range(0, nkernel):
            PsiComp[:, :, ii] = wgkf[ii] * calckernel(X, x, theta, nvar, type=kernel[ii], plscoeff=plscoeff)
        psi = np.sum(PsiComp, 2)

        # for i in range(0, n):
        #     psi[i] = np.exp(-1 * np.sum(theta * np.dot(((X[i, :] - x) ** p), (plscoeff ** p))))

    #calculate prediction
    f = fpc + np.dot(np.transpose(psi), mldivide(U,mldivide(np.transpose(U),(y - np.dot(PHI,BE) ))))
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
    tempterm1 = np.transpose(np.array([term1]))
    newterm1 = np.matlib.repmat(tempterm1,1,np.size(term2,0))
    SSqr = np.dot(SigmaSqr,(term1+term2))
    s = (abs(SSqr))**0.5

    #Switch prediction type
    outputtotal = ()
    for predtype1 in predtype:
        if predtype1.lower() == "pred":
            output = f
        elif predtype1.lower() == "ssqr":
            output = SSqr
        elif predtype1.lower() == "fpc":
            output = fpc
        elif predtype1.lower() == "lcb":
            output = f - np.dot(KrigInfo["sigmalcb"],SSqr)
        elif predtype1.lower() == "ebe":
            output = -SSqr
        elif predtype1.lower() == "ei":
            yBest = np.min(y)
            if SSqr.all() == 0:
                ExpImp = 0
            else:
                EITermOne = (yBest - f) * (0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((yBest - f) / np.transpose(s) )))
                EITermTwo = np.transpose(s) * (1 / np.sqrt(2 * np.pi)) * np.exp(-(1 / 2) * (((yBest - f) ** 2) / np.transpose(SSqr) ))
                # give penalty for CMA-ES optimizer, if both term produce 0. Otherwise, in certain condition it may leads to error in CMA-ES
                if EITermOne.all() == 0 and EITermTwo.all() == 0:
                    ExpImp = np.array([[np.random.uniform(np.finfo("float").tiny, np.finfo("float").tiny * 100)]])
                else:
                    ExpImp = (EITermOne + EITermTwo + realmin)
            output = -ExpImp
        elif predtype1.lower() == "poi":
            ProbImp = 0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((np.min(y) - f) / np.transpose(s) ))
            output = -ProbImp
        elif predtype1.lower() == "pof":
            ProbFeas = 0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((KrigInfo["limit"] - (-f)) / np.transpose(s) ))
            output = ProbFeas
        outputtotal = outputtotal+(output,)

    if npredtype == 1:
        if np.size(output) == 1:
            return output[0,0]
        else:
            return output
    else:
        return outputtotal