import numpy as np

def initkriginfo(type,objective=1):
    """
    Initialize the value of KrigInfo, the value will be set to default value.
    You can change the value of KrigInfo outside this function

    Inputs:
        type - "single" or "multi"
        objective (optional) - number of objective function (default is 1)

    Output:
        KrigInfo - A structure containing information of the constructed Kriging of the objective function.

    Default values:
        KrigInfo["X"] = [0]
        KrigInfo["y"] = [0]
        KrigInfo["nvar"] = [0]
        KrigInfo["problem"] = "branin"
        KrigInfo["nsamp"]= [0]
        KrigInfo["nrestart"] = 5
        KrigInfo["ub"]= [0]
        KrigInfo["lb"]= [0]
        KrigInfo["kernel"] = "gaussian"
        KrigInfo["nugget"] = -6
    """
    if type.lower() != "single" and type.lower() != "multi":
        raise TypeError ("Input must be 'single' or 'multi'.")
    else:
        KrigInfo = dict()
        if type.lower() == "single":
            KrigInfo["X"] = [0]
            KrigInfo["y"] = [0]
            KrigInfo["nvar"] = [0]
            KrigInfo["problem"] = "branin"
            KrigInfo["nsamp"] = [0]
            KrigInfo["nrestart"] = 5
            KrigInfo["ub"] = [0]
            KrigInfo["lb"] = [0]
            KrigInfo["kernel"] = "gaussian"
            KrigInfo["nugget"] = -6
            keys = ["Theta", "U", "Psi", "BE", "y_mean", "y_std", "SigmaSqr", "idx", "F", "wgkf", "plscoeff"]
            for key in keys:
                KrigInfo[key] = [0]

        elif type.lower() == "multi":
            KrigInfo["X"] = [0]
            KrigInfo["nvar"] = [0]
            KrigInfo["problem"] = "branin"
            KrigInfo["nsamp"] = [0]
            KrigInfo["nrestart"] = 5
            KrigInfo["krignum"] = 1
            KrigInfo["kernel"] = "gaussian"
            KrigInfo["nugget"] = -6
            KrigInfo["multiobj"] = True
            KrigInfo["standardization"] = True
            keys = ["y","Theta", "U", "Psi", "BE", "y_mean", "y_std", "SigmaSqr", "idx", "F", "wgkf","lb","ub", "plscoeff"]
            for key in keys:
                KrigInfo[key] = [0]*objective

    return KrigInfo

def copymultiKrigInfo(KrigMultiInfo,num):
    """
    Function for copying multi-objective KrigInfo into single KrigInfo

    Inputs:
        KrigMultiInfo - Multi-objective KrigInfo
        num - Index of objective

    Output:
        KrigNewInfo - A structure containing information of the constructed Kriging of the objective function
                      taken from KrigMultiInfo.
    """

    KrigNewInfo = dict()
    KrigNewInfo["X"] = KrigMultiInfo["X"]
    KrigNewInfo["y"] = KrigMultiInfo["y"]
    KrigNewInfo["nvar"] = KrigMultiInfo["nvar"]
    KrigNewInfo["problem"] = KrigMultiInfo["problem"]
    KrigNewInfo["nsamp"] = KrigMultiInfo["nsamp"]
    KrigNewInfo["nrestart"] = KrigMultiInfo["nrestart"]
    KrigNewInfo["ub"] = KrigMultiInfo["ub"]
    KrigNewInfo["lb"] = KrigMultiInfo["lb"]
    KrigNewInfo["kernel"] = KrigMultiInfo["kernel"]
    KrigNewInfo["nugget"] = KrigMultiInfo["nugget"]
    keys = ["Theta", "U", "Psi", "BE", "y_mean", "y_std", "SigmaSqr", "idx", "F", "wgkf", "plscoeff"]
    for key in keys:
        KrigNewInfo[key] = KrigMultiInfo[key][num]

    return KrigNewInfo