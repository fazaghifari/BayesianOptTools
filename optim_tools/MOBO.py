import numpy as np
from copy import deepcopy
import math
import scipy.io as sio
from surrogate_models.kriging_model import Kriging


class MOBOunc():
    """
    Perform unconstrained multi-objective Bayesian Optimization

    Args:
        moboInfo (dict): Dictionary containing necessary information for multi-objective Bayesian optimization.
        kriglist (list): List of Kriging object.
        autoupdate (bool): True or False, depends on your decision to evaluate your function automatically or not.
        multiupdate (int): Number of suggested samples returned for each iteration.

    Returns:
        Xbest (nparray): Matrix of final non-dominated solutions observed after optimization.
        Ybest (nparray): Matrix of responses of final non-dominated solutions after optimization.
        Metricbest (nparray): Matrix of metric value of final non-dominated solutions after optimization.
    """

    def __init__(self, moboInfo, kriglist, autoupdate=True, multiupdate=0):
        """
        Initialize MOBOunc class

        Args:
            moboInfo (dict): Dictionary containing necessary information for multi-objective Bayesian optimization.
            kriglist (list): List of Kriging object.
            autoupdate (bool): True or False, depends on your decision to evaluate your function automatically or not.
            multiupdate (int): Number of suggested samples returned for each iteration.
        """
        self.moboInfo = moboinfocheck(moboInfo)
        self.kriglist = kriglist
        self.krignum = len(self.kriglist)


def moboinfocheck(moboInfo, autoupdate, krignum):
    """
    Function to check the MOBO information and set MOBO Information to default value if
    required parameters are not supplied.

    Args:
         moboInfo (dict): Structure containing necessary information for multi-objective Bayesian optimization.
         autoupdate (bool): True or False, depends on your decision to evaluate your function automatically or not.

     Returns:
         moboInfo (dict): Checked/Modified MOBO Information
    """
    # Check necessary parameters
    if "nup" not in moboInfo:
        if autoupdate is True:
            raise ValueError("Number of updates for Bayesian optimization, moboInfo['nup'], is not specified")
        else:
            moboInfo["nup"] = 1
            print("Number of updates for Bayesian optimization has been set to 1")
    else:
        if autoupdate == True:
            pass
        else:
            moboInfo["nup"] = 1
            print("Manual mode is active, number of updates for Bayesian optimization is forced to 1")

    # Set default values
    if "acquifunc" not in moboInfo:
        moboInfo["acquifunc"] = "EHVI"
        print("The acquisition function is not specified, set to EHVI")
    else:
        availacqfun = ["ehvi", "parego"]
        if moboInfo["acquifunc"].lower() not in availacqfun:
            raise TypeError(moboInfo["acquifunc"], " is not a valid acquisition function.")
        else:
            print("The acquisition function is specified to ", moboInfo["acquifunc"], " by user")

    # Set necessary params for multiobjective acquisition function
    if moboInfo["acquifunc"].lower() == "ehvi":
        moboInfo["krignum"] = krignum
        if "refpoint" not in moboInfo:
            moboInfo["refpointtype"] = 'dynamic'
    elif moboInfo["acquifunc"].lower() == "parego":
        moboInfo["krignum"] = 1
        if "paregoacquifunc" not in moboInfo:
            moboInfo["paregoacquifunc"] = "EI"

    # If moboInfo.acquifuncopt (optimizer for the acquisition function) is not specified set to 'sampling+cmaes'
    if "acquifuncopt" not in moboInfo:
        moboInfo["acquifuncopt"] = "sampling+cmaes"
        print("The acquisition function optimizer is not specified, set to sampling+cmaes.")
    else:
        availableacqoptimizer = ['sampling+cmaes', 'sampling+fmincon', 'cmaes', 'fmincon']
        if moboInfo["acquifuncopt"].lower() not in availableacqoptimizer:
            raise TypeError(moboInfo["acquifuncopt"], " is not a valid acquisition function optimizer.")
        else:
            print("The acquisition function optimizer is specified to ", moboInfo["acquifuncopt"], " by user")

    if "nrestart" not in moboInfo:
        moboInfo["nrestart"] = 1
        print(
            "The number of restart for acquisition function optimization is not specified, setting BayesInfo.nrestart to 1.")
    else:
        if moboInfo["nrestart"] < 1:
            raise ValueError("BayesInfo['nrestart'] should be at least one")
        print("The number of restart for acquisition function optimization is specified to ",
              moboInfo["nrestart"], " by user")

    if "filename" not in moboInfo:
        moboInfo["filename"] = "temporarydata.mat"
        print("The file name for saving the results is not specified, set the name to temporarydata.mat")
    else:
        print("The file name for saving the results is not specified, set the name to ", moboInfo["filename"])

    return moboInfo
