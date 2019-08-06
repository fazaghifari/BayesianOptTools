import numpy as np
import multiprocessing as mp
from misc.sampling.samplingplan import sampling, standardize
from sklearn.cross_decomposition.pls_ import PLSRegression as pls
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize, fmin_cobyla
from surrogate_models.supports.trendfunction import polytruncation, compute_regression_mat
from surrogate_models.supports.krigloocv import loocv
import cma
import logging


class Kriging:
    """
        Create Kriging model based on the information from inputs and global variables.
        Inputs:
        KrigInfo (dict): Containing necessary information to create a Kriging model
        ub (int): log10 of hyperparam upper bound. Defaults to 5.
        lb (int): log10 of hyperparam lower bound. Defaults to -5.
        standardization (bool): Perform standardization to the samples. Defaults to False.
        standtype (str): Type of standardization. Defaults to "default".
            available options are "default" and "std"
        normy (bool): True or False, normalize y or not
        trainvar (bool): True or False, train Kriging variance or not

        Outputs:
         KrigInfo (dict): Trained kriging model

        Details of KrigInfo:
        REQUIRED PARAMETERS. These parameters need to be specified manually by
        the user. Otherwise, the process cannot continue.
            - KrigInfo['lb'] - Variables' lower bounds.
            - KrigInfo['ub'] - Variables' upper bounds.
            - KrigInfo['nvar'] - Number of variables.
            - KrigInfo['nsamp'] - Number of samples.
            - KrigInfo['X'] - Experimental design.
            - KrigInfo['y'] - Responses of the experimental design.

        EXTRA PARAMETERS. These parameters can be set by the user. If not
        specified, default values will be used (or computed for the experimetntal design and responses)
            - KrigInfo['problem'] - Function name to evaluate responses (No need to specify this if KrigInfo.X and KrigInfo.Y are specified).
            - KrigInfo['nugget'] - Nugget (noise factor). Default: 1e-6
            - KrigInfo['TrendOrder'] - Polynomial trend function order (note that this code uses polynomial chaos expansion). Default: 0 (ordinary Kriging).
            - KrigInfo['kernel'] - Kernel function. Available kernels are 'gaussian', 'exponential','matern32', 'matern52', and 'cubic'. Default: 'Gaussian'.
            - KrigInfo['nrestart'] - Number of restarts for hyperparameters optimization. Default: 1.
            - KrigInfo['LOOCVtype'] - Type of cross validation error. Default: 'rmse'.
            - KrigInfo['optimizer'] - Optimizer for hyperparameters optimization. Default: 'lbfgsb'.

        """

    def __init__(self, KrigInfo, num=None, ub=5, lb=-5, standardization=False, standtype="default", normy=True,
                 trainvar=True,
                 disp='WARNING'):
        """
        Initialize Kriging model

       Args:
            KrigInfo (dict): Dictionary that contains Kriging information.
            ub (int): log10 of hyperparam upper bound. Defaults to 5.
            lb (int): log10 of hyperparam lower bound. Defaults to -5.
            standardization (bool): Perform standardization to the samples. Defaults to False.
            standtype (str): Type of standardization. Defaults to "default".
                available options are "default" and "std"
            normy (bool): True or False, normalize y or not
            trainvar (bool): True or False, train Kriging variance or not
        """
        if trainvar is True:
            self.nbhyp = KrigInfo["nvar"] + 1
        else:
            self.nbhyp = KrigInfo["nvar"]

        KrigInfo = kriginfocheck(KrigInfo, lb, ub, self.nbhyp, loglvl=disp)
        KrigInfo["n_princomp"] = False
        self.type = 'kriging'
        self.KrigInfo = KrigInfo
        self.normy = normy
        self.standardization = standardization
        self.standtype = standtype
        self.num = num
        self.Y = KrigInfo["y"]
        self.X = KrigInfo["X"]

    def standardize(self):
        """
        Standardize Kriging samples and create regression matrix.

        Returns:
            None
        """
        # Create regression matrix
        self.KrigInfo['idx'] = polytruncation(self.KrigInfo["TrendOrder"], self.KrigInfo["nvar"], 1)

        # Standardize X and y
        if self.standardization is True:
            if self.standtype.lower() == "default":  # If standardization type is set to 'default'
                self.KrigInfo["normtype"] = "default"

                # Create normalization bound from -1 to 1
                bound = np.vstack((- np.ones(shape=[1, self.KrigInfo["nvar"]]),
                                   np.ones(shape=[1, self.KrigInfo["nvar"]])))

                # Normalize sample to -1 and 1
                if self.normy is True:  # If normalize y
                    self.KrigInfo["X_norm"], self.KrigInfo["y_norm"] = standardize(self.X, self.Y,
                                                                                   type=self.standtype.lower(),
                                                                                   normy=True,
                                                                                   range=np.vstack(
                                                                                       (np.hstack((self.KrigInfo["lb"],
                                                                                                   np.min(self.Y))),
                                                                                        np.hstack((self.KrigInfo["ub"],
                                                                                                   np.max(self.Y))))))
                    self.KrigInfo["norm_y"] = True

                else:
                    self.KrigInfo["X_norm"] = standardize(self.X, self.Y,
                                                          type=self.standtype.lower(),
                                                          range=np.vstack((self.KrigInfo["lb"], self.KrigInfo["ub"])))
                    self.KrigInfo["norm_y"] = False

            else:  # If standardization type is set to 'std'
                self.KrigInfo["normtype"] = "std"

                # create normalization with mean 0 and standard deviation 1
                if self.normy is True:
                    self.KrigInfo["X_norm"], self.KrigInfo["y_norm"], \
                    self.KrigInfo["X_mean"], self.KrigInfo["y_mean"], \
                    self.KrigInfo["X_std"], self.KrigInfo["y_std"] = standardize(
                        self.X, self.Y, type=self.standtype.lower(), normy=True)

                    self.KrigInfo["norm_y"] = True
                else:
                    self.KrigInfo["X_norm"], self.KrigInfo["X_mean"], self.KrigInfo["X_std"] = \
                        standardize(self.X, self.Y, type=self.standtype.lower())
                    self.KrigInfo["norm_y"] = False

                bound = np.vstack((np.min(self.KrigInfo["X_norm"], axis=0),
                                   np.max(self.KrigInfo["X_norm"], axis=0)))

            self.KrigInfo["standardization"] = True
            self.KrigInfo["F"] = compute_regression_mat(self.KrigInfo["idx"], self.KrigInfo["X_norm"], bound,
                                                        np.ones(shape=[self.KrigInfo["nvar"]]))
        else:
            self.KrigInfo["standardization"] = False
            self.KrigInfo["norm_y"] = False
            bound = np.vstack((np.min(self.KrigInfo["X"], axis=0),
                               np.max(self.KrigInfo["X"], axis=0)))
            self.KrigInfo["F"] = compute_regression_mat(self.KrigInfo["idx"], self.KrigInfo["X"], bound,
                                                        np.ones(shape=[self.KrigInfo["nvar"]]))

    def train(self, loglvl='WARNING'):
        """
        Train Kriging model
        
        Args:
            loglvl (str): level of logging function.
            
        Returns:
            None
        """""
        logging.basicConfig(level=loglvl)
        logging.INFO("Begin train hyperparam.")

        # Create multiple starting points
        if self.KrigInfo['nrestart'] < 1:
            xhyp = self.nbhyp * [0]
        else:
            _, xhyp = sampling('sobol', self.nbhyp, self.KrigInfo['nrestart'],
                               result="real", upbound=self.KrigInfo["ubhyp"], lobound=self.KrigInfo["lbhyp"])

        # Create array of solution candidate.
        bestxcand = np.zeros(shape=[self.KrigInfo['nrestart'], self.nbhyp])
        neglnlikecand = np.zeros(shape=[self.KrigInfo['nrestart']])

        # Optimize hyperparam if number of hyperparameter is 1
        if self.nbhyp == 1:
            res = minimize_scalar(likelihood.likelihood, bounds=(self.lb, self.ub), method='golden')
            best_x = np.array([res.x])


def kriginfocheck(KrigInfo, lb, ub, nbhyp, loglvl='WARNING'):
    """
    Function to check the Kriging information and set Kriging Information to default value if
    required parameters are not supplied.

    Args:
        KrigInfo: Dictionary that contains Kriging information.
        ub (int): log10 of hyperparam upper bound. Defaults to 5.
        lb (int): log10 of hyperparam lower bound. Defaults to -5.
        nbhyp (int): number of hyperparameter
        loglvl (str): level of logging function

    Returns:
        KrigInfo: Checked/Modified Kriging Information

    """
    logging.basicConfig(level=loglvl)
    eps = np.finfo(float).eps

    # Check if number of restart is specified. If not set to 1
    if 'nrestart' not in KrigInfo:
        KrigInfo['nrestart'] = 1
    elif KrigInfo['nrestart'] < 1:
        raise ValueError('Minimum value of KrigInfo["nrestart"] is 1')

    # Check if optimizer option is specified. If not set to lbfgsb
    if 'optimizer' not in KrigInfo:
        KrigInfo['optimizer'] = 'lbfgsb'
        print('The optimizer is not specified, set to lbfgsb')
    else:
        availoptmzr = ["lbfgsb", "cmaes", "cobyla", "slsqp"]  # check if the specified optimizer is available
        if KrigInfo['optimizer'].lower() not in availoptmzr:
            raise ValueError(KrigInfo["optimizer"], " is not a valid optimizer.")
        logging.INFO("The acquisition function is specified to ", KrigInfo["optimizer"], " by user")

    # Check if Trend order is specified. If not set to zero
    if 'TrendOrder' not in KrigInfo:
        KrigInfo['TrendOrder'] = 0
    elif KrigInfo['TrendOrder'] < 0:
        raise ValueError("The order of the polynomial trend should be a positive value.")
    else:
        pass

    # Check if nugget settings are specified. If not, set to -6 and 'fixed'
    if "nugget" not in KrigInfo:
        KrigInfo["nugget"] = -6
        KrigInfo["nuggetparam"] = "fixed"
        print("Nugget is not defined, set nugget to 1e-06")
    elif type(KrigInfo["nugget"]) is not list:
        KrigInfo["nuggetparam"] = "fixed"
        nnugget = 1
    elif len(KrigInfo["nugget"]) == 2:
        KrigInfo["nuggetparam"] = "tuned"
        nnugget = len(KrigInfo["nugget"]);
        if KrigInfo["nuggetparam"][0] > KrigInfo["nuggetparam"][1]:
            raise TypeError("The lower bound of the nugget should be lower than the upper bound.")

    # Check Kernel Settings
    if "kernel" not in KrigInfo:
        KrigInfo["kernel"] = ["gaussian"]
        nkernel = 1
        print("Kernel is not defined, set kernel to gaussian")
    elif type(KrigInfo["kernel"]) is not list:
        nkernel = 1
        KrigInfo["kernel"] = [KrigInfo["kernel"]]
    else:
        nkernel = len(KrigInfo["kernel"])
    KrigInfo["nkernel"] = nkernel

    # Check overall hyperparam
    lbhyp = lb * np.ones(shape=[nbhyp])
    ubhyp = ub * np.ones(shape=[nbhyp])
    if nnugget == 1 and nkernel == 1:  # Fixed nugget, one kernel function
        pass
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
    elif nnugget > 1 and nkernel == 1:  # Tunable nugget, one kernel function
        lbhyp = np.hstack((lbhyp, KrigInfo["nugget"][0]))
        ubhyp = np.hstack((ubhyp, KrigInfo["nugget"][1]))
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
        scaling[-1] = (KrigInfo["nugget"][1] - KrigInfo["nugget"][0]) / (ub - lb)
    elif nnugget == 1 and nkernel > 1:  # Fixed nugget, multiple kernel functions
        lbhyp = np.hstack((lbhyp, np.zeros(shape=[nkernel]) + eps))
        ubhyp = np.hstack((ubhyp, np.ones(shape=[nkernel])))
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
        scaling[-nkernel:] = 1 / (ub - lb)
    elif nnugget > 1 and nkernel > 1:  # Tunable nugget, multiple kernel functions
        lbhyp = np.hstack((lbhyp, KrigInfo["nugget"][0], np.zeros(shape=[nkernel]) + eps))
        ubhyp = np.hstack((ubhyp, KrigInfo["nugget"][1], np.ones(shape=[nkernel])))
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
        scaling[-nkernel - 1] = (KrigInfo["nugget"][1] - KrigInfo["nugget"][0]) / (ub - lb)
        scaling[-nkernel:] = 1 / (ub - lb)
    KrigInfo["lbhyp"] = lbhyp
    KrigInfo["ubhyp"] = ubhyp

    return KrigInfo
