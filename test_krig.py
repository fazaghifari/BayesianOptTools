import sys
sys.path.insert(0, '/home/tjim/BayesianOptTools')

import logging
import time

import numpy as np
import miscellaneous.surrogate_support.initinfo
import surrogate_models.kriging

# Only used for testing model
import matplotlib.cm
import matplotlib.pyplot as plt
import miscellaneous.sampling.samplingplan
import miscellaneous.surrogate_support.prediction
import testcase.analyticalfcn.cases


def generate_kriging_model(samples, responses):
    """Generate a Kriging model.

    Generate a Kriging model based on input sample variables and
    corresponding responses using Pram's best-practise Kriging code.

    Args:
        samples (np.array): A 2D array of sample design variables o
            shape (n_samples, n_variables).
        responses (np.array): The responses to the design variables.
    """
    # Set Kriging model parameters (start with auto-gen defaults)
    krig_params = miscellaneous.surrogate_support.initinfo.initkriginfo('single')
    krig_params['X'] = samples
    krig_params['y'] = responses
    krig_params['nvar'] = samples.shape[1]
    krig_params['nsamp'] = samples.shape[0]
    krig_params['nrestart'] = 5
    krig_params['ub'] = samples.max(axis=0)
    krig_params['lb'] = samples.min(axis=0)
    krig_params['kernel'] = ['gaussian']
    krig_params['TrendOrder'] = 0
    krig_params['nugget'] = -6
    krig_params['n_princomp'] = 2

    # Run Kriging
    t = time.time()
    model = surrogate_models.kriging.kriging(krig_params,
                                             standardization=True,
                                             normtype="default",
                                             normalize_y=False,
                                             disp=True)
    elapsed = time.time() - t
    logging.info(f'Elapsed Kriging model training time: {elapsed:.2f} s.')
    return model


def predict(model, inputs):
    """Extract result from Kriging model with sample inputs.

    Args:
        model (dict): KrigInfo datastructure as produced by
            generate_kriging_model. Internally by
            surrogate_models.kriging.kriging.
        inputs (np.array): Array of input variables to query.
    """
    neval = 10000
    sampling = miscellaneous.sampling.samplingplan.sampling
    samplenormout, sampleeval = sampling('rlh',
                                         model['nvar'],
                                         neval,
                                         result="real",
                                         upbound=model['ub'],
                                         lobound=model['lb'])
    xx = np.linspace(-5, 10, 100)
    yy = np.linspace(0, 15, 100)
    Xevalx, Xevaly = np.meshgrid(xx, yy)
    Xeval = np.zeros(shape=[neval, 2])
    Xeval[:, 0] = np.reshape(Xevalx, (neval))
    Xeval[:, 1] = np.reshape(Xevaly, (neval))

    #Evaluate output
    yeval = np.zeros(shape=[neval,1])
    yact = np.zeros(shape=[neval,1])
    yeval= miscellaneous.surrogate_support.prediction.prediction(Xeval, model, "pred")
    yact = testcase.analyticalfcn.cases.evaluate(Xeval,"branin")
    hasil = np.hstack((yeval,yact))

    #Evaluate RMSE
    subs = np.transpose((yact-yeval))
    subs1 = np.transpose((yact-yeval)/yact)
    RMSE = np.sqrt(np.sum(subs**2)/neval)
    RMSRE = np.sqrt(np.sum(subs1**2)/neval)
    MAPE = 100*np.sum(abs(subs1))/neval
    print("RMSE = ",RMSE)
    print("RMSRE = ",RMSRE)
    print("MAPE = ",MAPE,"%")

    yeval1 = np.reshape(yeval,(100,100))
    x1eval = np.reshape(Xeval[:,0],(100,100))
    x2eval = np.reshape(Xeval[:,1],(100,100))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x1eval, x2eval, yeval1, cmap=matplotlib.cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()


if __name__ == '__main__':

    logging.basicConfig(level='DEBUG',
                        format='%(levelname)s - %(message)s')

    # # Imports for generating test samples
    # import testcase.analyticalfcn.cases
    # import miscellaneous.sampling.samplingplan
    #
    # # Generate test samples
    # n_samples = 15
    # n_vars = 2
    # lb = np.array([-5, 0])
    #
    # sampling = miscellaneous.sampling.samplingplan.sampling
    # samplenorm, design_samples = sampling('rlh', n_vars, n_samples, result="real",
    #                                       upbound=ub, lobound=lb)
    #
    # # Evaluate samples with a test function
    # responses = testcase.analyticalfcn.cases.evaluate(design_samples, "branin")

    # Sampling
    design_samples = np.loadtxt("xi.dat")
    responses = np.transpose([np.loadtxt("cd.dat")])

    model = generate_kriging_model(design_samples, responses)
    predict(model, inputs)
