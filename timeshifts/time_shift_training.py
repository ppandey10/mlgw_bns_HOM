import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Sans Serif']})
rc('text', usetex=True)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor

from src.neural_network import TimeshiftsGPR, TimeshiftsNN

training_timeshifts = np.load('timeshifts_20k_q_3.0.npy')
training_param_array = np.load('params_20k_q_3.0.npy')

print(training_param_array.shape)
print(training_timeshifts.shape)

param_names = ['q', 'l1', 'l2', 'c1', 'c2']

def train(use_gpr: bool = False):
    """Train the time-shift model (RFF+Ridge by default, GPR if requested)."""
    if use_gpr:
        timeshifts_model = TimeshiftsGPR(
            training_params=training_param_array,
            training_timeshifts=training_timeshifts,
        ).fit()
        timeshifts_model.save_model(filename="timeshifts_model_HOM.pkl")
        return

    timeshifts_model = TimeshiftsNN(
        training_params=training_param_array,
        training_timeshifts=training_timeshifts,
    ).fit()
    timeshifts_model.save_model(filename="timeshifts_rff_surrogate.pkl")


if __name__ == '__main__':
    train()