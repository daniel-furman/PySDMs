# Module: PySDMs
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Last modified : 4.10.2021

from PySDMs import PySDMs
import numpy as np
import pandas as pd
import os

seed = 190

DATA = os.path.dirname(os.path.abspath(__file__))
test_data = [
    os.path.join(DATA, x)
    for x in ['data/train-rasters-2.5m/bclim*.asc','data/env_train/env_train_xv_']
]

test_output = [
    os.path.join(DATA, x)
    for x in ['outputs/']
]


def test_PySDMs():

    # Simplified class use:
    # Run PySDMs object over random seed of choice
    # ####################################################################
    # Data IO

    data = pd.read_csv(test_data[1]+str(seed) + '.csv')

    exp_name = 'xv_'+str(seed)
    mod_list = ['et', 'rf']

    asc_input_dir = test_data[0]
    df_input_dir = test_data[1]


    # ####################################################################
    # Class

    # Initialization
    x_vigilis = PySDMs(data, seed, 'pa', 'xv_'+str(seed),
        normalize=False, silent=True, mod_list=mod_list)

    # Model Fitting with self.fit() and model inspection
    learner = x_vigilis.fit()
    x_vigilis.validation_visuals(190, 191, AUC_seed=190)

    # Geo-classification with self.interpolate()
    x_vigilis.interpolate(asc_input_dir, df_input_dir, test_output[0], seed)

    # The final output:
    assert os.path.isfile(test_output[0] + 'probability_1.tif')
