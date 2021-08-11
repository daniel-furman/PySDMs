# Module: PySDMs
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Last modified : 8.11.21

from PySDMs import PySDMs
import numpy as np
import pandas as pd
import os
from pycaret import classification as pycaret, internal


def test_PySDMs():

    #Data IO
    DATA = os.path.dirname(os.path.abspath(__file__))
    data = [
        os.path.join(DATA, x)
        for x in ['data/env_train/env_train_cr_192_metadata.csv',
                  'data/train-rasters/bclim*.tif']
    ]
    output = [
        os.path.join(DATA, x)
        for x in ['outputs/']][0]

    # Run PySDMs object over the random seed(s)
    species_code = 'cr'

    for seed in [192,]:

        # Data IO
        df_input_dir = 'test/data/env_train/env_train_'+species_code+'_'
        tif_input_dir = data[1]
        mod_list = ['rf', 'et']
        col_names = ['pa','bclim7', 'bclim4', 'bclim3', 'bclim2', 'bclim5', 'bclim6',
           'bclim11', 'bclim8', 'bclim9', 'bclim10', 'bclim15', 'bclim13',
           'bclim19', 'bclim16', 'bclim12', 'bclim17', 'bclim1', 'bclim18']

        data = pd.read_csv(data[0])
        exp_name = species_code+'_'+str(seed)
        train_data = data[(data['fold']!=3) | (data['fold']!=4)][col_names]
        test_data = data[(data['fold']==3) | (data['fold']==4)][col_names]
        print()
        print('test % of total data: ', len(test_data['pa'])/(len(train_data['pa'])+len(test_data['pa'])))

        # Initialize class
        coastal_redwoods = PySDMs(train_data, test_data, seed, target='pa',
                exp_name=species_code+'_'+str(seed), normalize=False, metric='AUC',
                silent=True, mod_list=mod_list)

        # Model Fitting with self.fit()
        learner = coastal_redwoods.fit()
        pd.Series(pycaret.get_config('X_train').columns).to_csv(output+'features_'+str(seed)+'.csv')
        assert os.path.isfile(output + 'cr_192.pkl')

        # Geo-classification with self.interpolate()
        coastal_redwoods.interpolate(tif_input_dir, df_input_dir, output)
        coastal_redwoods.validation_visuals(min_seed=seed, max_seed=seed+1, pycaret_outdir=output)

    assert os.path.isfile(output + 'probability_1.tif')
