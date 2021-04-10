# Module: PySDMs/internal
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Last modified : 4/10/21
# https://github.com/daniel-furman/PySDMs

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, style
from pycaret import classification as pycaret, internal
import pickle as pk

def fit(data, seed, target, exp_name, train_size, normalize, metric, fold,
    silent, mod_list, pycaret_outdir='outputs/', deep_learning=False):

      """Model training with PyCaret pipelines. Constructs soft voters via
      exhaustive best subset selection among the above BRTs (below mod_list).
      The function then adds an assortment of neural networks, at two, three,
      and four hidden layers. The final model is selected from the best
      validation (F1 score, 30% hold out  default) set and saves the .pkl to
      file alongside the scores.

      Returns:
          The voter with the best validation metric performance."""

      # ####################################################################
      # PyCaret Model Fitting with BRTs and soft voters

      # Setup PyCaret environment
      exp_clf = pycaret.setup(data, target=target,
          session_id=seed, experiment_name=exp_name,
          normalize=normalize, train_size=train_size,
          fold=fold, log_experiment=True, verbose=False,
          fold_strategy='stratifiedkfold', silent=silent)

      # K-fold cross validation training with PyCaret "compare_models"
      classifier_list = pycaret.compare_models(n_select=len(
          mod_list), verbose=False, include=mod_list,
          sort=metric)
      pycaret_training_df = pycaret.pull()
      mod_keys = pycaret_training_df.index.tolist()

      # Build soft voters with best subset selection among the BRTs
      scoring_pandas = pd.DataFrame(np.zeros(7).reshape(-1, 7),
          columns=['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa',
          'MCC'])
      soft_voting_models = list()
      for j in np.arange(0, len(mod_list)):
          for i in np.arange(j+2, len(mod_list)+1):
              soft_voter = pycaret.blend_models(
                  estimator_list=classifier_list[j:i], method='soft',
                  verbose=False)
              soft_voting_models.append(soft_voter)
              pull_temp = pycaret.pull()
              scoring_pandas = scoring_pandas.append(pull_temp.loc['Mean'])
      # Drop dummy start row and reindex
      scoring_pandas.drop(scoring_pandas.index[0], inplace=True)
      scoring_pandas.index = np.arange(0, len(scoring_pandas['Accuracy']))
      # Grab the most accurate soft_voter
      soft_voter = soft_voting_models[scoring_pandas['F1'].idxmax()]
      # Add the various keys and models to the class vars
      classifier_list.append(soft_voter)
      mod_keys.append('soft_voter')

      # ####################################################################
      # PyCaret Model Fitting with neural nets

      # Add deep learners if deep_learning (also set normalization to True)
      if deep_learning:
          # Create the architectures
          MLP_2 = internal.tunable.TunableMLPClassifier(hidden_layer_sizes=(300, 150),
              activation='relu', solver='adam', random_state=seed)
          MLP_2 = pycaret.create_model(MLP_2, verbose=False)
          MLP_3 = internal.tunable.TunableMLPClassifier(hidden_layer_sizes=(300, 150,
              100), activation='relu', solver='adam', random_state=seed)
          MLP_3 = pycaret.create_model(MLP_3, verbose=False)
          MLP_4 = internal.tunable.TunableMLPClassifier(hidden_layer_sizes=(450, 300, 150,
              100), activation='relu', solver='adam', random_state=seed)
          MLP_4 = pycaret.create_model(MLP_4, verbose=False)

          # Add another soft voter with BRT + deep learners
          soft_voter_dl = pycaret.blend_models(estimator_list=[
              classifier_list[0], MLP_2, MLP_3, MLP_4],
              method='soft', verbose=False)

          # Add to function vars
          classifier_list.append(MLP_2)
          classifier_list.append(MLP_3)
          classifier_list.append(MLP_4)
          classifier_list.append(soft_voter_dl)
          mod_keys.append('MLP_2')
          mod_keys.append('MLP_3')
          mod_keys.append('MLP_4')
          mod_keys.append('soft_voter_dl')

      # ####################################################################
      # PyCaret Model Evaluation, with cross val and hold-out

      # Evaluate the models with k-fold cv and held-out partitions
      cv_mean, names, ho_mean = list(), list(), list()

      print('The RandomState seed for this run is ' + str(seed) + '\n')
      for i in np.arange(0, len(classifier_list)):
          # first cv:
          pycaret.create_model(classifier_list[i], verbose=False)
          scorei = pycaret.pull()
          cv_mean.append(scorei[metric].iloc[0:fold])
          names.append(mod_keys[i])
          # then validation:
          pycaret.predict_model(classifier_list[i], verbose=False)
          scorei = pycaret.pull()
          ho_mean.append(scorei[metric].loc[0])
          if not silent:
              print('    >%s: CV (mean/sd): %.4f (%.4f) | Hold-out: %.4f' %
                  (names[i], np.mean(cv_mean[i]),
                  np.std(cv_mean[i]), ho_mean[i]))

      # ####################################################################
      # Finalize

      # Write the held out validation metrics to file:
      pd.DataFrame(ho_mean).to_csv(pycaret_outdir+'holdout_'+str(
          seed)+'.csv')

      # Grab model with best validation performance
      ho_mean_np = np.array(ho_mean)
      id_argmax = ho_mean_np.argmax(axis=0)
      model_key = mod_keys[id_argmax]

      # Grab the best model
      final_model_key = model_key
      if model_key == 'soft_voter':
          final_model_key = soft_voter

      if deep_learning:
          finalizing_dictionary = {'soft_voter_dl':soft_voter_dl,
            'MLP_4':MLP_4, 'MLP_2':MLP_2, 'MLP_3':MLP_3}
          for key_iter in finalizing_dictionary:
              if model_key == key_iter:
                  final_model_key = finalizing_dictionary[key_iter]


      # Refit the best model with 100% of the training data
      final_voter = pycaret.finalize_model(final_model_key)
      with open(pycaret_outdir+exp_name+'.pkl', 'wb') as f:
          pk.dump(final_voter, f)
      with open(pycaret_outdir+exp_name+'.pkl', 'rb') as f:
          voter_pkl = pk.load(f)
      print('     >', model_key)
      if not silent:
          print('     >', voter_pkl)

      # Return the final classifier to end the function
      return(final_voter)
