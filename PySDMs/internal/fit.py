# Module: PySDMs/internal
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Last modified : 8/9/21
# https://github.com/daniel-furman/PySDMs

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, style
from pycaret import classification as pycaret, internal
import pickle as pk

def internal_fit(data, test_data, seed, target, exp_name, normalize, metric, fold,
    silent, mod_list=['et', 'catboost', 'rf', 'lightgbm', 'xgboost', 'gbc'],
    pycaret_outdir='outputs/', deep_learning=False, soft_voters=True, tuning=True):

      """Semi-Auto ML classification training with PyCaret, considering tree-based
        methods, neural nets (on CPU), and two-model soft voters (exhaustive
        search). Requires a Pandas data-frame with a target and explanatory
        features. Returns the voter with the best validation
        metric performance (user-defined metric). See Pycaret.org for more
        and for customization purposes (classification module).

      Returns:
          The learner with the best validation metric performance."""

      # ####################################################################
      # PyCaret Model Fitting with BRTs and soft voters

      # Setup PyCaret environment
      exp_clf = pycaret.setup(data, target=target, test_data=test_data,
          session_id=seed, experiment_name=exp_name,
          normalize=normalize, fold=fold, log_experiment=True, verbose=False,
          fold_strategy='stratifiedkfold', silent=silent) #numeric_features=[]

      # K-fold cross validation training with PyCaret "compare_models"
      classifier_list = pycaret.compare_models(n_select=len(mod_list),
          verbose=False, include=mod_list, sort=metric)
      pycaret_training_df = pycaret.pull()
      mod_keys = pycaret_training_df.index.tolist()

      if tuning:
          #custom rf grid search values for hpo
          n_estimators = [50, 60, 70, 80]
          sqrt_features = int(np.sqrt(len(pycaret.get_config('X_train').iloc[0,:])))
          max_features = [sqrt_features, sqrt_features+1, sqrt_features+2]
          min_samples_split= [sqrt_features, sqrt_features+1, sqrt_features+2]
          rf_tuned_parameters = {'max_features': max_features, 'min_samples_split': min_samples_split, 'n_estimators': n_estimators}

          #tune rf
          rf = pycaret.create_model('rf', verbose=False)
          tuned_rf, tuner_object = pycaret.tune_model(rf, verbose=False,
                custom_grid=rf_tuned_parameters, return_tuner=True)
          classifier_list.append(tuned_rf)
          mod_keys.append('tuned_rf')

          #tune rf
          et = pycaret.create_model('et', verbose=False)
          tuned_et, tuner_object = pycaret.tune_model(et, verbose=False,
                custom_grid=rf_tuned_parameters, return_tuner=True)
          classifier_list.append(tuned_et)
          mod_keys.append('tuned_et')

          #tune catboost
          catboost = pycaret.create_model('catboost', verbose=False)
          tuned_cb, tuner_object = pycaret.tune_model(catboost, verbose=False,
                search_library='optuna', return_tuner=True)
          classifier_list.append(tuned_cb)
          mod_keys.append('tuned_cb')

      # Build soft voters with best subset selection among 2 models blends
      if soft_voters:
          scoring_pandas = pd.DataFrame(np.zeros(7).reshape(-1, 7),
                columns=['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa',
                'MCC'])
          soft_voting_models = []
          for j in np.arange(0, len(mod_list)):
              for i in np.arange(0, len(mod_list)):
                  if i != j:
                      soft_voter = pycaret.blend_models(estimator_list=[classifier_list[j],
                            classifier_list[i]], method='soft', verbose=False)
                      soft_voting_models.append(soft_voter)
                      pull_temp = pycaret.pull()
                      scoring_pandas = scoring_pandas.append(pull_temp.loc['Mean'])
          scoring_pandas.drop(scoring_pandas.index[0], inplace=True)
          scoring_pandas.index = np.arange(0, len(scoring_pandas['Accuracy']))
          # Grab the most accurate soft_voter
          soft_voter = soft_voting_models[scoring_pandas['F1'].idxmax()]
          classifier_list.append(soft_voter)
          mod_keys.append('soft_voter')

      # Add deep learners if deep_learning (also set normalization to True)
      if deep_learning:
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
      # PyCaret Model Evaluation

      # Comparing cross val and hold-out strategies
      # Evaluate the models with k-fold cv and held-out partitions
      cv_mean, names, ho_mean = [], [], []
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
      if tuning:
          finalizing_dictionary = {'tuned_rf':tuned_rf, 'tuned_et':tuned_et, 'tuned_cb':tuned_cb}
          for key_iter in finalizing_dictionary:
              if model_key == key_iter:
                  final_model_key = finalizing_dictionary[key_iter]

      # Refit the best model with 100% of the training data
      final_voter = pycaret.finalize_model(final_model_key)

      # Test dumping and loading of .pkl model.
      with open(pycaret_outdir+exp_name+'.pkl', 'wb') as f:
          pk.dump(final_voter, f)
      with open(pycaret_outdir+exp_name+'.pkl', 'rb') as f:
          voter_pkl = pk.load(f)

      print('     >', model_key)
      print('     >', voter_pkl)

      # Returns the best learner
      return(final_voter)
