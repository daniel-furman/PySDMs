# Module: PySDMs/internal
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Last modified : 4/10/21
# https://github.com/daniel-furman/PySDMs

import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot as plt, style
from pycaret import classification as pycaret
import pickle as pk

def validation_visuals(data, F1_min_seed, F1_max_seed, AUC_seed,
      pycaret_outdir='outputs/'):

      """A function that generates a validation-set F1 score boxplot and a AUC
      ROC analysis plot with CV. The function was developed for visualizing
      performance across multiple runs between consecutive seed ints.

      F1_min_seed/F1_max_seed: int
          The min and max seed model runs to grab for the F1 boxplot.

      AUC_seed: int
          The seed model run to perform CV ROC analysis."""

      # ####################################################################
      # F1 Validation Scores Data IO

      validation_scores_ensemble, validation_box_plots = list(), list()
      for i in np.arange(F1_min_seed, F1_max_seed).tolist():
          validation_scores_ensemble.append(pd.read_csv(pycaret_outdir +
          'holdout_' + str(i) + '.csv', index_col = 'Unnamed: 0')['0'])
      validation_scores_ensemble = pd.DataFrame(validation_scores_ensemble)
      for i in np.arange(0,len(list(validation_scores_ensemble))):
          validation_box_plots.append(validation_scores_ensemble[i])
      validation_scores_individual = list()
      for i in np.arange(0, len(validation_scores_ensemble[0])):
          validation_scores_individual.append(np.max(
              validation_scores_ensemble.iloc[i,:]))

      # ####################################################################
      # F1 Score Hold-Out Set BoxPlot across many runs

      style.use('ggplot')
      plt.rcParams["figure.figsize"] = (2.25, 5.25)
      plt.boxplot(validation_scores_individual, 'k+', 'k^',
          medianprops = dict(color='black'), labels=['final_voter'])
      plt.title('Validation-Set F1 Scores')
      plt.ylabel('F1 score')
      y = validation_scores_individual
      x = np.random.normal(1, 0.030, size=len(y))
      plt.plot(x, y, 'r.', alpha=0.35, markersize=11.5)
      plt.savefig(pycaret_outdir + 'F1_score.png', dpi=400)
      plt.show()

      # ####################################################################
      # AUC K-fold Cross Validation ROC Plot for one run

      style.use('default')

      # Data IO
      X = data
      y = X['pa']
      X = X.drop(['pa'], axis=1)
      X = pd.DataFrame(sklearn.preprocessing.StandardScaler(
          ).fit_transform(X), columns=list(X))
      n_samples, n_features = X.shape
      cv = sklearn.model_selection.StratifiedKFold(n_splits=10)
      with open(pycaret_outdir + 'xv_' + str(AUC_seed) + '.pkl', 'rb') as f:
          classifier = pk.load(f)
          # Classification and ROC Analysis
      tprs, aucs = list(), list()
      mean_fpr = np.linspace(0, 1, 100)
      fig, ax = plt.subplots(figsize=(7.5,4.5))
      for i, (train, test) in enumerate(cv.split(X, y)):
          classifier.fit(X.iloc[train, :], y.iloc[train])
          viz = sklearn.metrics.plot_roc_curve(classifier, X.iloc[
              test, :], y.iloc[test], alpha=0.4, lw=1, ax=ax)
          interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
          interp_tpr[0] = 0.0
          tprs.append(interp_tpr)
          aucs.append(viz.roc_auc)
      ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          label='Chance', alpha=.8)
      mean_tpr = np.mean(tprs, axis=0)
      mean_tpr[-1] = 1.0
      mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
      std_auc = np.std(aucs)
      ax.plot(mean_fpr, mean_tpr, color='b',
          label=r'final_voter (AUC = %0.3f $\pm$ %0.3f)' % (
          mean_auc, std_auc), lw=2, alpha=.8)
      std_tpr = np.std(tprs, axis=0)
      tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
      tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
      ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey',
          alpha=.2, label=r'$\pm$ 1 std. deviation')
      ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
          title="ROC Curve: 10-fold Cross Validation")
      ax.legend(loc="lower right")
      handles, labels = ax.get_legend_handles_labels()
      ax.legend(handles[-3:], labels[-3:])
      plt.savefig(pycaret_outdir + 'ROC_plot.png', dpi = 400)
      plt.show()
