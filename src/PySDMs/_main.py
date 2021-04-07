# Module: PySDMs
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Last modified : 4/7/21
# https://github.com/daniel-furman/PySDMs

import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot as plt, style
from pycaret import classification as pycaret, internal
import geopandas as gpd
import rasterio
from pyimpute import load_targets, impute
from IPython.display import display, Markdown
from IPython import get_ipython
import glob
import pickle as pk
import pylab

class PySDMs(object):

    """
    An object-oriented class for Species Distribution Modeling (SDM).
    PySDMs does most of its heavy lifting in the modeling portion of the
    SDM framework, with the interpolate functions mainly packaging
    the geo-classification step in an object-oriented mannner. The pre-processing
    steps of a SDM workflow are left out, primarily because they are easier to
    do in R (see bib links at the end of the Jupyter notebook in examples/).

    PySDMs was primarily developed for my research project on climate change
    impacts for Joshua tree and Desert Night Lizards.

    Functions
    -------

    self.fit(): Model training with PyCaret, considering tree-based
        methods, neural nets, and best-subset-selection soft voting blends.
        Requires a data-frame with a classification target and numerical
        explanatory features. Returns the voter with the best validation
        metric performance (default metric=F1).

    self.interpolate(): Geo-classification function for model interpolation to
        raster feature surfaces. Saves to file both probabilistic and binary
        distribution predictions.

    self.validation_performance(): F1 score and AUC visuals. Oriented for
        PySDMs workflows with multiple runs (see examples).

    """

    def __init__(self, data, seed, target, exp_name, train_size=0.7,
        normalize=True, metric='F1', fold=10, silent=False):

        """
        data: Pandas DataFrame
            Data-frame with the classification target variable and the
            explanatory features to be included in the model.

        seed: int
            The RandomState seed for the experiment.

        target: string
            The col name of the species presence vs. absence target.

        exp_name: string
            Name that encodes the experiment. A key for recovering the PyCaret
            workflow with pycaret.get_logs(exp_name).

        train_size: int, default = 0.7
            The ratio of the data to use for training (1-train_size used for
            hold out validation).

        normalize: bool, default = True
            Whether or not to perform standardization via the z-score method
            during the modeling experiment.

        metric: string, default = 'F1'
            The metric to use for model comparison, see PyCaret.setup() for
            available keys.

        fold: int, default = 10
            The number of CV k-folds for the stratified kfold strategy.

        silent: bool, default = False
            When set to True most of the output is hidden and the PyCaret
            workflow runs without user confirmation, designed for multiple runs
            between different seeds/data samples/data partitions (see the
            examples).

        Returns:
            The initialized PySDMs object.

        """

        # Init Variable Assignment
        self.data = data
        self.target = target
        self.seed = seed
        self.exp_name = exp_name
        self.train_size = train_size
        self.normalize = normalize
        self.metric = metric
        self.fold = fold
        self.silent = silent
        self.mod_list = ['et', 'catboost', 'rf', 'lightgbm', 'xgboost', 'gbc']

    def fit(self, pycaret_outdir='outputs/', deep_learning=False):

        """
        Model training with PyCaret pipelines. First constructs soft voters via
        exhaustive best subset selection among the above BRTs (below mod_list).
        The function then adds an assortment of neural networks, at two, three,
        and four hidden layers. The final model is selected from the best
        validation (F1 score, 30% hold out  default) set and saves the .pkl to
        file alongside the scores.

        pycaret_outdir: string, default = 'outputs/'
            The directory location for outputs.

        deep_learning: bool, default = False
            Whether or not to construct neural nets, recommended to also set
            normalization to True

        Returns:
            The voter with the best validation metric performance.

        Example
        -------
        >>> import PySDMS
        >>> x_vigilis = PySDMs(...) (see __init___)
        >>> learner = x_vigilis.fit()

        """

        # ####################################################################
        # PyCaret Model Fitting with BRTs and soft voters

        # Setup PyCaret environment
        self.exp_clf = pycaret.setup(self.data, target=self.target,
            session_id=self.seed, experiment_name=self.exp_name,
            normalize=self.normalize, train_size=self.train_size,
            fold=self.fold, log_experiment=True, verbose=False,
            fold_strategy='stratifiedkfold', silent=self.silent)

        # K-fold cross validation training with PyCaret "compare_models"
        self.classifier_list = pycaret.compare_models(n_select=len(
            self.mod_list), verbose=False, include=self.mod_list,
            sort=self.metric)
        self.pycaret_training_df = pycaret.pull()
        self.mod_keys = self.pycaret_training_df.index.tolist()

        # Build soft voters with best subset selection among the BRTs
        self.scoring_pandas = pd.DataFrame(np.zeros(7).reshape(-1, 7),
            columns=['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa',
            'MCC'])
        self.soft_voting_models = list()
        for j in np.arange(0, len(self.mod_list)):
            for i in np.arange(j+2, len(self.mod_list)+1):
                self.soft_voter = pycaret.blend_models(
                    estimator_list=self.classifier_list[j:i], method='soft',
                    verbose=False)
                self.soft_voting_models.append(self.soft_voter)
                pull_temp = pycaret.pull()
                self.scoring_pandas = self.scoring_pandas.append(
                    pull_temp.loc['Mean'])
        self.scoring_pandas.drop(self.scoring_pandas.index[0], inplace=True)
        self.scoring_pandas.index = np.arange(0, len(self.scoring_pandas[
            'Accuracy']))
        self.soft_voter = self.soft_voting_models[self.scoring_pandas[
            'F1'].idxmax()]
                # Add the various keys and models to the class vars
        self.classifier_list.append(self.soft_voter)
        self.mod_keys.append('soft_voter')

        # ####################################################################
        # PyCaret Model Fitting with neural nets

        # Add deep learners if deep_learning (also set normalization to True)
        if deep_learning:
            MLP_2 = internal.tunable.TunableMLPClassifier(hidden_layer_sizes=(300, 150),
                activation='relu', solver='adam', random_state=self.seed)
            self.MLP_2 = pycaret.create_model(MLP_2, verbose=False)
            MLP_3 = internal.tunable.TunableMLPClassifier(hidden_layer_sizes=(300, 150,
                100), activation='relu', solver='adam', random_state=self.seed)
            self.MLP_3 = pycaret.create_model(MLP_3, verbose=False)
            MLP_4 = internal.tunable.TunableMLPClassifier(hidden_layer_sizes=(450, 300, 150,
                100), activation='relu', solver='adam', random_state=self.seed)
            self.MLP_4 = pycaret.create_model(MLP_4, verbose=False)

            # Add another soft voter with BRT + deep learners
            self.soft_voter_dl = pycaret.blend_models(estimator_list=[
                self.classifier_list[0], self.MLP_2, self.MLP_3, self.MLP_4],
                method='soft', verbose=False)

            self.classifier_list.append(self.MLP_2)
            self.classifier_list.append(self.MLP_3)
            self.classifier_list.append(self.MLP_4)
            self.classifier_list.append(self.soft_voter_dl)
            self.mod_keys.append('MLP_2')
            self.mod_keys.append('MLP_3')
            self.mod_keys.append('MLP_4')
            self.mod_keys.append('soft_voter_dl')

        # Print a markdown title
        display(Markdown('--- \n ### PyCaret Model Fitting: \n --- ' ))

        # ####################################################################
        # PyCaret Model Evaluation, with cross val and hold-out

        # Evaluate the models with k-fold cv and held-out partitions
        self.cv_mean, self.names, self.ho_mean = list(), list(), list()

        print('The RandomState seed for this run is ' + str(self.seed) + '\n')
        for i in np.arange(0, len(self.classifier_list)):
            # first cv:
            pycaret.create_model(self.classifier_list[i], verbose=False)
            scorei = pycaret.pull()
            self.cv_mean.append(scorei[self.metric].iloc[0:self.fold])
            self.names.append(self.mod_keys[i])
            # then validation:
            pycaret.predict_model(self.classifier_list[i], verbose=False)
            scorei = pycaret.pull()
            self.ho_mean.append(scorei[self.metric].loc[0])
            if not self.silent:
                print('    >%s: CV (mean/sd): %.4f (%.4f) | Hold-out: %.4f' %
                    (self.names[i], np.mean(self.cv_mean[i]),
                    np.std(self.cv_mean[i]), self.ho_mean[i]))

        # Write the held out validation metrics to file:
        pd.DataFrame(self.ho_mean).to_csv(pycaret_outdir+'holdout_'+str(
            self.seed)+'.csv')

        # Grab model with best validation performance
        self.ho_mean_np = np.array(self.ho_mean)
        self.id_argmax = self.ho_mean_np.argmax(axis=0)
        self.model_key = self.mod_keys[self.id_argmax]

        # Create the final key with the `self.metric` validated model
        self.final_model_key = self.model_key
        if self.model_key == 'soft_voter':
            self.final_model_key = self.soft_voter
        if self.model_key == 'soft_voter_dl':
            self.final_model_key = self.soft_voter_dl
        if self.model_key == 'MLP_4':
            self.final_model_key = self.MLP_4
        if self.model_key == 'MLP_2':
            self.final_model_key = self.MLP_2
        if self.model_key == 'MLP_3':
            self.final_model_key = self.MLP_3

        # ####################################################################
        # Finalize

        # Refit model with 100% of data
        self.final_voter = pycaret.finalize_model(self.final_model_key)
        with open(pycaret_outdir+self.exp_name+'.pkl', 'wb') as f:
            pk.dump(self.final_voter, f)
        with open(pycaret_outdir+self.exp_name+'.pkl', 'rb') as f:
            self.voter_pkl = pk.load(f)
        display(Markdown('\n### The Final Model:'))
        print('     >', self.model_key)
        if not self.silent:
            print('     >', self.voter_pkl)

        # Return the final classifier to end the function
        return(self.final_voter)

    def interpolate(self, asc_input_dir, df_input_dir, img_output_dir, seed):

        """
        Geo-classification function for model interpolation to raster
        surfaces of the feature variables. Outputs both a probabilistic and
        binary prediction. See PyImpute's README.md for a very basic intro
        to geo-classification in Python.

        asc_input_dir, string
            The directory containing your raster surfaces, called in a manner
            such that glob can grab all of your training features. For
            example, if your files are saved as bclim1.asc, bclim2.asc, etc.,
            then you would include bclim*.asc at the end of the string (e.g.,
            'data/train-rasters-2.5m/bclim*.asc'').

        df_input_dir, string
            The input directory for the training dataframe, as called in
            self.fit() above for the data variable. Should include all of the
            directory path except for the random seed (end of the name) and
            the .csv extension (e.g., 'data/env_train/env_train_xv_').

        img_output_dir
            The output directory for the geo-classification results.

        seed, int
            The randomstate seed of the modeling run.

        """

        display(Markdown("### Geo-classification Interpolation \n ---"))
        # ####################################################################
        # Geo-classification Data IO

        # Raster features (e.g., BioClim).
        # Make sure the normalization is consistent with the model fitting.
        raster_features = sorted(glob.glob(asc_input_dir))

        # Geo-classification setup
        print('Training features / target shape:\n')
        train_xs = np.array(pd.read_csv(df_input_dir + str(seed)
            + '.csv').iloc[:,1:13])
        train_y = np.array(pd.read_csv(df_input_dir + str(seed)
            + '.csv').iloc[:,0])
        target_xs, raster_info = load_targets(raster_features)
        print('     >', train_xs.shape, train_y.shape)

        # ####################################################################
        # Geospatial Prediction

        # Grab the classifier for the given seed
        with open('outputs/xv_' + str(seed) + '.pkl', 'rb') as f:
            classifier = pk.load(f)
        classifier.fit(train_xs, train_y)
        # PyImpute geo-spatial classification step
        impute(target_xs, classifier, raster_info, outdir=img_output_dir)
        # Plot the output image
        interpolation = rasterio.open(img_output_dir + 'probability_1.tif')
        # Image plotter function
        def plotit(x, title, cmap="Greens"):
            plt.rcParams["figure.figsize"] = (8.2, 5.5)
            pylab.imshow(x, cmap=cmap, interpolation='nearest')
            pylab.colorbar()
            pylab.title(title, fontweight = 'bold')
            plt.show()
        plotit(interpolation .read(1),'Night Lizard Interpolation (prob.)')


    def validation_visuals(self, F1_min_seed, F1_max_seed, AUC_seed,
        pycaret_outdir='outputs/'):

        """
        A function that generates a validation-set F1 score boxplot and a AUC
        ROC analysis plot with CV. The function was developed for visualizing
        performance across multiple runs between consecutive seed ints.

        F1_min_seed/F1_max_seed: int
            The min and max seed model runs to grab for the F1 boxplot.

        AUC_seed: int
            The seed model run to perform CV ROC analysis.

        """

        display(Markdown('\n### Model Performance Visualizations: \n ---'))

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

        display(Markdown("F1 validation-set (30% held out) scores across" +
            " a dozen run's most predictive models:"))
        style.use('ggplot')
        plt.rcParams["figure.figsize"] = (2.25, 5.25)
        plt.boxplot(validation_scores_individual, 'k+', 'k^',
            medianprops = dict(color='black'), labels=['final_voter'])
        plt.title('Validation-Set F1 Scores')
        plt.ylabel('F1 score')
        y = validation_scores_individual
        x = np.random.normal(1, 0.030, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.35, markersize=11.5)
        plt.savefig(pycaret_outdir + 'F1_score.png', dpi = 400)
        plt.show()

        # ####################################################################
        # AUC K-fold Cross Validation ROC Plot for one run

        display(Markdown("AUC scores for a single run, with k-fold CV" +
            " across 100% of the data:"))
        style.use('default')

        # Data IO
        X = self.data
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
