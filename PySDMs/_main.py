# Module: PySDMs
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Last modified : 8/9/21
# https://github.com/daniel-furman/PySDMs

from IPython.display import display, Markdown
from IPython import get_ipython
import re
import os

from PySDMs.internal.interpolate import internal_interpolate as internal_interpolate
from PySDMs.internal.validation_visuals import internal_validation_visuals as internal_validation_visuals
from PySDMs.internal.fit import internal_fit as internal_fit

# Markdown setup for notebook environment
#def md_formatter(md, pp, cycle):
    #pp.text(md.data)
#text_plain = get_ipython().display_formatter.formatters['text/plain']
#text_plain.for_type(Markdown, md_formatter)

class PySDMs(object):

    """
    An object-oriented Python class for semi-auto ML geo-classification
    (running on PyCaret). Compares gradient boosted tree algorithms by
    default, with options to include soft voters and NNs. Designed for
    Species Distribution Modeling applications.

    Functions
    -------

    self.fit(): Classification training with PyCaret, considering tree-based
        methods, (CPU) neural nets, and two-model soft voters (exhaustive
        search). Requires a Pandas data-frame with a target and explanatory
        features. Returns the voter with the best validation
        metric performance (user-defined metric). See Pycaret.org for more
        and for customization purposes (classification module).

    self.interpolate(): Geo-classification for model interpolation to
        raster feature surfaces. Saves to file probabilistic and binary
        predictions.

    self.validation_performance(): Data vizualizations for the user-defined
        scoring metric (box+whisker) and ROC AUC visuals (10-fold stratified
        Cross Val) (see example notebooks in the Git Repo)."""

    def __init__(self, data, test_data, seed, target, exp_name,
        normalize=True, metric='F1', fold=10, silent=False,
        mod_list = ['et', 'catboost', 'rf', 'lightgbm', 'xgboost', 'gbc'],
        pycaret_outdir='test/outputs/'):

        """data: Pandas DataFrame
            Data-frame with the classification target variable and the
            explanatory features to be included in the model.

        test_data: Pandas DataFrame
            Data-frame with the classification target variable and the
            explanatory features to be used for the validation of
            the model.

        seed: int
            The RandomState seed for the experiment.

        target: string
            The col name of the species presence vs. absence target.

        exp_name: string
            Name that encodes the experiment. A key for recovering the PyCaret
            workflow with pycaret.get_logs(exp_name).

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

        mod_list: string list, default #{BRTs}
            See PyCaret for mod strings. The initial set of models to consider
            for best subset soft voter selection.

        Returns:
            The initialized PySDMs object."""

        # Init Variable Assignment
        self.data = data
        self.test_data = test_data
        self.target = target
        self.seed = seed
        self.exp_name = exp_name
        self.normalize = normalize
        self.metric = metric
        self.fold = fold
        self.silent = silent
        self.mod_list = mod_list
        self.species_name = re.split('[_]', exp_name)[0]

        DATA = os.path.dirname(os.path.abspath(__file__))
        self.output_dir =  [os.path.join(DATA, x)
            for x in [pycaret_outdir]][0]



    def fit(self, deep_learning=False, soft_voters=False, tuning=False):
        # ML classification fitting
        #display(Markdown('--- \n ### PyCaret Model Fitting: \n --- ' ))
        return(internal_fit(self.data, self.test_data, self.seed, self.target,
            self.exp_name, self.normalize, self.metric, self.fold,
            self.silent, self.mod_list, self.output_dir, deep_learning,
            soft_voters, tuning))

    def interpolate(self, asc_input_dir, df_input_dir, img_output_dir):
        # Geospatial step for raster predictions
        #display(Markdown("--- \n ### Geo-classification Interpolation: \n ---"))
        return(internal_interpolate(asc_input_dir, df_input_dir,
            img_output_dir, self.seed, self.species_name))

    def validation_visuals(self, min_seed, max_seed, pycaret_outdir):
        # Dataviz for model fitting
        #display(Markdown('--- \n ### Model Performance Plots: \n ---'))
        return(internal_validation_visuals(min_seed, max_seed, self.output_dir, self.species_name))
