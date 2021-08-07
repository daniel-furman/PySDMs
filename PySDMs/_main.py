# Module: PySDMs
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Last modified : 4/10/21
# https://github.com/daniel-furman/PySDMs

from IPython.display import display, Markdown
from IPython import get_ipython
import re
import os

# Markdown setup
def md_formatter(md, pp, cycle):
    pp.text(md.data)
text_plain = get_ipython().display_formatter.formatters['text/plain']
text_plain.for_type(Markdown, md_formatter)

#from PySDMs.internal.interpolate import interpolate as internal_interpolate
exec(open('/Users/danielfurman/Desktop/work/projects/DAT:Artathon/code/PySDMs-edit/PySDMs/internal/interpolate.py').read())
#from PySDMs.internal.validation_visuals import validation_visuals as internal_validation_visuals
exec(open('/Users/danielfurman/Desktop/work/projects/DAT:Artathon/code/PySDMs-edit/PySDMs/internal/validation_visuals.py').read())
#from PySDMs.internal.fit import fit as internal_fit
exec(open('/Users/danielfurman/Desktop/work/projects/DAT:Artathon/code/PySDMs-edit/PySDMs/internal/fit.py').read())


class PySDMs(object):

    """An object-oriented class for Species Distribution Modeling (SDM).
    PySDMs does most of its heavy lifting in the modeling portion of the
    SDM framework, with the interpolate functions mainly packaging
    the geo-classification step in an object-oriented manner. The pre-processing
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
        PySDMs workflows with multiple runs (see examples)."""

    def __init__(self, data, test_data, seed, target, exp_name,
        normalize=True, metric='F1', fold=10, silent=False,
        mod_list = ['et', 'catboost', 'rf', 'lightgbm', 'xgboost', 'gbc']):

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
        self.output_dir = 'outputs/'

    def fit(self, deep_learning=False, soft_voters=False, tuning=True):

        """Model training with PyCaret pipelines. Constructs soft voters via
        exhaustive best subset selection among the above BRTs (below mod_list).
        The function then adds an assortment of neural networks, at two, three,
        and four hidden layers. The final model is selected from the best
        validation (F1 score, 30% hold out  default) set and saves the .pkl to
        file alongside the scores.

        Local fit vars are relatively straightforward True/False that determine
        which types of models are tried.

        Returns:
            The voter with the best validation metric performance."""

        # Print a markdown title
        display(Markdown('--- \n ### PyCaret Model Fitting: \n --- ' ))
        return(internal_fit(self.data, self.test_data, self.seed, self.target,
            self.exp_name, self.normalize, self.metric, self.fold,
            self.silent, self.mod_list, self.output_dir, deep_learning,
            soft_voters, tuning))

    def interpolate(self, asc_input_dir, df_input_dir, img_output_dir):

        """Geo-classification function for model interpolation to raster
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
            The randomstate seed of the modeling run."""

        display(Markdown("--- \n ### Geo-classification Interpolation: \n ---"))
        return(internal_interpolate(asc_input_dir, df_input_dir,
            img_output_dir, self.seed, self.species_name))

    def validation_visuals(self, min_seed, max_seed):

        """A function that generates a validation-set F1 boxplot and a AUC
        ROC analysis plot with CV. The function was developed for visualizing
        performance across multiple runs between consecutive seed ints.

        min_seed/max_seed: int
            The min and max seed model runs to grab for the F1 boxplot.

        AUC_seed: int
            The seed model run to perform CV ROC analysis."""

        display(Markdown('--- \n ### Model Performance Plots: \n ---'))
        return(internal_validation_visuals(min_seed, max_seed, self.output_dir, self.species_name))
