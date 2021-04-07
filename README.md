## RecFeatureSelect (Recursive Feature Selection). 

Feature selection via recursive removal of the most correlated pair. The feature importance scores are used as the rankings, deciding which variable to drop at each call.

* The main function can be found from the source folder, RecFeatureSelect.
* The input data consists of the original covariance matrix, the feature importance scores, a spearman correlation threshold, and the raw data. 
* After the run the function will save the final covariance matrix to file as "cov.csv". All correlations will be less than the input threshold.  

### Package Layout

* [LICENSE](https://github.com/daniel-furman/RecFeatureSelect/blob/main/LICENSE) - the MIT license, which applies to this package
* README.md - the README file, which you are now reading
* [requirements.txt](https://github.com/daniel-furman/RecFeatureSelect/blob/main/requirements.txt) - prerequisites to install this package, used by pip
* [setup.py](https://github.com/daniel-furman/RecFeatureSelect/blob/main/setup.py) - installer script
* [RecFeatureSelect](https://github.com/daniel-furman/RecFeatureSelect/tree/main/RecFeatureSelect)/ - the library code itself
* [tests](https://github.com/daniel-furman/RecFeatureSelect/tree/main/test)/ - unit tests

---

Example of runtime interface, with the recursive calls and the final saved covariance matrix (all beneath the threshold):
Comparisons during run | Final covariance matrix
:---------------------------------:|:----------------------------------------:
![](data/img1.png) | ![](data/img2.png)

### Longer Description:

This function selects de-correlated features for a modeling experiment by filtering the most similar pair at each call. The algorithm reaches the
stopping case when all pairs of features are below the Spearman's statistic `threshold`. The feature importances are used as the ranking.

* covariance: Pandas object containing the covariance matrix, with
        correlations between modeling variables, by definition containing
        ones along the diagonal. Variable names should be above the
        entries and absent from the rows.

* feature_importance: Pandas object containing a model's feature importance
        scores in the first row, with the same order of variables as the
        covariance matrix. Variable names should be above the row. Feature
        importance is generally defined as techniques that assign a score to
        input features based on how useful they are at predicting a target
        variable. Importance scores can vary, and you should therefore
        at least take a look at the associated uncertainties.

* threshold: A correlation value for which features are filtered below,
        Thresholds between 0.5 - 0.7 are commonly used (e.g. Dormann et al.,
        2013, doi: 10.1111/j.1600-0587.2012.07348.x).

* raw_data: The raw feature dataframe that constructed the covariance matrix.

### Warnings:

* The Pandas dataframes should have the same order of variables.
* Make sure dependencies are installed: pandas, np, scipy.
