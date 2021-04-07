## PySDMs (Python Species Distribution Models). 

Feature selection via recursive removal of the most correlated pair. The feature importance scores are used as the rankings, deciding which variable to drop at each call.

* The main function can be found from the source folder, RecFeatureSelect.
* The input data consists of the original covariance matrix, the feature importance scores, a spearman correlation threshold, and the raw data. 
* After the run the function will save the final covariance matrix to file as "cov.csv". All correlations will be less than the input threshold.  

### Package Layout

* [PySDMs](https://github.com/daniel-furman/PySDMs/tree/main/src/PySDMs)/ - the library code itself
* [LICENSE](https://github.com/daniel-furman/PySDMs/blob/main/LICENSE) - the MIT license, which applies to this package
* README.md - the README file, which you are now reading
* [requirements.txt](https://github.com/daniel-furman/PySDMs/blob/main/requirements.txt) - prerequisites to install this package, used by pip
* [setup.py](https://github.com/daniel-furman/PySDMs/blob/main/setup.py) - installer script
* [tests](https://github.com/daniel-furman/PySDMs/tree/main/test)/ - unit tests

---

Example of PySDMs output:

Modeling Metrics| Geo-classification of Species Distribution
:---------------------------------:|:----------------------------------------:
![](examples/night_lizards/data/auc.png) | ![](examples/night_lizards/data/range.png)

### Longer Description:
