# Module: PySDMs/internal
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Last modified : 4/10/21
# https://github.com/daniel-furman/PySDMs

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import rasterio
from pyimpute import load_targets, impute
import glob
import pickle as pk
import pylab

def interpolate(asc_input_dir, df_input_dir, img_output_dir, seed):

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
          classifier=pk.load(f)
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
          pylab.title(title, fontweight='bold')
          plt.show()

      return(plotit(interpolation .read(1),'Night Lizard Interpolation (prob.)'))