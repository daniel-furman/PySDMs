# Module: PySDMs/internal
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Last modified : 8/9/21
# https://github.com/daniel-furman/PySDMs

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import rasterio
from pyimpute import load_targets, impute
import glob
import pickle as pk
import pylab

def internal_interpolate(asc_input_dir, df_input_dir, img_output_dir, seed,
                         species_name):

      """Geo-classification for model interpolation to
        raster feature surfaces. Saves to file probabilistic and binary
        predictions.

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
      
      raster_features = sorted(glob.glob(asc_input_dir))
      print('Training features / target shape:\n')
      df = pd.read_csv(df_input_dir + str(seed) + '.csv')
      train_xs = np.array(df.iloc[:,1:len(df)])
      train_y = np.array(df.iloc[:,0])
      target_xs, raster_info = load_targets(raster_features)
      print('     >', train_xs.shape, train_y.shape)

      # ####################################################################
      # Geospatial Prediction

      with open('test/outputs/' + species_name + '_' + str(seed) + '.pkl', 'rb') as f:
          classifier=pk.load(f)
      classifier.fit(train_xs, train_y)
      impute(target_xs, classifier, raster_info, outdir=img_output_dir)
      interpolation = rasterio.open(img_output_dir + 'probability_1.tif')
      response = rasterio.open(img_output_dir + 'responses.tif')

      # Image plotter function
      def plotit(x, title, cmap="Greens"):
          plt.figure()
          plt.rcParams["figure.figsize"] = (8.2, 5.5)
          pylab.imshow(x, cmap=cmap, interpolation='nearest')
          pylab.colorbar()
          pylab.title(title, fontweight='bold')
          plt.show()

      return(plotit(interpolation.read(1),'Near-Current Interpolation'),
        plotit(response.read(1),'Near-Current Interpolation'))
