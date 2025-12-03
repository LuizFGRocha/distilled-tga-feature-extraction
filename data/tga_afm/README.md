Dataset with paired TGA data and statistics of AFM observations for regression.

When loaded with `numpy.load()` returns a data object with 3 keys:

- samples: Numpy string array with the name of the samples (33 rows)

- X: Numpy array containing the TGA data (33 x 4 x 1024)
    - Dim 1: Sample
    - Dim 2: Axis (Temperature(T), Weight(W), dW/dT, d²W/d²T)
    - Dim 3: Curve data points

- Y: Numpy array containing the statistics for the observed flakes characteristics extracted from AFM (33 x 5 x 5)
    - Dim 1: Sample
    - Dim 2: Characteristic (Min Ferret, Max Ferret, Height, Area, Volume)
    - Dim 3: Statistic (Mean, Variance, Skewness, Kurtosis, Median)


PS: I designed the label dataset Y like this in order to simplify the access of determined sets of characteristics. Ex.:

- Training with the skewness of the min ferret as the labels: Y[:, 0, 2]
- Training with a vector containing all the height statistics as the labels: Y[:, 2, :].reshape((Y.shape[0], -1))