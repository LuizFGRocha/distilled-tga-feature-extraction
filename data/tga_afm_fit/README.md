Dataset with paired TGA data and fit parameters of AFM observations for regression.

When loaded with `numpy.load()` returns a data object with 3 keys:

- samples: Numpy string array with the name of the samples (33 rows)

- X: Numpy array containing the TGA data (33 x 4 x 1024)
    - Dim 1: Sample
    - Dim 2: Axis (Temperature(T), Weight(W), dW/dT, d²W/d²T)
    - Dim 3: Curve data points

- Y: Numpy array containing the statistics for the observed flakes characteristics extracted from AFM (33 x 5 x 2)
    - Dim 1: Sample
    - Dim 2: Characteristic (Min Ferret, Max Ferret, Height, Area, Volume)
    - Dim 3: Parameter (Mean, Standard Deviation)