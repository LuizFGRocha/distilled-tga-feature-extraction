Dataset with complete TGA data of all samples for feature extraction.

When loaded with `numpy.load()` returns a data object with 2 keys:

- samples: Numpy string array with the name of the samples (113 rows)
- TGA: Numpy array containing the TGA data (113 x 4 x 1024)
    - Dim 1: Sample
    - Dim 2: Axis (Temperature(T), Weight(W), dW/dT, d²W/d²T)
    - Dim 3: Curve data points