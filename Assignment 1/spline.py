import numpy as np
import pandas as pd

# Generate example data
np.random.seed(0)
data = pd.DataFrame({
    'A': np.random.rand(10),
    'B': np.random.rand(10),
    'C': np.random.randint(1, 4, size=10)
})
data.loc[[1, 4, 7], 'A'] = np.nan  # Introduce missing values
data.loc[[2, 5], 'B'] = np.nan
data.loc[[0, 3, 8], 'C'] = np.nan

print("Original Data:")
print(data)

# Apply Spline Interpolation
spline_imputed = data.copy()
spline_imputed['A'] = spline_imputed['A'].interpolate(method='spline', order=2)
spline_imputed['B'] = spline_imputed['B'].interpolate(method='spline', order=2)
spline_imputed['C'] = spline_imputed['C'].interpolate(method='spline', order=2)

# Handle any remaining NaNs (e.g., at boundaries)
spline_imputed['A'] = spline_imputed['A'].fillna(method='bfill').fillna(method='ffill')
spline_imputed['B'] = spline_imputed['B'].fillna(method='bfill').fillna(method='ffill')
spline_imputed['C'] = spline_imputed['C'].fillna(method='bfill').fillna(method='ffill')

print("\nSpline Interpolation:")
print(spline_imputed)
