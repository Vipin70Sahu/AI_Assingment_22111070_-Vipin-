# data come
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

# Imputation Code

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

# Mode Imputation
mode_imputed = data.copy()

# Find the mode for each column that requires imputation
for column in mode_imputed.columns:
    mode = mode_imputed[column].mode()
    if not mode.empty:
        mode_imputed[column] = mode_imputed[column].fillna(mode[0])

print("\nMode Imputation:")
print(mode_imputed)
