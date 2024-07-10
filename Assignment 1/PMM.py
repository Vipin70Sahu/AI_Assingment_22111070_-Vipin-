# Import necessary libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.imputation.mice import MICEData

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

# Predictive Mean Matching (PMM) using MICE
pmm_imputed = data.copy()
imp = MICEData(pmm_imputed)
imp.update_all()
pmm_imputed = imp.data
print("\nPredictive Mean Matching (PMM) using MICE:")
print(pmm_imputed)
