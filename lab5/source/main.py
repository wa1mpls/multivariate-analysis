import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# Load iris dataset
iris = load_iris(as_frame=True)
X = iris.data  # Features
y = iris.target  # Target



# Calculate correlation matrix
corr_coeff = X.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_coeff, cmap='coolwarm', annot=True, linewidths=1, vmin=-1)
plt.title('Correlation Matrix Heatmap')
plt.savefig('correlation_matrix_heatmap.png')  # Save the plot as an image
plt.show()

# Split the dataset into two data frames: sepal-related features and petal-related features
X1 = X.iloc[:, :2]  # Extract first two columns to create a sepal-related features dataset
X2 = X.iloc[:, 2:]  # Extract last two columns to create a petal-related features dataset

# Standardize the data
scaler = StandardScaler()
X1_sc = scaler.fit_transform(X1)
X2_sc = scaler.fit_transform(X2)

# Print scaled data
#print("Scaled data for sepal-related features:")
#print(X1_sc)
#print("\nScaled data for petal-related features:")
#print(X2_sc)

# Choose number of canonical variates pairs
n_comp = 2

# Define CCA
cca = CCA(scale=False, n_components=n_comp)

# Fit our scaled data
cca.fit(X1_sc, X2_sc)

# Transform our datasets to obtain canonical variates
X1_c, X2_c = cca.transform(X1_sc, X2_sc)

# Compute canonical correlation coefficients
comp_corr = [np.corrcoef(X1_c[:, i], X2_c[:, i])[1][0] for i in range(n_comp)]

# Plot canonical correlation coefficients
plt.figure()
plt.bar(['CC1', 'CC2'], comp_corr, color='lightgrey', width=0.8, edgecolor='k')
plt.xlabel('Canonical Correlation Components')
plt.ylabel('Correlation Coefficient')
plt.title('Canonical Correlation Coefficients')
plt.savefig('canonical_correlation_coefficients.png')  # Save the plot as an image
plt.show()

# Print loadings for canonical variate of X1 dataset
print("Loadings for canonical variate of X1 dataset:")
print(cca.x_loadings_)

# Print loadings for canonical variate of X2 dataset
print("\nLoadings for canonical variate of X2 dataset:")
print(cca.y_loadings_)

# Create coefficient DataFrame
coef_df = pd.DataFrame(np.round(cca.coef_, 2), columns=X2.columns)
coef_df.index = X1.columns
print("\nCoefficient DataFrame:")
print(coef_df)

# Plot heatmap for coefficient DataFrame
plt.subplot(1, 2, 2)
sns.heatmap(coef_df, cmap='coolwarm', annot=True, linewidths=1, vmin=-1)
plt.title('Coefficient Heatmap')

plt.tight_layout()
plt.savefig('heatmap_and_coefficient_heatmap.png')  # Save the plot as an image
plt.show()