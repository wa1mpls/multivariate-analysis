# Import required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

# Load the CSV file into a DataFrame
df = pd.read_csv("bfi.csv")

# Display column names
print(df.columns)

# Dropping unnecessary columns
df.drop(['gender', 'education', 'age'], axis=1, inplace=True)

# Dropping rows with missing values
df.dropna(inplace=True)

# Calculate Bartlett's test of sphericity
chi_square_value, p_value = calculate_bartlett_sphericity(df)

# Print chi-square value and p-value
print("Chi-square value:", chi_square_value)
print("P-value:", p_value)

# Calculate Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy
kmo_all, kmo_model = calculate_kmo(df)

# Print KMO model value
print("KMO model value:", kmo_model)

# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer(n_factors=25, rotation=None)
fa.fit(df)

# Check Eigenvalues
ev, v = fa.get_eigenvalues()

# Create scree plot
plt.scatter(range(1, df.shape[1] + 1), ev)
plt.plot(range(1, df.shape[1] + 1), ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()

# Save the plot as an image file
plt.savefig('scree_plot.png')

# Display the plot
plt.show()

# Create factor analysis object and perform factor analysis with 5 factors
fa_5 = FactorAnalyzer(n_factors=5, rotation="varimax")
fa_5.fit(df)

# Check Eigenvalues for 5 factors
ev_5, v_5 = fa_5.get_eigenvalues()

# Create scree plot for 5 factors
plt.scatter(range(1, df.shape[1] + 1), ev_5)
plt.plot(range(1, df.shape[1] + 1), ev_5)
plt.title('Scree Plot (5 Factors)')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()

# Save the plot as an image file for 5 factors
plt.savefig('scree_plot_5_factors.png')

# Display the plot for 5 factors
plt.show()

# Loadings for 5 factors
fa_5.loadings_

# Get variance of each factor for 5 factors
fa_5.get_factor_variance()

# Loadings for 5 factors
loadings_df = pd.DataFrame(fa_5.loadings_, columns=['Factor'+str(i+1) for i in range(5)], index=df.columns)
print("Loadings for 5 factors:")
print(loadings_df)

# Check Eigenvalues for 5 factors
eigenvalues_df = pd.DataFrame(ev_5[:5], columns=['Eigenvalues'], index=['Factor'+str(i+1) for i in range(5)])
print("\nEigenvalues for 5 factors:")
print(eigenvalues_df)

# Get variance of each factor for 5 factors
factor_variance_df = pd.DataFrame({
    'Variance Explained': fa_5.get_factor_variance()[0][:5],
    'Proportion of Variance': fa_5.get_factor_variance()[1][:5],
    'Cumulative Proportion': np.cumsum(fa_5.get_factor_variance()[1][:5])
}, index=['Factor'+str(i+1) for i in range(5)])
print("\nVariance of each factor for 5 factors:")
print(factor_variance_df)




