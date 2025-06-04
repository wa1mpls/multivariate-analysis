# importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


# set NumPy options
np.set_printoptions(suppress=True)
pd.set_option('display.max_rows', 20)

# read CSV data with Pandas
data = pd.read_csv("wine.csv")

# setup data column
data.columns = ["V" + str(i) for i in range(1, len(data.columns) + 1)]
# independent variables data
X = data.loc[:, "V2":]
# dependent variable data
y = data.V1

print("## Data:")
print(data)

print("## Head:")
print(data.head())

print("## Tail:")
print(data.tail())

print("## Info:")
data.info()

# plotting multivariate data
pd.plotting.scatter_matrix(data.loc[:, "V2":"V14"], diagonal="kde", figsize=(20, 15))
plt.show()

for i in range(2, 14):
    sns.lmplot(x="V" + str(i), y="V" + str(i + 1), data=data, hue="V1", fit_reg=False)

ax = data[["V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14"]].plot(figsize=(20, 15))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax = data[["V2", "V3", "V4", "V5", "V6"]].plot(figsize=(20, 15))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax = data[["V7", "V8", "V9", "V10", "V11"]].plot(figsize=(20, 15))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax = data[["V12", "V13", "V14"]].plot(figsize=(20, 15))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax = data[["V12", "V13"]].plot(figsize=(20, 15))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# calculating summary statistics for multivariate data

print(X.apply(np.mean))
print(X.apply(np.std))
print(X.apply(np.max))
print(X.apply(np.min))


# means and variances per group
def print_mean_and_sd_by_group(variables, group_variable):
    data_group_by = variables.groupby(group_variable)

    print("## Means:")
    print(data_group_by.apply(np.mean))

    print("\n## Standard deviations:")
    print(data_group_by.apply(np.std))

    print("\n## Sample sizes:")
    print(pd.DataFrame(data_group_by.apply(len)))


print_mean_and_sd_by_group(X, y)


def calc_within_groups_variance(variable, group_variable):
    # find out how many values the group variable can take
    levels = sorted(set(group_variable))
    num_levels = len(levels)
    # get the mean and standard deviation for each group:
    num_total = 0
    denom_total = 0
    for level_i in levels:
        level_i_data = variable[group_variable == level_i]
        level_i_length = len(level_i_data)
        # get the standard deviation for group i:
        sdi = np.std(level_i_data)
        num_i = level_i_length * sdi ** 2
        denom_i = level_i_length
        num_total = num_total + num_i
        denom_total = denom_total + denom_i
    # calculate the within-groups variance
    v_w = num_total / (denom_total - num_levels)
    return v_w


print("## v_w:")
print(calc_within_groups_variance(X.V2, y))


# between-groups variance and within-groups variance for a variable
def calc_between_groups_variance(variable, group_variable):
    # find out how many values the group variable can take
    levels = sorted(set(group_variable))
    num_levels = len(levels)
    # calculate the overall grand mean:
    grand_mean = np.mean(variable)
    # get the mean and standard deviation for each group:
    num_total = 0
    denom_total = 0
    for level_i in levels:
        level_i_data = variable[group_variable == level_i]
        level_i_length = len(level_i_data)
        # get the mean and standard deviation for group i:
        mean_i = np.mean(level_i_data)
        sdi = np.std(level_i_data)
        num_i = level_i_length * ((mean_i - grand_mean) ** 2)
        denom_i = level_i_length
        num_total = num_total + num_i
        denom_total = denom_total + denom_i
    # calculate the between-groups variance
    v_b = num_total / (num_levels - 1)
    return v_b


print("## v_b:")
print(calc_between_groups_variance(X.V2, y))


def calc_separations(variables, group_variable):
    # calculate the separation for each variable
    for variable_name in variables:
        variable_i = variables[variable_name]
        v_w = calc_within_groups_variance(variable_i, group_variable)
        v_b = calc_between_groups_variance(variable_i, group_variable)
        sep = v_b / v_w
        print("variable", variable_name, "Vw=", v_w, "Vb=", v_b, "separation=", sep)


calc_separations(X, y)


# between-groups covariance and within-groups covariance for two variables
def calc_within_groups_covariance(variable1, variable2, group_variable):
    levels = sorted(set(group_variable))
    num_levels = len(levels)
    cov_w = 0.0
    # get the covariance of variable 1 and variable 2 for each group:
    for level_i in levels:
        level_i_data1 = variable1[group_variable == level_i]
        level_i_data2 = variable2[group_variable == level_i]
        mean1 = np.mean(level_i_data1)
        mean2 = np.mean(level_i_data2)
        level_i_length = len(level_i_data1)
        # get the covariance for this group:
        term1 = 0.0
        for level_i_data1j, level_i_data2j in zip(level_i_data1, level_i_data2):
            term1 += (level_i_data1j - mean1) * (level_i_data2j - mean2)
        cov_group_i = term1  # covariance for this group
        cov_w += cov_group_i
    total_length = len(variable1)
    cov_w /= total_length - num_levels
    return cov_w


print("## cov_w:")
print(calc_within_groups_covariance(X.V8, X.V11, y))


def calc_between_groups_covariance(variable1, variable2, group_variable):
    # find out how many values the group variable can take
    levels = sorted(set(group_variable))
    num_levels = len(levels)
    # calculate the grand means
    variable1mean = np.mean(variable1)
    variable2mean = np.mean(variable2)
    # calculate the between-groups covariance
    cov_b = 0.0
    for level_i in levels:
        level_i_data1 = variable1[group_variable == level_i]
        level_i_data2 = variable2[group_variable == level_i]
        mean1 = np.mean(level_i_data1)
        mean2 = np.mean(level_i_data2)
        level_i_length = len(level_i_data1)
        term1 = (mean1 - variable1mean) * (mean2 - variable2mean) * level_i_length
        cov_b += term1
    cov_b /= num_levels - 1
    return cov_b


print("## cov_b:")
print(calc_between_groups_covariance(X.V8, X.V11, y))

# calculating correlations for multivariate data
corr = stats.pearsonr(X.V2, X.V3)
print("p-value:\t", corr[1])
print("cor:\t\t", corr[0])

corr_mat = X.corr()
print(corr_mat)

plt.figure()
sns.heatmap(corr_mat, vmax=1., square=False).xaxis.tick_top()
plt.show()

# Hinton diagram
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))
    ax.patch.set_facecolor('lightgray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    for (x, y), w in np.ndenumerate(matrix):
        color = 'red' if w > 0 else 'blue'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    n_ticks = matrix.shape[0]
    ax.xaxis.tick_top()
    ax.set_xticks(range(n_ticks))
    ax.set_xticklabels(list(matrix.columns), rotation=90)
    ax.set_yticks(range(n_ticks))
    ax.set_yticklabels(matrix.columns)
    ax.grid(False)
    ax.autoscale_view()
    ax.invert_yaxis()


plt.figure()
hinton(corr_mat)
plt.show()


def most_highly_correlated(my_dataframe, num_to_report):
    # find the correlations
    cor_matrix = my_dataframe.corr()
    # set the correlations on the diagonal or lower triangle to zero,
    # so they will not be reported as the highest ones:
    cor_matrix *= np.tri(*cor_matrix.values.shape, k=-1).T
    # find the top n correlations
    cor_matrix = cor_matrix.stack()
    cor_matrix = cor_matrix.reindex(cor_matrix.abs().sort_values(ascending=False).index).reset_index()
    # assign human-friendly names
    cor_matrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
    return cor_matrix.head(num_to_report)


print(most_highly_correlated(X, 10))


# standardising variables
standardisedX = scale(X)
standardisedX = pd.DataFrame(standardisedX, index=X.index, columns=X.columns)

print(standardisedX.apply(np.mean))
print(standardisedX.apply(np.std))


# principal component analysis
pca = PCA().fit(standardisedX)


def pca_summary(pca, standardised_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(list(zip(a, b, c)), index=names, columns=columns)
    if out:
        print("Importance of components:")
        print(summary)
    return summary


summary = pca_summary(pca, standardisedX)
print(summary.sdev)
print(np.sum(summary.sdev**2))


# how many principal components to retain
def scree_plot(pca, standardised_values):
    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()


plt.figure()
scree_plot(pca, standardisedX)

print(summary.sdev**2)

# loadings for the principal components
print(pca.components_[0])
print(np.sum(pca.components_[0]**2))


def calc_pc(variables, loadings):
    # find the number of samples in the data set and the number of variables
    num_samples, num_variables = variables.shape
    # make a vector to store the component
    pc = np.zeros(num_samples)
    # calculate the value of the component for each sample
    for i in range(num_samples):
        value_i = 0
        for j in range(num_variables):
            value_ij = variables.iloc[i, j]
            loading_j = loadings[j]
            value_i = value_i + (value_ij * loading_j)
        pc[i] = value_i
    return pc


print(calc_pc(standardisedX, pca.components_[0]))
print(pca.transform(standardisedX)[:, 0])

print(pca.components_[1])
print(np.sum(pca.components_[1]**2))


# scatter plots of the principal components
def pca_scatter(pca, standardised_values, classifs):
    foo = pca.transform(standardised_values)
    bar = pd.DataFrame(list(zip(foo[:, 0], foo[:, 1], classifs)), columns=["PC1", "PC2", "Class"])
    sns.lmplot(x="PC1", y="PC2", data=bar, hue="Class", fit_reg=False)


pca_scatter(pca, standardisedX, y)
print_mean_and_sd_by_group(standardisedX, y)

for i in range(2, 14):
    sns.lmplot(x="V" + str(i), y="V" + str(i + 1), data=data,
hue="V1", fit_reg=False)
    
plt.show()
print_mean_and_sd_by_group(standardisedX, y)

# importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# set NumPy options
np.set_printoptions(suppress=True)
pd.set_option('display.max_rows', 20)

# read CSV data with Pandas
data = pd.read_csv("wine.csv")

# setup data column
data.columns = ["V" + str(i) for i in range(1, len(data.columns) + 1)]
# independent variables data
X = data.loc[:, "V2":]
# dependent variable data
y = data.V1

# plotting multivariate data
pd.plotting.scatter_matrix(data.loc[:, "V2":"V14"], diagonal="kde", figsize=(20, 15))
plt.show()

# Linear Discriminant Analysis (LDA)
lda = LinearDiscriminantAnalysis()
lda.fit(standardisedX, y)

# Transform the data
X_lda = lda.transform(standardisedX)

# Scatter plot of the LDA components
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=y, palette='viridis')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('Scatter plot of LDA components')
plt.legend(title='Class')
plt.show()

# Explained variance ratio
print("Explained variance ratio:", lda.explained_variance_ratio_)

# Coefficients of the linear discriminants
print("Coefficients of the linear discriminants:")
print(lda.scalings_)

# Group means
group_means = pd.DataFrame(lda.means_, columns=standardisedX.columns)
print("Group means:")
print(group_means)

# Group sizes
group_sizes = pd.DataFrame(lda.priors_, columns=['Group Size'], index=lda.classes_)
print("Group sizes:")
print(group_sizes)

# Within scatter matrix
print("Within scatter matrix:")
print(lda.covariance_)

# Between scatter matrix
print("Between scatter matrix:")
print(lda.explained_variance_ratio_)


# Calculate the separations using LDA components
def calc_lda_separations(lda_transformed_data, group_variable):
    lda_separations = {}
    for i in range(lda_transformed_data.shape[1]):
        lda_component = lda_transformed_data[:, i]
        v_w = calc_within_groups_variance(lda_component, group_variable)
        v_b = calc_between_groups_variance(lda_component, group_variable)
        lda_separations[f'LD{i+1}'] = v_b / v_w
    return lda_separations

lda_separations = calc_lda_separations(X_lda, y)
print("LDA Separations:")
print(lda_separations)
