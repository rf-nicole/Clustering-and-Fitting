# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:11:13 2025

@author: ropaf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from errors import error_prop, deriv, covar_to_corr

def read_and_clean_data(filepath, missing_threshold=0.5):
    """
    Function will clean and read the data in WorldBank format
    Copies the data to use for clustering

    Parameters:
    - filepath (str): Path to the data file
    - missing_threshold (float): Max fraction of allowed missing values per row

    Returns:
    - original_df (pd.DataFrame): Original cleaned data
    - cluster_df (pd.DataFrame): Copy for clustering (not normalized)
    """
    
    
    df = pd.read_csv(filepath)
    
    # Drop unnecessary columns that wont be used
    df = df.drop(columns=['Country Code', 'Indicator Code'])

    # some rows have too many missing values so drop these
    df = df.dropna(thresh=int((1 - missing_threshold) * len(df.columns)))

    # Reset index for safety
    df = df.reset_index(drop=True)

    # Copy for clustering
    cluster_df = df.copy()

    return df, cluster_df

forest_area, _ = read_and_clean_data('forest_land_percentage.csv')
agri_land, _ = read_and_clean_data('agriculture_land_percentage.csv')
urban_pop, _ = read_and_clean_data('urban_population_growth.csv')

#combining all three datasets into one
combined_df = pd.concat([forest_area, agri_land, urban_pop], ignore_index=True)
# Indicators and years i'm working with
indicators = [
    'Forest area (% of land area)',
    'Agricultural land (% of land area)',
    'Urban population growth (annual %)'
]

years = [1980, 2000, 2020]

#filtering the forest area dataframe
filtered_df = combined_df[combined_df['Indicator Name'].isin(indicators)]

# Melt to long format for easy filtering
long_df = filtered_df.melt(
    id_vars=['Country Name', 'Indicator Name'],
    var_name='Year',
    value_name='Value'
)

#cleaning the columns
long_df['Year'] = long_df['Year'].astype(str)
long_df = long_df[long_df['Year'].isin(map(str, years))]

#pivot so each row = country, columns = indicator_year
pivot_df = long_df.pivot_table(
    index='Country Name',
    columns=['Indicator Name', 'Year'],
    values='Value'
)

#flatten columns
pivot_df.columns = [f"{ind}_{year}" for ind, year in pivot_df.columns]

#drop rows with too many NaNs
pivot_df = pivot_df.dropna()

from sklearn.preprocessing import StandardScaler
#saving the original plot for comparison
original_data = pivot_df.copy()
#normalising the data
scaler = StandardScaler()
normalized_data = pd.DataFrame(
    scaler.fit_transform(pivot_df),
    index=pivot_df.index,
    columns=pivot_df.columns
)

#doing the K-means Clustering
from sklearn.cluster import KMeans

#starting with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(normalized_data)

original_data['Cluster'] = cluster_labels
# Mapping cluster numbers to descriptive labels
cluster_labels_map = {
    0: 'Agricultural & Moderately Urbanising',
    1: 'Forested & Low Urban Growth',
    2: 'Low Forest & Slowing Urban Growth'
}

# Add a column with these human-readable labels
original_data["Cluster Label"] = original_data['Cluster'].map(cluster_labels_map).fillna('Unknown')

# Reset index just in case country names are in the index
original_data = original_data.reset_index()

#grouping by the cluster label and print 2 countries from each as an example
for label, group in original_data.groupby('Cluster Label'):
    sample = group.sample(n=2, random_state=42)
    print(f"\nCluster: {label}")
    for _, row in sample.iterrows():
        print(f" - {row['Country Name']}")

#adding cluster labels to original data
print(np.unique(cluster_labels))

from sklearn.decomposition import PCA
#reduce to 2D for plotting
pca = PCA(n_components=2)
pca_components = pca.fit_transform(normalized_data)
pca_cluster_centers = pca.transform(kmeans.cluster_centers_)

#plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_components[:, 0], 
                y=pca_components[:, 1],
                hue=original_data['Cluster Label'], palette='Set2', s=100)

plt.scatter(
    pca_cluster_centers[:,0],
    pca_cluster_centers[:,1],
    c='black',
    s=100,
    marker='X',
    label='Cluster Centers'
)
plt.title('Country Clusters + Centers based on Land Use & Urban Growth)')
plt.xlabel('PCA 1 (Land Use & Urbanization Pattern)')
plt.ylabel('PCA 2')
plt.legend(title='Cluster Description', loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()


#showing back-transformed cluster centers
original_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_means = original_data.groupby('Cluster').mean(numeric_only=True)
pd.set_option('display.max_columns', None)  # So it doesn't cut off
print(cluster_means)

cluster_centers_df = pd.DataFrame(
    original_centers,
    columns=pivot_df.columns
)
cluster_centers_df['Cluster Label'] = [cluster_labels_map[i] for i in range(kmeans.n_clusters)]

print("\nCluster Centers in Original Scale:\n")
print(cluster_centers_df.round(2))

# Show mean of each indicator by Cluster Label (original, non-normalized data)
cluster_summary = original_data.groupby('Cluster Label').mean(numeric_only=True)

# Round for readability
cluster_summary = cluster_summary.round(2)

# Print the summary
pd.set_option('display.max_columns', None)
print(cluster_summary)
# Transpose so indicators are on x-axis
transposed = cluster_summary.T


#fitting with an exponential model
def exp_model(x, a, b):
    return a * np.exp(b * x)

#for a list of countries in each cluster
countries = ['Indonesia','Israel', 'United Kingdom']
indicator = 'Urban population growth (annual %)'

#Setup plot
plt.figure(figsize=(14, 8))

# Color palette for visual distinction
colors = ['red', 'green', 'blue']

#loop through each country of choice
for i, country in enumerate(countries):
    #filter data for the country and indicator
    data = long_df[
        (long_df['Country Name'] == country) &
        (long_df['Indicator Name'] == indicator)
    ].sort_values('Year')
    
   
    x = data['Year'].astype(int).values
    y = data['Value'].values
    x_rel = x - x[0] if len(x) > 0 else []

    if len(x) >= 3 and not np.isnan(y).any():
        popt, pcov = curve_fit(exp_model, x_rel, y, maxfev=10000)
    else:
        print(f"Insufficient or invalid data for {country}")
        continue
    
    
    #predict over a future range by making an array for forecasting
    x_future = np.linspace(0, 60, 200)  # e.g., 1980 to 2040
    forecast = exp_model(x_future, *popt)
    if np.all(np.isfinite(pcov)):
       sigma = error_prop(x_future, exp_model, popt, pcov)
       lower_forecast = forecast - sigma
       upper_forecast = forecast + sigma
    else:
       print(f"⚠️ Skipping confidence range for {country} (bad covariance matrix)")
       lower_forecast = upper_forecast = forecast  # No range shown


    #plot the observed data
    plt.plot(x,y,'o', label=f"{country} Data", color=colors[i])
    #plot best-fit curve
    plt.plot(x_future + x[0], forecast, '-', color=colors[i], label=f"{country} Fit")
    #plot confidence range
    plt.fill_between(x_future + x[0], lower_forecast, upper_forecast, color=colors[i], alpha=0.2)

#final plot formatting
plt.xlabel('Year')
plt.ylabel('Urban Population Growth (%)')
plt.title('Urban Population Growth with Exponential Fit and Confidence Ranges')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
