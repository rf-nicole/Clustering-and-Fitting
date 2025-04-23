# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:11:13 2025

@author: ropaf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    df = df.drop(columns=["Country Code", "Indicator Code"])

    # some rows have too many missing values so drop these
    df = df.dropna(thresh=int((1 - missing_threshold) * len(df.columns)))

    # Reset index for safety
    df = df.reset_index(drop=True)

    # Copy for clustering
    cluster_df = df.copy()

    return df, cluster_df

forest_area, _ = read_and_clean_data("forest_land_percentage.csv")
agri_land, _ = read_and_clean_data("agriculture_land_percentage.csv")
urban_pop, _ = read_and_clean_data("urban_population_growth.csv")

#combining all three datasets into one
combined_df = pd.concat([forest_area, agri_land, urban_pop], ignore_index=True)
# Indicators and years i'm working with
indicators = [
    "Forest area (% of land area)",
    "Agricultural land (% of land area)",
    "Urban population growth (annual %)"
]

years = [1980, 2000, 2020]

#filtering the forest area dataframe
filtered_df = combined_df[combined_df["Indicator Name"].isin(indicators)]

# Melt to long format for easy filtering
long_df = filtered_df.melt(
    id_vars=["Country Name", "Indicator Name"],
    var_name="Year",
    value_name="Value"
)

#cleaning the columns
long_df["Year"] = long_df["Year"].astype(str)
long_df = long_df[long_df["Year"].isin(map(str, years))]

#pivot so each row = country, columns = indicator_year
pivot_df = long_df.pivot_table(
    index="Country Name",
    columns=["Indicator Name", "Year"],
    values="Value"
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

original_data["Cluster"] = cluster_labels
# Mapping cluster numbers to descriptive labels
cluster_labels_map = {
    0: "Agricultural & Moderately Urbanising",
    1: "Forested & Low Urban Growth",
    2: "Low Forest & Slowing Urban Growth"
}

# Add a column with these human-readable labels
original_data["Cluster Label"] = original_data["Cluster"].map(cluster_labels_map).fillna('Unknown')

# Reset index just in case country names are in the index
original_data = original_data.reset_index()

# Group by the cluster label and print 2 countries from each
for label, group in original_data.groupby("Cluster Label"):
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

#plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1],
                hue=original_data["Cluster Label"], palette="Set2", s=100)
plt.title("Country Clusters based on Land Use & Urban Growth")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster Description", loc='best')
plt.grid(True)
plt.show()

cluster_means = original_data.groupby("Cluster").mean(numeric_only=True)
pd.set_option("display.max_columns", None)  # So it doesn't cut off
print(cluster_means)


