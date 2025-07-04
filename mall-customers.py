import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Loading the raw mall customer dataset
raw_df = pd.read_csv("Mall_Customers-Original_Dataset.csv")

# Dropping off  CustomerID — because it is not really useful for our clustering logic
df = raw_df.drop("CustomerID", axis=1)

# Gender to numeric: Just using LabelEncoder since it is quick
gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

# Saving 
df.to_csv("Mall_Customers_Cleaned.csv", index=False)

# Relation of income to spending first
plt.figure(figsize=(8, 5))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], 
            color='skyblue', edgecolor='black', s=70)
plt.title("Annual Income vs. Spending Score")  # checking visually
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.tight_layout()
plt.savefig("rawdata_income_vs_spending.png")
# all plots to be shown together at the end 

# Trying a few different cluster sizes and seeing where the 'elbow' bends
elbow_scores = []
possible_k = list(range(1, 11))  # Trying from 1 to 10 clusters

for k in possible_k:
    trial_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    trial_kmeans.fit(df.values)
    elbow_scores.append(trial_kmeans.inertia_)

# Visualizing the elbow method result — to see where it starts flattening out
plt.figure(figsize=(8, 5))
plt.plot(possible_k, elbow_scores, 'o--', color='orange')
plt.title("Elbow Method: Optimal K?")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_method.png")

# For now, picking K=5 (for a good bend look)
final_k = 5
kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
cluster_assignments = kmeans.fit_predict(df.values)

# Append cluster back into the dataset
df["Cluster"] = cluster_assignments

# Saving the cleaned dataset but with clusters
df.to_csv("Mall_Customers_Cleaned_with_Clusters.csv", index=False)

# Now reducing dimensionality to 2D for visualization —  for a clean scatter plot
pca_model = PCA(n_components=2)
pca_result = pca_model.fit_transform(df.drop("Cluster", axis=1).values)  # avoiding leakage of 'Cluster' here

# Plotting the clusters on PCA-reduced axes
plt.figure(figsize=(8, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1], 
            c=cluster_assignments, cmap="viridis", s=70, edgecolors='k')
plt.title("Customer Segments (2D PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_clusters.png")

# for getting the silhouette score
sil_score = silhouette_score(df.drop("Cluster", axis=1).values, cluster_assignments)
print(f"Silhouette Score for K={final_k}: {sil_score:.3f}")
with open("silhouette_score.txt", "w") as f:
    f.write(f"Silhouette Score for K={final_k}: {sil_score:.3f}\n")


# Showing all plots at once at the end — for easy and smooth printing 
plt.show()
