import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("scada_pipeline.csv")

# -------------------------------
# Drop unnecessary columns
# -------------------------------
df = df.drop(columns=["target", "event_type"], errors="ignore")

# -------------------------------
# Timestamp feature engineering
# -------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

df = df.drop(columns=["timestamp"])

# -------------------------------
# Remove ID column if exists
# -------------------------------
df = df.drop(columns=["segment_id"], errors="ignore")

# -------------------------------
# Final feature matrix
# -------------------------------
X = df.copy()
feature_columns = X.columns  # IMPORTANT FIX

# -------------------------------
# Feature scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Elbow Method
# -------------------------------
wcss = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# -------------------------------
# Train final KMeans model
# -------------------------------
k = 3  # change after elbow analysis

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# -------------------------------
# Add cluster labels
# -------------------------------
df["Cluster"] = clusters

# -------------------------------
# Cluster distribution
# -------------------------------
print("\nCluster Counts:\n")
print(df["Cluster"].value_counts())

# -------------------------------
# Silhouette Score
# -------------------------------
score = silhouette_score(X_scaled, clusters)
print("\nSilhouette Score:", round(score, 3))

# -------------------------------
# Cluster Centers (FIXED)
# -------------------------------
centers = scaler.inverse_transform(kmeans.cluster_centers_)

centers = pd.DataFrame(
    centers,
    columns=feature_columns
)

print("\nCluster Centers:\n")
print(centers)

# -------------------------------
# Cluster Summary
# -------------------------------
print("\nCluster Summary:\n")
print(df.groupby("Cluster").mean(numeric_only=True))

# -------------------------------
# PCA Visualization
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis"
)

plt.title("K-Means Clusters (PCA View)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")

plt.show()