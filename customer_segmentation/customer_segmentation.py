import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Mall_Customers.csv')

# Data Preprocessing
df.dropna(inplace=True)
df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Cluster Analysis
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=df.columns[:-1])
cluster_counts = df['Cluster'].value_counts()

# Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=df)
plt.title('Customer Segments')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.show()

# Cluster Centers
print(cluster_centers)
