import pandas as pd
datasets = pd.read_csv('/mnt/c/users/usitgrams/desktop/meta-learning-base/datasets.csv')
names = datasets['name']
datasets = datasets.drop(columns=['id', 'name', 'cc', 'tp', 'st','et','st.1','pr','b','de','hcd'])

from sklearn.cluster import KMeans

# Ohne preprocessor
kmeans = KMeans(n_clusters=10)
kmeans.fit(datasets)
results = pd.Series(kmeans.labels_)
names_results = pd.concat([names,results], axis=1)

# Normalizer
from sklearn.preprocessing import Normalizer
transformed_dataset = Normalizer().fit_transform(datasets)
kmeanstransformed = KMeans(n_clusters=24)
kmeanstransformed.fit(transformed_dataset)
resultstransformed = pd.Series(kmeanstransformed.labels_)
names_resultstransformed = pd.concat([names,resultstransformed], axis=1)

# Standard Scaler
from sklearn.preprocessing import StandardScaler
scaled_dataset = StandardScaler().fit_transform(datasets)
kmeansscaled = KMeans(n_clusters=20)
kmeansscaled.fit(scaled_dataset)
resultsscaled = pd.Series(kmeansscaled.labels_)
names_resultsscaled = pd.concat([names,resultsscaled], axis=1)

# Datasets zusammenführen
from sklearn.metrics import pairwise_distances_argmin_min
closest, _ = pairwise_distances_argmin_min(kmeanstransformed.cluster_centers_, transformed_dataset)
test = names_resultstransformed.loc[closest,:]
test2 = datasets.loc[closest,:]

# Inertia zeichnen
import matplotlib.pyplot as plt
SSE = []
for cluster in range(1,160):
    kmeans = KMeans(n_clusters = cluster)
    kmeans.fit(transformed_dataset)
    SSE.append(kmeans.inertia_)

frame = pd.DataFrame({'Cluster':range(1,160), 'SSE':SSE})
plt.figure()
plt.plot(frame['Cluster'], frame['SSE'], linewidth=4)
plt.xlabel('Zahl der Cluster', fontsize=15)
plt.ylabel('Inertia', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('/mnt/c/users/usitgrams/desktop/inertia.png')
plt.show()

# Inertia Abnahme je Schritt
for iner in range(1,159):
    print(str(iner) + ' - ' + str(SSE[iner - 1] - SSE[iner]))

# Distanzen der gewählten Datasets
from itertools import combinations
from sklearn.metrics.pairwise import paired_euclidean_distances
for comb in combinations(test2.index,2):
    print(str(comb) + ':' + str(paired_euclidean_distances(test2[test2.index == comb[0]], test2[test2.index == comb[1]])))

# Zahl je Cluster
k = pd.Series(kmeanstransformed.labels_)
for zahl in k.unique():
    print(kmeanstransformed.labels_.tolist().count(zahl))

# 1 2 3 4 5 7 8 10 13 16 19
names_resultstransformed.loc[:,1]
names_resultstransformed = names_resultstransformed[names_resultstransformed[0] in [1, 2, 3, 4, 5, 7, 8, 10, 13, 16]]
