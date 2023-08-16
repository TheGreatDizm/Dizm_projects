# We will use Agglomerative Clustering, a type of hierarchical clustering that follows a bottom up approach. We begin by treating each data point as its own cluster. Then, we join clusters together that have the shortest distance between them to create larger clusters. This step is repeated until one large cluster is formed containing all of the data points.
#
# Hierarchical clustering requires us to decide on both a distance and linkage method. We will use euclidean distance and the Ward linkage method, which attempts to minimize the variance between clusters.
# Start by visualizing some data points:
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# plt.scatter(x, y)
# plt.show()

# Now we compute the ward linkage using euclidean distance, and visualize it using a dendrogram:
# how? from scipy.cluster.hierarchy import dendrogram, linkage

data = list(zip(x, y))

# linkage_data = linkage(data, method='ward', metric='euclidean')
# dendrogram(linkage_data)

# plt.show()

# Here, we do the same thing with Python's scikit-learn library. Then, visualize on a 2-dimensional plot:

hierarchial_cluster = AgglomerativeClustering(n_clusters=2,
affinity='euclidean', linkage='ward')
labels=hierarchial_cluster.fit_predict(data)

plt.scatter(x, y, c=labels)
plt.show()