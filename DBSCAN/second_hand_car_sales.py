import numpy as np
from matplotlib import pyplot as plt
"""
Load data
"""

X_full = np.load('secondary_hand_car_sales.npy', allow_pickle=True)

# extract year, milage and price
X = X_full[:, (4, 5, 6)]

"""
Clustering using DBSCAN
"""
# first normalize using the standard scaler
from sklearn.preprocessing import StandardScaler as Scaler

z_scorer = Scaler()
X_normalized = z_scorer.fit_transform(X)

# now cluster using DBSCAN
from sklearn.cluster import DBSCAN as DBSCAN

#first work with default parameters
dbscan = DBSCAN()
cluster_labels = dbscan.fit_predict(X_normalized)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

plt.figure()
plt.subplot(1,2,1)
for i in range(n_clusters):
    plt.scatter(X[cluster_labels==i,0], X[cluster_labels==i,1], label=f'cluster {i}')
if -1 in cluster_labels:
    plt.scatter(X[cluster_labels==-1,0], X[cluster_labels==-1,1], label='outlier', color='gray')
plt.xlabel('year')
plt.ylabel('mileage (miles)')

plt.subplot(1,2,2)
for i in range(n_clusters):
    plt.scatter(X[cluster_labels==i,0], X[cluster_labels==i,2], label=f'cluster {i}')
if -1 in cluster_labels:
    plt.scatter(X[cluster_labels == -1, 0], X[cluster_labels == -1, 2], label='outlier', color='gray')
plt.xlabel('year')
plt.ylabel('price (pounds)')
plt.legend()
plt.tight_layout()
plt.savefig('clustering_default_params.png')
plt.show()

print(f'number of clusters found by DBSCAN (default parameters): {n_clusters}')


"""
Find optimal hyperparameter epsilon through grid search
"""
from sklearn.metrics import silhouette_score as sil_coeff

eps_grid = np.logspace(start=-3, stop=0, num=20)
num_clusters = []
sil_coeffs = []

for eps in eps_grid:
    dbscan = DBSCAN(eps=eps, min_samples=10)
    cluster_labels = dbscan.fit_predict(X_normalized)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    # record number of clusters and global silhouette coefficient
    num_clusters.append(n_clusters)
    if n_clusters > 1:  # only one cluster -> no possibility to compute silhouette value
        sil_coeffs.append(sil_coeff(X_normalized, cluster_labels))
    else:
        sil_coeffs.append(-1)

    print(f'eps={eps:.3f} --> {n_clusters} clusters')

num_clusters = np.array(num_clusters)
sil_coeffs = np.array((sil_coeffs))

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(eps_grid, num_clusters)
plt.ylabel('number of clusters')
plt.xlabel(r'$\epsilon$')

plt.subplot(1, 2, 2)
plt.plot(eps_grid, sil_coeffs)
plt.ylabel('silhouette coefficient')
plt.xlabel(r'$\epsilon$')
plt.ylim([-1.1, 1.1])
plt.savefig('hyperparameter_study.png')
plt.tight_layout()
plt.show()


eps_opt = eps_grid[np.argmax(sil_coeffs)]
num_opt = num_clusters[np.argmax(sil_coeffs)]
print(f'optimal parameters w.r.t. silhouette value: eps={eps_opt} --> {num_opt} clusters')


"""
Show optimal clustering (eps for maximum value of silhouette coefficient)
"""
dbscan = DBSCAN(eps=eps_opt, min_samples=10)
cluster_labels = dbscan.fit_predict(X_normalized)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

plt.figure()
plt.subplot(1,2,1)
for i in range(n_clusters):
    plt.scatter(X[cluster_labels==i,0], X[cluster_labels==i,1], label=f'cluster {i}')
if -1 in cluster_labels:
    plt.scatter(X[cluster_labels==-1,0], X[cluster_labels==-1,1], label='outlier', color='gray')
plt.xlabel('year')
plt.ylabel('mileage (miles)')
plt.title(fr'$\epsilon=${eps_opt:.3f}')

plt.subplot(1,2,2)
for i in range(n_clusters):
    plt.scatter(X[cluster_labels==i,0], X[cluster_labels==i,2], label=f'cluster {i}')
if -1 in cluster_labels:
    plt.scatter(X[cluster_labels == -1, 0], X[cluster_labels == -1, 2], label='outlier', color='gray')
plt.xlabel('year')
plt.ylabel('price (pounds)')
plt.legend()
plt.tight_layout()
plt.savefig('clustering_optimal_params.png')
plt.show()
