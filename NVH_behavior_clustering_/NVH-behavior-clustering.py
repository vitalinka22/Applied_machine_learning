import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("brake_data_assignment.txt", delimiter = ",")
print(f"shape of data: {data.shape}")

# extract feature channels
channels = ["p", "v", "alpha", "f", "l"]
units = ['bar', 'km/h', 'deg', 'kHz', 'dB(A)']

# value ranges and mean/median

for i, chn in enumerate(channels):
    print(f"rahge of {chn}: \t{np.min(data[:, i]):.2f} - {np.max(data[:, i]):.2f} [{units[i]}]")
    print(
        f'mean of {chn}: \t\t{np.mean(data[:, i]):.2f}[{units[i]}]. \t median: {np.median(data[:, i]):.2f}[{units[i]}]\n')
    
# number of brake stops before data claning
N = data.shape[0]
N_noisy = len(np.where(data[:, -1] > 0)[0])
print(f"number of brake stops: {N}. \t number of noisy brake stops: {N_noisy} ")

"""
Data cleaning:
    - remove negative brake pressure values
    - remove negative velocities larger than -20km/h
    - remove steering angles larger than 60 deg
    - [optional] remove non-audible noise larger than 14 kHz
    - [optional] remove sound levels that are not audible anyway (e.g. 0<SPL<40)
"""

# unpack into column vectors

idx_remove = np.where((data[:, 0] < 0|
                      (data[:, 1] < -20) |
                      (np.abs(data[:, 2]) > 60) |
                      (data[:, 3] > 14) |
                      ((data[:, 4] < 40) & (data[:, 4] > 0)))[0])

print(f"number of rows to remove for cleaning: {len(idx_remove)}")

# remove outliers and see what's left
data = np.delete(data, idx_remove, axis = 0)
print(f"new data shape: {data.shape}\n")
p, v, alpha, f, l = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]

for i, chn in enumerate(channels):
    print(f'[after cleaning]: range of {chn}: \t{np.min(data[:, i]):.2f} - {np.max(data[:, i]):.2f} [{units[i]}]')
    print(
        f'[after cleaning]: mean of {chn}: \t{np.mean(data[:, i]):.2f}[{units[i]}]. \t median: {np.median(data[:, i]):.2f}[{units[i]}]\n')
    
# distribution of values per channel
plt.figure(figsize=(8, 4), dpi=100)
for i, chn in enumerate(channels):
    plt.subplot(2, 3, i + 1)
    if chn == 'f' or chn == 'l':
        idx_plot = data[:, i] > 0
        plt.hist(data[idx_plot, i], bins=100, density=True, color='black')
    else:
        plt.hist(data[:, i], bins=100, density=True, color='black')

    plt.xlabel(f'{chn} [{units[i]}]')
    plt.ylabel('probability')

plt.tight_layout()
plt.show()

# loading conditions labeled whether noisy or not
idx_noisy = data[:, 4] > 0

plt.figure(figsize=(8, 3), dpi=100)
plt.subplot(1, 3, 1)
plt.scatter(data[~idx_noisy, 0], data[~idx_noisy, 1], color='gray')
plt.scatter(data[idx_noisy, 0], data[idx_noisy, 1], color='red')
plt.xlabel(f'{channels[0]} [{units[0]}]')
plt.ylabel(f'{channels[1]} [{units[1]}]')

plt.subplot(1, 3, 2)
plt.scatter(data[~idx_noisy, 0], data[~idx_noisy, 2], color='gray')
plt.scatter(data[idx_noisy, 0], data[idx_noisy, 2], color='red')
plt.xlabel(f'{channels[0]} [{units[0]}]')
plt.ylabel(f'{channels[2]} [{units[2]}]')

plt.subplot(1, 3, 3)
plt.scatter(data[~idx_noisy, 1], data[~idx_noisy, 2], color='gray')
plt.scatter(data[idx_noisy, 1], data[idx_noisy, 2], color='red')
plt.xlabel(f'{channels[1]} [{units[1]}]')
plt.ylabel(f'{channels[2]} [{units[2]}]')
plt.legend(['no noise', 'noise'])

plt.tight_layout()
plt.show()

# and the requested diagram f vs l. looks much cleaner than in the beginning!
plt.figure(figsize=(4, 4), dpi=100)
plt.plot(f, l, linestyle='none', marker='.', markersize=10, color='black')
plt.xlim([0.0, 16])
plt.ylim([0, 120])
plt.xlabel('frequency [kHz]')
plt.ylabel('SPL [dB(A)]')
plt.show()

"""
Let's do some math. If a dominant instability is given in a range of 0.5kHz, 
and the frequencies are in between 0.5 and 14.0, then we need to compute the 
epsilon value that will separate those sounds. if the range of 13.5kHz (max-min) 
is scaled to 0-1, then a frequency window of 0.5kHz will become 0.5/13.5 = 0.037
But since epsilon is the radius and not the diameter, it will be 0.5/13.5/2 = 0.0185
This epsilon should capture a dominant instability, as everything within this ball 
will be reachable. 

Now for the level scaling. The values range from 40 to 100, giving a range of 60. 
Frequencies vary by 13.5kHz, levels by 60dB, which is a factor of 60/13.5 = 4.4445

To keep the aspect ratio of the clusters that we can identify by eye, we need 
to scale not from 0 to 1 in both dimensions, but: 
    - frequency: 0 - 1
    - levels: 0 - 1/4.445=0.225

"""


# we need to take only the noisy stops obviously, not the complete data set
idx_noisy = np.where(f > 0)[0]

# we will use min-max normalization in this example
f_range = np.max(f[idx_noisy]) - np.min(f[idx_noisy])
l_range = np.max(l[idx_noisy]) - np.min(l[idx_noisy])

eps_max = 0.5 / 2 / f_range
scale_range_f = 1
scale_range_l = 1 / (l_range / f_range)


def scale_to_range(x: np.ndarray, xmin: float = 0, xmax: float = 1) -> np.ndarray:
    a = np.min(x)
    b = np.max(x)

    x_scaled = (x - a) / (b - a) * (xmax - xmin) + xmin

    return x_scaled


# let's re-scale both dimensions to the same value range such that we can make
# use of Eucledian distance metrics
f_norm = scale_to_range(x=f[idx_noisy], xmin=0, xmax=scale_range_f)
l_norm = scale_to_range(x=l[idx_noisy], xmin=0, xmax=scale_range_l)
X = np.vstack((f_norm, l_norm)).T

plt.figure(figsize=(4, 4), dpi=100)
plt.plot(f_norm, l_norm, linestyle='none', marker='.', markersize=10, color='black')
plt.xlabel('frequency [kHz]')
plt.ylabel('SPL [dB(A)]')
plt.title('rescaled data')
plt.show()

from sklearn.cluster import DBSCAN as DBSCAN
# note that your eps and min_samples depend on your scaling method
clustering = DBSCAN(eps=0.03, min_samples=4, metric='euclidean').fit(X)
labels_dbscan = clustering.labels_
K = len(np.unique(labels_dbscan))
print(f'number of clusters found by DBSCAN: {K}')

plt.figure(figsize=(4, 4), dpi=100)
leg_entr = []
for c in np.unique(labels_dbscan):
    plt.plot(X[labels_dbscan == c, 0], X[labels_dbscan == c, 1], linestyle='none', marker='.', markersize=10, )
    leg_entr.append('cluster ' + str(c))
# plt.axis('scaled')
plt.xlabel('frequency [kHz]')
plt.ylabel('SPL [dB(A)]')
plt.legend(leg_entr)
plt.show()

# let's get the information back into the real data range, and assign labels
labels = np.ones_like(f) * (-1)  # fill with -1 for no noisy / outlier
for c in np.unique(labels_dbscan):
    idx_reduced = np.where(labels_dbscan == c)[0]
    idx_full = idx_noisy[idx_reduced]
    labels[idx_full] = c

labels[~idx_noisy] = -1  # make sure to give all non-noisy stops a -1

# summary of clusters and mean frequency values
clusters = np.unique(labels)
clusters = np.delete(clusters, clusters == -1)  # get rid of outliers

cluster_freqs = [np.mean(f[labels == c]) for c in clusters]  # mean cluster frequency
cluster_num = [np.sum(labels == c) for c in clusters]  # member per cluster
cluster_mem_idx = [np.where(labels == c)[0] for c in clusters]  # index of cluster members

for i, c in enumerate(clusters):
    print(f'cluster {c}: \tmean frequency {cluster_freqs[i]:.2f} kHz, \
          level range {np.min(l[cluster_mem_idx[i]]):.2f} - {np.max(l[cluster_mem_idx[i]]):.2f} dB(A)')

plt.figure(figsize=(4, 4), dpi=100)
for c in clusters:
    if c != -1:
        plt.plot(f[labels == c], l[labels == c], linestyle='none', marker='.', markersize=10, )
plt.xlabel('frequency [kHz]')
plt.ylabel('SPL [dB(A)]')
plt.title('clean clustering result')
plt.xlim([0, 16])
plt.ylim([40, 120])
plt.show()

"""
And finally we can go into some more detailed analysis of the loading conditions

We will see:
    - cluster at 2.8kHz while driving in the city
    - cluster at 12.1kHz while driving overland
    - cluster at 5.8kHz while driving on the autobahn
    - cluster at 3.6kHz at harsh braking (emergency)
"""

plt.figure(figsize=(8, 3), dpi=100)
plt.subplot(1, 3, 1)
plt.plot(data[labels == -1, 0], data[labels == -1, 1], linestyle='none', marker='.', markersize=10, color='gray',
         alpha=0.5)
plt.xlabel(f'{channels[0]} [{units[0]}]')
plt.ylabel(f'{channels[1]} [{units[1]}]')
for c in clusters:
    plt.plot(data[labels == c, 0], data[labels == c, 1], linestyle='none', marker='.', markersize=10, alpha=0.5)

plt.subplot(1, 3, 2)
plt.plot(data[labels == -1, 0], data[labels == -1, 2], linestyle='none', marker='.', markersize=10, color='gray',
         alpha=0.5)
plt.xlabel(f'{channels[0]} [{units[0]}]')
plt.ylabel(f'{channels[2]} [{units[2]}]')
for c in clusters:
    plt.plot(data[labels == c, 0], data[labels == c, 2], linestyle='none', marker='.', markersize=10, alpha=0.5)

plt.subplot(1, 3, 3)
plt.plot(data[labels == -1, 1], data[labels == -1, 2], linestyle='none', marker='.', markersize=10, color='gray',
         alpha=0.5)
plt.xlabel(f'{channels[1]} [{units[1]}]')
plt.ylabel(f'{channels[2]} [{units[2]}]')
leg_entr = ['no sound']
for i, c in enumerate(clusters):
    plt.plot(data[labels == c, 1], data[labels == c, 2], linestyle='none', marker='.', markersize=10, alpha=0.5)
    leg_entr.append('cluster f=' + str(np.round(cluster_freqs[i], 1)) + 'kHz')

plt.legend(leg_entr, bbox_to_anchor=(1.1, 1.05))
plt.tight_layout()
plt.show()
