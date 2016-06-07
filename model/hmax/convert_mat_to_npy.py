"""
Big data analysis of object shape representations

Convert matlab outputs to numpy npy format.

Created on May 9, 2016

Goker Erdogan
https://github.com/gokererdogan
"""

import scipy.stats as spst
import scipy.io as spio
import numpy as np
import sklearn.decomposition as d

m = spio.loadmat('results/activations.mat')

# save row labels
row_labels = [str(s[0]) for s in m['image_names'][0]]
open('hmax_row_labels.txt', 'w').write(repr(row_labels))

# separate matrices for each band/patch size and save them into npy files
c1 = m['c1'][0]
c2 = m['c2'][0]

pca = d.PCA(n_components=0.95)

for i in range(c1.size):
    x = c1[i].astype('float')
    x = x.T

    # get rid of constant columns
    d = np.max(x, axis=0) - np.min(x, axis=0)
    x = x[:, np.logical_not(np.isclose(d, 0.0))]    

    # normalize data
    # x = spst.zscore(x)
    c1[i] = x

    print(np.min(x), np.max(x))
    np.save("results/hmax_c1_{0:d}.npy".format(i), x)

    x = pca.fit_transform(x)
    np.save("results/hmax_c1_{0:d}_pca.npy".format(i), x)

x = np.concatenate(c1, axis=1)
np.save("results/hmax_c1.npy", x)

x = pca.fit_transform(x)
np.save("results/hmax_c1_pca.npy", x)

for i in range(c2.size):
    x = c2[i].astype('float')
    x = x.T

    # get rid of constant columns
    d = np.max(x, axis=0) - np.min(x, axis=0)
    x = x[:, np.logical_not(np.isclose(d, 0.0))]    
    
    # normalize data
    # x = spst.zscore(x)
    c2[i] = x

    print(np.min(x), np.max(x))
    np.save("results/hmax_c2_{0:d}.npy".format(i), x)

    x = pca.fit_transform(x)
    np.save("results/hmax_c2_{0:d}_pca.npy".format(i), x)

x = np.concatenate(c2, axis=1)
np.save("results/hmax_c2.npy", x)

x = pca.fit_transform(x)
np.save("results/hmax_c2_pca.npy".format(i), x)


