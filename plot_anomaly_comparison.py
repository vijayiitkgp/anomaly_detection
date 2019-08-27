"""
============================================================================
Comparing anomaly detection algorithms for outlier detection on toy datasets
============================================================================

This example shows characteristics of different anomaly detection algorithms
on 2D datasets. Datasets contain one or two modes (regions of high density)
to illustrate the ability of algorithms to cope with multimodal data.

For each dataset, 15% of samples are generated as random uniform noise. This
proportion is the value given to the nu parameter of the OneClassSVM and the
contamination parameter of the other outlier detection algorithms.
Decision boundaries between inliers and outliers are displayed in black
except for Local Outlier Factor (LOF) as it has no predict method to be applied
on new data when it is used for outlier detection.

The :class:`sklearn.svm.OneClassSVM` is known to be sensitive to outliers and
thus does not perform very well for outlier detection. This estimator is best
suited for novelty detection when the training set is not contaminated by
outliers. That said, outlier detection in high-dimension, or without any
assumptions on the distribution of the inlying data is very challenging, and a
One-class SVM might give useful results in these situations depending on the
value of its hyperparameters.

:class:`sklearn.covariance.EllipticEnvelope` assumes the data is Gaussian and
learns an ellipse. It thus degrades when the data is not unimodal. Notice
however that this estimator is robust to outliers.

:class:`sklearn.ensemble.IsolationForest` and
:class:`sklearn.neighbors.LocalOutlierFactor` seem to perform reasonably well
for multi-modal data sets. The advantage of
:class:`sklearn.neighbors.LocalOutlierFactor` over the other estimators is
shown for the third data set, where the two modes have different densities.
This advantage is explained by the local aspect of LOF, meaning that it only
compares the score of abnormality of one sample with the scores of its
neighbors.

Finally, for the last data set, it is hard to say that one sample is more
abnormal than another sample as they are uniformly distributed in a
hypercube. Except for the :class:`sklearn.svm.OneClassSVM` which overfits a
little, all estimators present decent solutions for this situation. In such a
case, it would be wise to look more closely at the scores of abnormality of
the samples as a good estimator should assign similar scores to all the
samples.

While these examples give some intuition about the algorithms, this
intuition might not apply to very high dimensional data.

Finally, note that parameters of the models have been here handpicked but
that in practice they need to be adjusted. In the absence of labelled data,
the problem is completely unsupervised so model selection can be a challenge.
"""



import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd


def file_processing(file, features, label, classes):
    
    data = pd.read_csv(file, encoding = "ISO-8859-1")

    classes_new = {'GREEN': 1,'RED': 0} 
    
    print("size of old data is:", data.shape)
    data = data[data['avg_sim']!=0]
    data = data[data['add_sim']!=0]
    # data = data[data['new_address'].to_string().isspace()==False]
    print("size of new data is:", data.shape)
    data = data[data['name_sim']!=0]
    print("size of latest new data is:", data.shape)
    
    data_temp= data[features]

    data_temp.to_csv(r'train_model_export_dataframe_score_6.csv', index = None, header=True)
    
    
    data[label] = [classes_new[item] for item in data[label] ]
    temp = data.drop(columns = [label])
    X = data[features]
    y = data[label]
   
    return X, y


print(__doc__)

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# Example settings
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

# define outlier/anomaly detection methods to be compared
anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Isolation Forest", IsolationForest(behaviour='new',
                                         contamination=outliers_fraction,
                                         random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction))]

# Define datasets
# blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
# datasets = [
#     make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
#                **blobs_params)[0],
#     make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
#                **blobs_params)[0],
#     make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
#                **blobs_params)[0],
#     4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
#           np.array([0.5, 0.25])),
#     14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))

plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1
rng = np.random.RandomState(42)

file = "export_dataframe_score_3.csv"
# features = ['score', 'name_sim', 'add_sim']
features = ['score', 'name_sim', 'add_sim', 'avg_sim', 'match_partial_ratio', 'match_ratio', 'match_token_set_ratio', 'match_token_sort_ratio']
#  score  name_sim   add_sim   avg_sim  match_partial_ratio  match_ratio  match_token_set_ratio  match_token_sort_ratio
# [0.14586648 0.03366048 0.265949   0.17974676 0.18904001 0.05718042 0.07151823 0.05703861]
# [0.11277863 0.07356189 0.25331652 0.21011871 0.11576029 0.10005042 0.06052418 0.07388936]
label = 'color'
classes= [0, 1]
# classes_new = {'GREEN': 1,'RED': 0} 
  
# # traversing through dataframe 
# # Gender column and writing 
# # values where key matches 
# data[label] = [gender[item] for item in data[label] 
X, y = file_processing(file, features, label, classes)
datasets = [X]

for i_dataset, X in enumerate(datasets):
    # Add outliers
    # X = np.concatenate([X, rng.uniform(low=-6, high=6,
    #                    size=(n_outliers, 2))], axis=0)

    for name, algorithm in anomaly_algorithms:
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        # fit the data and tag outliers
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit(X).predict(X)



        colors = np.array(['#377eb8', '#ff7f00'])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()
