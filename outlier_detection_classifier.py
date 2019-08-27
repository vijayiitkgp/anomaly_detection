import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# from train_model import *
# %matplotlib inline
import matplotlib.font_manager

from pyod.models.abod import ABOD
from pyod.models.knn import KNN
import pandas as pd

from pyod.utils.data import generate_data, get_outliers_inliers

# from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
# from pyod.models.knn import KNN
from pyod.models.lof import LOF

def file_processing(file, features, label, classes):
    
    data = pd.read_csv(file, encoding = "ISO-8859-1")
    # data=data[data['flag']==True]
    # data = data.drop(['verify_id', 'link','CreationDate','PincodeMatch','CityMatch','StateMatch','ApplicantID','ProductID','CheckNo'], axis = 1)
        
    # features =['score','NamePercentage','AddPercentage']
    # classes = ['0', '1']
    #features =['score','AddPercentage']
    classes_new = {'GREEN': 0,'RED': 1} 
  
    # traversing through dataframe 
    # Gender column and writing 
    # values where key matches 
    data[label] = [classes_new[item] for item in data[label] ]
    X = data[features]
    y = data[label]
    # print(y.head)
    return X, y


file = "export_dataframe_score_3.csv"
features = ['score', 'name_sim', 'add_sim', 'avg_sim', 'match_partial_ratio', 'match_ratio', 'match_token_set_ratio', 'match_token_sort_ratio']
label = 'color'
classes= [0, 1]
# classes_new = {'GREEN': 1,'RED': 0} 

X, y = file_processing(file, features, label, classes)

#generate random data with two features
X_train, Y_train = X.values, y.values #generate_data(n_train=200,train_only=True, n_features=2)

# by default the outlier fraction is 0.1 in generate data function 
outliers_fraction = 0.001

# store outliers and inliers in different numpy arrays
x_outliers = X_train[np.where(Y_train == 1)]
x_inliers = X_train[np.where(Y_train == 0)]
# x_outliers, x_inliers = get_outliers_inliers(X_train,Y_train)

n_inliers = len(x_inliers)
print(n_inliers)
n_outliers = len(x_outliers)
print(n_outliers)

#separate the two features and use it to plot the data 
F1 = X_train[:,[0]].reshape(-1,1)
F2 = X_train[:,[1]].reshape(-1,1)

# create a meshgrid 
xx , yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))

# scatter plot 
plt.scatter(F1,F2)
plt.xlabel('F1')
plt.ylabel('F2') 
# plt.show()

# Create a dictionary and add all the models that you want to use to detect the outliers:
random_state = np.random.RandomState(42)

classifiers = {
        'Angle based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
        'Cluster based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
        'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
        'Histogram base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'Average KNN': KNN(method='mean',contamination=outliers_fraction)
}
# Fit the data to each model we have added in the dictionary, Then, see how each model is detecting outliers:

#set the figure size
plt.figure(figsize=(10, 10))

for i, (clf_name,clf) in enumerate(classifiers.items()) :
    # fit the dataset to the model
    clf.fit(X_train)

    # predict raw anomaly score
    scores_pred = clf.decision_function(X_train)*-1

    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X_train)
    print(y_pred, Y_train )
    pred = pd.DataFrame(y_pred)
    file_pred ='pred_'+clf_name+'.csv'
    pred.to_csv (file_pred, index = None, header=True)
    y_train_df = pd.DataFrame(Y_train)
    file_train ='y_train_df'+clf_name+'.csv'
    y_train_df.to_csv (file_train, index = None, header=True)
    
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers, clf_name)

    # no of errors in prediction
    n_errors = (y_pred != Y_train).sum()
    print('No of Errors : ',clf_name, n_errors)
    
 '''   
[0 0 0 ... 0 0 0] [0 0 0 ... 0 0 0]
OUTLIERS :  0 INLIERS :  197516 Angle based Outlier Detector (ABOD)
No of Errors :  Angle based Outlier Detector (ABOD) 143
[0 0 0 ... 0 0 0] [0 0 0 ... 0 0 0]
OUTLIERS :  198 INLIERS :  197318 Cluster based Local Outlier Factor (CBLOF)
No of Errors :  Cluster based Local Outlier Factor (CBLOF) 325
[0 0 0 ... 0 0 0] [0 0 0 ... 0 0 0]
OUTLIERS :  190 INLIERS :  197326 Feature Bagging
No of Errors :  Feature Bagging 333
[0 0 0 ... 0 0 0] [0 0 0 ... 0 0 0]
OUTLIERS :  198 INLIERS :  197318 Histogram base Outlier Detection (HBOS)
No of Errors :  Histogram base Outlier Detection (HBOS) 283
C:\Users\VP999274\AppData\Roaming\Python\Python37\site-packages\sklearn\ensemble\iforest.py:247: FutureWarning: behaviour="old" is deprecated and will be removed in version 0.22. Please use behaviour="new", which makes the decision_function change to match other anomaly detection algorithm API.
  FutureWarning)
[0 0 0 ... 0 0 0] [0 0 0 ... 0 0 0]
OUTLIERS :  198 INLIERS :  197318 Isolation Forest
No of Errors :  Isolation Forest 281
[0 0 0 ... 0 0 0] [0 0 0 ... 0 0 0]
OUTLIERS :  139 INLIERS :  197377 K Nearest Neighbors (KNN)
No of Errors :  K Nearest Neighbors (KNN) 266
[0 0 0 ... 0 0 0] [0 0 0 ... 0 0 0]
OUTLIERS :  108 INLIERS :  197408 Average KNN
No of Errors :  Average KNN 237

'''

#     # rest of the code is to create the visualization

#     # threshold value to consider a datapoint inlier or outlier
#     threshold = stats.scoreatpercentile(scores_pred,100 *outlier_fraction)

#     # decision function calculates the raw anomaly score for every point
#     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
#     Z = Z.reshape(xx.shape)

#     subplot = plt.subplot(1, 2, i + 1)

#     # fill blue colormap from minimum anomaly score to threshold value
#     subplot.contourf(xx, yy, Z, levels = np.linspace(Z.min(), threshold, 10),cmap=plt.cm.Blues_r)

#     # draw red contour line where anomaly score is equal to threshold
#     a = subplot.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')

#     # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
#     subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')

#     # scatter plot of inliers with white dots
#     b = subplot.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c='white',s=20, edgecolor='k') 
#     # scatter plot of outliers with black dots
#     c = subplot.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c='black',s=20, edgecolor='k')
#     subplot.axis('tight')

#     subplot.legend(
#         [a.collections[0], b, c],
#         ['learned decision function', 'true inliers', 'true outliers'],
#         prop=matplotlib.font_manager.FontProperties(size=10),
#         loc='lower right')

#     subplot.set_title(clf_name)
#     subplot.set_xlim((-10, 10))
#     subplot.set_ylim((-10, 10))
# plt.show() 
