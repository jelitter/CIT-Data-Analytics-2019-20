#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 11:58:04 2019

@author: farshadtoosi
"""

import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Note: Not all the libararies above are necessarily needed for this project and not all
# the libraries you need for this project are necessarily listed above.


""" Your Name and Your student ID: 
    Isaac Sanchez
    R00156019
"""


def task1():
    """ Task 1 """
    def test_accuracy(ds):
        array = ds.values
        # 6th column (index 5) is 'y' for Dataset 1, 'loan' for Dataset 2
        X = array[:, 0:5]
        Y = array[:, 5]
        X_train, X_validation, Y_train, Y_validation = train_test_split(
            X, Y, test_size=1, random_state=1)

        # Tested all these algorithms. LogisticRegression (LR) resulted in the highest accuracy.
        #   LR: Accuracy: 0.882714 - Error: 0.117286 (Std Dev: 0.000201)
        #   LDA: Accuracy: 0.882438 - Error: 0.117562 (Std Dev: 0.000389)
        #   KNN: Accuracy: 0.869857 - Error: 0.130143 (Std Dev: 0.002554)
        #   CART: Accuracy: 0.818735 - Error: 0.181265 (Std Dev: 0.007136)
        #   NB: Accuracy: 0.870327 - Error: 0.129673 (Std Dev: 0.001553)

        model = ('LR', LogisticRegression(
            solver='liblinear', multi_class='ovr'))
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        cv_results = cross_val_score(
            model[1], X_train, Y_train, cv=kfold, scoring='accuracy')
        print('%s: Accuracy: %f - Error: %f (Std Dev: %f)' %
              (model[0], cv_results.mean(), (1 - cv_results.mean()), cv_results.std()))
        return 1 - cv_results.mean()

    # Dataset 1
    dataset_1 = dataset[['age', 'job', 'poutcome', 'balance', 'default', 'y']]
    print("\n💽 Dataset 1:")
    error_1 = test_accuracy(dataset_1)

    # Dataset 2
    dataset_2 = dataset[['age', 'job',
                         'poutcome', 'balance', 'default', 'loan']]
    print("\n💽 Dataset 2:")
    error_2 = test_accuracy(dataset_2)

    print("\n🔎 Dataset 1" if error_1 < error_2 else "Dataset 2",
          "has higher accuracy (lower error)")


def task2():
    """ Task 2 """
    # am = raw_dataset[['age','marital']]
    am = dataset[['age', 'marital']]

    plt.figure(figsize=(15, 8))

    # Generate object header for the data points
    obj_names = []
    for i in range(1, len(am)+1):
        obj = "Object " + str(i)
        obj_names.append(obj)

    # Create a pandas DataFrame with the names and (x, y) coordinates
    data = pd.DataFrame({
        'Object': obj_names,
        'X_value': am['marital'],
        'Y_value': am['age']
    })

    # 0 - divorced
    # 1 - married
    # 2 - single

    marital_status = am['marital'].unique()
    marital_status_str = [s.title() for s in raw_dataset['marital'].unique()]

    # Initialize the centroids
    c1 = (marital_status[0], 50)
    c2 = (marital_status[1], 50)
    c3 = (marital_status[2], 50)

    def calculate_distance(centroid, X, Y):
        distances = []

        # Unpack the x and y coordinates of the centroid
        c_x, c_y = centroid

        # Iterate over the data points and calculate the distance
        for x, y in list(zip(X, Y)):
            root_diff_x = (x - c_x) ** 2
            root_diff_y = (y - c_y) ** 2
            distance = np.sqrt(root_diff_x + root_diff_y)
            distances.append(distance)

        return distances

    # Calculate the distance and assign them to the DataFrame accordingly
    data['C1_Distance'] = calculate_distance(c1, data.X_value, data.Y_value)
    data['C2_Distance'] = calculate_distance(c2, data.X_value, data.Y_value)
    data['C3_Distance'] = calculate_distance(c3, data.X_value, data.Y_value)

    # Get the minimum distance centroids
    data['Cluster'] = data[['C1_Distance', 'C2_Distance',
                            'C3_Distance']].apply(np.argmin, axis=1)

    # Map the centroids accordingly and rename them
    data['Cluster'] = data['Cluster'].map(
        {'C1_Distance': marital_status_str[0], 'C2_Distance': marital_status_str[1], 'C3_Distance': marital_status_str[2]})

    # Get a preview of the data
    print("\n🎯 Data points and distances to clusters centers:")
    print(data.head())

    # Calculate the coordinates of the new centroid from cluster 1
    x_new_centroid1 = data[data['Cluster'] ==
                           marital_status_str[0]]['X_value'].mean()
    y_new_centroid1 = data[data['Cluster'] ==
                           marital_status_str[0]]['Y_value'].mean()

    # Calculate the coordinates of the new centroid from cluster 2
    x_new_centroid2 = data[data['Cluster'] ==
                           marital_status_str[1]]['X_value'].mean()
    y_new_centroid2 = data[data['Cluster'] ==
                           marital_status_str[1]]['Y_value'].mean()

    # Calculate the coordinates of the new centroid from cluster 3
    x_new_centroid3 = data[data['Cluster'] ==
                           marital_status_str[2]]['X_value'].mean()
    y_new_centroid3 = data[data['Cluster'] ==
                           marital_status_str[2]]['Y_value'].mean()

    # Print the coordinates of the new centroids
    print('\n🎯 Centroid 1 ({}) ({}, {})'.format(
        marital_status_str[0], x_new_centroid1, y_new_centroid1))
    print('🎯 Centroid 2 ({}) ({}, {})'.format(
        marital_status_str[1], x_new_centroid2, y_new_centroid2))
    print('🎯 Centroid 3 ({}) ({}, {})\n'.format(
        marital_status_str[2], x_new_centroid3, y_new_centroid3))

    # Specify the number of clusters (3) and fit the data X
    kmeans = KMeans(n_clusters=3, random_state=0).fit(am)

    # Get the cluster centroids
    print(kmeans.cluster_centers_)

    # Plotting the cluster centers and the data points on a 2D plane

    sns.set(font_scale=1.6)

    # Scatter plots dataset
    plt.scatter(am['age'], am['marital'])

    # Scatter plot centroids
    plt.scatter(kmeans.cluster_centers_[
                :, 0], am['marital'].unique(), c='red', marker='D', s=150)

    plt.title('Marital Status by Age')
    plt.xlabel('Age')
    plt.ylabel('Marital Status')
    plt.tick_params(labelsize=12, pad=6)
    plt.yticks(np.arange(3), marital_status_str)

    age1 = kmeans.cluster_centers_[0, 0]
    age2 = kmeans.cluster_centers_[1, 0]
    age3 = kmeans.cluster_centers_[2, 0]

    # Annotate centroids' ages
    plt.annotate(f"{round(age1, 1)} years old",
                 xy=(age1, marital_status[0]),
                 xytext=(age1 + 2, marital_status[0] - 0.2),
                 arrowprops=dict(facecolor='red', shrink=0.01))

    plt.annotate(f"{round(age2, 1)} years old",
                 xy=(age2, marital_status[1]),
                 xytext=(age2 + 2, marital_status[1] - 0.2),
                 arrowprops=dict(facecolor='red', shrink=0.01))

    plt.annotate(f"{round(age3, 1)} years old",
                 xy=(age3, marital_status[2]),
                 xytext=(age3 + 2, marital_status[2] + 0.15),
                 arrowprops=dict(facecolor='red', shrink=0.01))

    plt.show()


def task3():
    """ Task 3 """


def task4():
    """ Task 4 """


def task5():
    """ Task 5 """


def task6():
    """ Task 6 """


def task7():
    """ Task 7 """
