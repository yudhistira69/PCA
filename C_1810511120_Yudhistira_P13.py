# -*- coding: utf-8 -*-
"""
Created on Wed Jun 9 11:20:19 2021

@author: Asus
"""
#import library
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score

#Baca data
df = pd.read_csv('data.csv')
features = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
 'compactness_mean','concavity_mean','concave points_mean','symmetry_mean',
 'fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se',
 'smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se',
'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst',
 'smoothness_worst','compactness_worst','concavity_worst','concave points_worst',
 'symmetry_worst','fractal_dimension_worst']

#pisahkan data (X) dengan target (y)
X = df.loc[:, features].values
y = df.loc[:,['diagnosis']].values

#Normalisasi data dengan menggunakan min-max
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

#Klasifikasi SVM menggunakan seluruh data dengan seluruh fitur tanpa reduksi dimensi
print("Klasifikasi SVM menggunakan seluruh data dengan seluruh fitur tanpa reduksi dimensi")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=5)
print("Accuracy: %0.2f" % (scores.mean()))

#Reduksi dimensi menggunakan PCA
pca = PCA(n_components=30)
principalComponents = pca.fit_transform(X)
variance_ratio = pca.explained_variance_ratio_

#Dengan 3 PC
print("Dengan 3 PC")
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)
variance_ratio = pca.explained_variance_ratio_
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, principalComponents, y, cv=5)
print("Accuracy: %0.2f" % (scores.mean()))

#Dengan 5 PC.
print("Dengan 5 PC")
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(X)
variance_ratio = pca.explained_variance_ratio_
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, principalComponents, y, cv=5)
print("Accuracy: %0.2f" % (scores.mean()))

#Dengan 10 PC (Principal Component)
print("Dengan 10 PC")
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X)
variance_ratio = pca.explained_variance_ratio_
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, principalComponents, y, cv=5)
print("Accuracy: %0.2f" % (scores.mean()))







