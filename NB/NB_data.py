# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:19:08 2020

@author: sujin
"""
import pandas as pd

#import data
filename = 'NB_data.xlsx'
data = pd.read_excel(filename, index_col = 0, na_values= ['',' - ',0])
data = data.T
data.dropna(how = 'all', inplace=True)
data = data.fillna(0)

#feautres, labels
labels = data[['substrate']]
features = data.drop('substrate', axis = 1)

#encoding
from sklearn.preprocessing import StandardScaler

#train test data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size = 0.2)
#Standard Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#NB model
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#validation & evaluation
from sklearn.model_selection import cross_val_score, cross_val_predict
print("\n**cross validation score**\n", cross_val_score(clf, X_train, y_train, cv = 3, scoring = 'accuracy'))

from sklearn.metrics import plot_confusion_matrix, classification_report
y_train_pred = cross_val_predict(clf, X_train, y_train, cv = 3)

report = classification_report(y_train, y_train_pred)
print("\n**classification report**\n", report)
report_test = classification_report(y_test, y_pred)
print("\n**classification report**\n", report_test)

import matplotlib.pyplot as plt
plt.figure(1)
plot_confusion_matrix(clf, X_train, y_train)
plt.title("train data")
plt.figure(2)
plot_confusion_matrix(clf, X_test, y_test)
plt.title("test data")
plt.show()
plt.clf()




