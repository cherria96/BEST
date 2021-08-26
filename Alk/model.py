# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:47:41 2021

@author: sujin
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from functions import comparison
from functions_2 import display_scores, comparison2, rmse, mape, split_data
from sklearn.inspection import permutation_importance
import pandas as pd

class ML:
    def __init__(self, model, features, target):
        self.model = model
        self.features = features
        self.target = target
    def model_fit(self, x, y, early_stopping = False):
        if early_stopping == False:
            self.model.fit(x, y)
        else:
            X_train, X_val, y_train, y_val = train_test_split(x,y)
            self.model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_val, y_val)], early_stopping_rounds = 20)  
            return X_train, X_val, y_train, y_val 
    def predict(self, x, y):
        pred = self.model.predict(x)
        rmse(y, pred)
        mape(y, pred)
        compare = comparison(pred, y, title = 'prediction')
        comparison2(pred, y, title = 'prediction')
        plt.scatter(compare.actual_value, compare.error)
        plt.ylim([0,100])
        plt.show()
        return compare, pred
    def cross_val(self, x, y):
        display_scores(self.model, x, y)
    def permutation_importance_plot(self, X, y):
        result = permutation_importance(self.model, X, y, n_repeats= 20, random_state = 20, n_jobs = -1)
        importances = pd.Series(result.importances_mean, index = self.features.columns)
        importances.sort_values(inplace = True)
        fig, ax = plt.subplots()
        importances.plot.bar(yerr = result.importances_std, ax = ax)
        ax.set_title("feature importances using permutation on model")
        ax.set_ylabel('Mean accuracy decrease')
        fig.tight_layout()
        plt.show()
        return importances


import tensorflow as tf

def build_model(input_shape, n_hidden=10, n_neurons = 30, learning_rate = 0.01):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = input_shape))
    for layer in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    
    model.compile(loss = 'mse', optimizer = optimizer)
    return model


