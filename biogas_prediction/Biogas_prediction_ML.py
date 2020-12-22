# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:20:55 2020

@author: sujin
"""

import Biogas_prediction_excel_data as B
import pandas as pd
import numpy as np
filename = B.filename
sheet_name = B.sheet_name
dataset = B.data1
target_attributes = B.target_attributes
corr_value = B.corr_value

targets = dataset[target_attributes]
features = dataset[B.feature_attributes]

#train, test data split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

#scaling data
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
standardscaler= StandardScaler()
X_train = standardscaler.fit_transform(x_train)
X_test = standardscaler.transform(x_test)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


#Deep Learning model fitting
print('trial #4 : Deep Learning')
import tensorflow as tf
X = tf.keras.layers.Input(shape = [len(B.feature_attributes)])
H = tf.keras.layers.Dense(20)(X)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('relu')(H)

H = tf.keras.layers.Dense(40)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('relu')(H)

H = tf.keras.layers.Dense(80)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('relu')(H)

Y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X,Y)
model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError())
print(model.summary())

#Visualization_Tensorboard
import os 
root_logdir = os.path.join(os.curdir, 'my_logs')
def get_run_logdir():
    import time
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()

tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

#model fitting

while True:
    batch_size = input('batch_size:')
    epochs = input('epochs: ')
    verbose = input('To hide = 0 / To show = 1 : ')
    history = model.fit(X_train,y_train, batch_size = int(batch_size), epochs = int(epochs), verbose = int(verbose), 
              shuffle = True, callbacks= [tensorboard_cb])
    model.evaluate(X_test, y_test)
    comparison = pd.DataFrame(model.predict(X_test), columns= ['prediction_value'] )
    comparison['actual_value'] = pd.DataFrame(y_test)
    print(comparison)
    ans = input('continue? [y/n] : ')
    if ans == 'n':
        break


#Visualization of the tensorflow callback history
import matplotlib.pyplot as plt
df = pd.DataFrame(history.history)

print('Deep learning loss(%) :', df['loss'].iloc[-1])
df.plot(figsize = (8,5))
plt.xlabel('epoch')
plt.ylabel('loss(%)')
plt.grid(True)
plt.gca().set_ylim(0,100)
plt.show()

#visualization of the result
plt.figure(figsize = (12,8))
plt.plot(comparison.index, comparison[['prediction_value']], label = 'prediction_value')
plt.scatter(comparison.index, comparison[['actual_value']], color='orange', label = 'actual_value')
plt.xlabel('index')
plt.ylabel('biogas production')
plt.ylim(min(targets)*0.8, max(targets)*1.2)
plt.legend()
plt.title('Deep Learning')
plt.grid(True)
plt.show()

comparison.reset_index().plot(kind='line', x='index', y=['prediction_value', 'actual_value'], 
                                  title = 'Deep Learning', grid = True, figsize = (12,8), ylim = [min(targets)*0.8, max(targets)*1.2])

'''
#analyze the result 
list_big_loss=[]
for i, r in comparison.iterrows():
    if abs(r['prediction_value'] - r['actual_value'])/r['actual_value'] > 0.1:
        list_big_loss.append(i)
test_data = pd.DataFrame(x_test, columns = B.feature_attributes)
test_data = test_data.iloc[list_big_loss].reset_index(drop=True)
test_data = pd.concat([pd.DataFrame(list_big_loss, columns=['index']), test_data], axis=1)
pd.set_option('display.max_columns', len(B.feature_attributes))
print(test_data)

#validate with new file

from Biogas_prediction_importing import import_data
     
data2= import_data(usecols='E,J,L,M,S,T,W,Y,AA,AC,AD,AS,AT,AU,BC,BD')
x_new = data2[B.feature_attributes]
y_new = data2[target_attributes]
X_new = standardscaler.transform(x_new)
model.evaluate(X_new, y_new)
            
'''

