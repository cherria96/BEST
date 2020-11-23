# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:10:54 2020

@author: sujin
"""
import Biogas_prediction_excel_data as B
import pandas as pd
import numpy as np

dataset = B.data1
target_attributes = B.target_attributes
corr_value = B.corr_value

targets = dataset[target_attributes]
features = dataset[B.feature_attributes]

#train, test data split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

def plot_learning_curves(model, X, y):
    '''training plot along with the size of training set'''
    import matplotlib.pyplot as plt

    x_train, x_test, y_train, y_test =train_test_split(X,y,test_size=0.2, random_state= 42)
    standardscaler= StandardScaler()
    X_train = standardscaler.fit_transform(x_train)
    X_test = standardscaler.transform(x_test)
    train_errors, val_errors = [],[]
    for m in range(1, len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_test, y_test_predict))
    plt.plot(np.sqrt(train_errors),'r-+', linewidth = 2, label = 'training set')
    plt.plot(np.sqrt(val_errors), 'b-', linewidth = 3, label = 'test set')
    plt.legend()
    plt.xlabel('size of training set')
    plt.ylabel('RMSE')
    plt.ylim(0,20000)
    plt.title(str(model))

#scaling data
x_train, x_test, y_train, y_test =train_test_split(features, targets,test_size=0.2, random_state= 42)
standardscaler= StandardScaler()
X_train = standardscaler.fit_transform(x_train)
X_test = standardscaler.transform(x_test)

'''
X_poly_train = PolynomialFeatures(degree=len(B.feature_attributes)).fit_transform(X_train)
X_poly_test = PolynomialFeatures(degree=len(B.feature_attributes)).fit(X_test)
'''
#Linear Regression model fitting
print('model #1 : Linear regression')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, features, targets)
lin_pred = lin_reg.predict(X_train)
print('Equation : y=', end='')
for i in range(len(B.feature_attributes)):
    print(lin_reg.coef_[i], '*', features.columns[i], end='')
    if i < len(B.feature_attributes)-1:
        print('+', end='')
    else:
        print()
lin_mse = mean_squared_error(y_train, lin_pred)
lin_rmse = np.sqrt(lin_mse)
print('regression rmse:', lin_rmse)
print()

#SGD regression model fitting
print('model #2 : SGD regression')
from sklearn.linear_model import SGDRegressor
epoch = input ('epoch: ')
lr = input('learning rate: ')
sgd_reg = SGDRegressor(max_iter= int(epoch), tol = 0.001, penalty = None, eta0=float(lr))
sgd_reg.fit(X_train,y_train)
sgd_pred = sgd_reg.predict(X_train)
sgd_mse = mean_squared_error(y_train, sgd_pred)
sgd_rmse = np.sqrt(sgd_mse)
print('SGD regression rmse:', sgd_rmse)

#Decision Tree Regresor model fitting
print('model #2 : Decision Tree Regression')
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
tree_pred = tree_reg.predict(X_train)
tree_mse = mean_squared_error(y_train, tree_pred)
tree_rmse = np.sqrt(tree_mse)
print('Decision tree regression rmse:', tree_rmse)
print()

#RandomForest tree Regressor model fitting
print('model #3 : RandomForest Tree Regression')
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)
forest_pred = forest_reg.predict(X_train)
forest_mse = mean_squared_error(y_train, forest_pred)
forest_rmse = np.sqrt(forest_mse)
print('RandomForest tree regression rmse:', forest_rmse)
print()


#SVM model fitting
print('model #4 : SVM')
from sklearn.svm import SVR
sv_reg = SVR()
sv_reg.fit(X_train, y_train)
sv_pred = sv_reg.predict(X_train)
sv_mse = mean_squared_error(y_train, sv_pred)
sv_rmse = np.sqrt(sv_mse)
print('Support vector regression rmse:', sv_rmse)
print()

#Evaluation _cross validation
from sklearn.model_selection import cross_val_score

tree_score = cross_val_score(tree_reg, X_train, y_train, scoring = 'neg_mean_squared_error', cv = 10)
tree_rmse_score = np.sqrt(-tree_score)
lin_score = cross_val_score(lin_reg, X_train, y_train, scoring = 'neg_mean_squared_error', cv = 10)
lin_rmse_score = np.sqrt(-lin_score)
sgd_score = cross_val_score(sgd_reg, X_train, y_train, scoring= 'neg_mean_squared_error', cv = 10)
sgd_rmse_score = np.sqrt(-sgd_score)
forest_score = cross_val_score(forest_reg, X_train, y_train, scoring = 'neg_mean_squared_error', cv = 10)
forest_rmse_score = np.sqrt(-forest_score)
#sv_score = cross_val_score(sv_reg, X_train, y_train, scoring = 'neg_mean_squared_error', cv = 10)
#sv_rmse_score = np.sqrt(-sv_score)
print('** rmse result from cross validation**')
print('linear reg : ', lin_rmse_score.mean())
print('sgd reg : ', sgd_rmse_score.mean())
print('decision tree reg : ', tree_rmse_score.mean())
print('randomforest tree reg : ', forest_rmse_score.mean())
#print('support vector reg : ', sv_rmse_score.mean())
print()

#visualization of the result

model_list = [lin_reg, sgd_reg, tree_reg, forest_reg]
for model in model_list:
    comparison = pd.DataFrame(model.predict(X_test), columns= ['prediction_value'] )
    comparison['actual_value'] = pd.DataFrame(y_test.values)
    plt.figure(figsize = (12,8))
    plt.plot(comparison.index, comparison[['prediction_value']], label = 'prediction_value')
    plt.scatter(comparison.index, comparison[['actual_value']], color='orange', label = 'actual_value')
    plt.xlabel('index')
    plt.ylim(min(targets)*0.8, max(targets)*1.2)
    plt.legend()
    plt.title(str(model))
    plt.grid(True)
    plt.show()
    
    comparison.reset_index().plot(kind='line', x='index', y=['prediction_value', 'actual_value'], ylabel = target_attributes,
                                  title = str(model), grid = True, figsize = (12,8), ylim = [min(targets)*0.8, max(targets)*1.2] )
    
#   plot_learning_curves(model, features, targets)


'''
test_color = 'r'
train_color = 'b'
for feature, target in zip(X_test[:,0], y_test):
    plt.scatter( feature, target, color=test_color, )
    plt.xlabel(features.columns[0])
    plt.ylabel(target_attributes)
for feature, target in zip(X_train[:,0], y_train):
    plt.scatter( feature, target, color=train_color ) 
'''  


