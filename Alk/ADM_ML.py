# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:08:32 2021

@author: sujin
"""

import pandas as pd
import numpy as np
import sys

path = 'DAT_Seasonal sampling.xlsx'

data = pd.read_excel(path, sheet_name = list(range(1,7)), header = 2 ,usecols = "A:BP")
sys.path.append('C:/Users/cherr/Desktop/git/study/Biogas production')
sys.path.append('C:/Users/cherr/Desktop/git/study/functions')
sys.path.append(r'C:\Users\cherr\Desktop\git\study\www')



##data preprocessing
site = ["BB", "GB", "GH", "SR", "SC", "SBK"]
df_list = []
for i, name in zip(range(1,7), site):
    data[name] = data.pop(i)
    data[name].drop(0, inplace = True)
    data[name] = data[name][data[name]['Season'] != 'Winter 2020']
    data[name].dropna(subset =['Season'], inplace = True)
    data[name].drop(['Day', 'Date'], axis = 1, inplace = True)
    data[name]['Sample type'].fillna(method = 'ffill', inplace = True)
    data[name].reset_index(drop=True, inplace = True)
    data[name]['site'] = name
    df_list.append(data[name])
    for j in data[name].columns:
        if 'Unnamed' in j:
            data[name].drop(j, axis = 1, inplace = True)

##concate datasets & drop na values (data preprocessing)
total = pd.concat(df_list, ignore_index = True)
total.iloc[:,2:34] = total.iloc[:,2:34].astype('float')
total['Alkalinity'] = np.where((total.Alkalinity.isna()) & (total.pH < 4.5), 0, total['Alkalinity'])
na_val= np.where((total.Alkalinity.isna()) | (total.Protein.isna()))
final = total.dropna(subset = ['Alkalinity', 'Protein', 'F'])

final = final.drop(total.iloc[:, 30:36], axis = 1)
final.reset_index(inplace = True, drop = True)
final = final.drop(np.where((final.K == 0) & (final.Ca == 0) & (final.Mg == 0) & (final.NH4 == 0) & (final.Na == 0))[0])
final.reset_index(inplace = True, drop = True)
final = final.drop(np.where(final.Alkalinity <= 100)[0])
final.reset_index(inplace = True, drop = True)
final['TVFA + EtOH'] = final['TVFA'] + final['EtOH']
#final = final.drop([66])
final.rename(columns = {'Alkalinity' : 'TA', 'Sample type' : 'Sample', 'site' : 'Site'}, inplace = True)
final.loc[final['Sample'] == 'Methanogenic Reactor', 'Sample'] = 'M'
final.loc[final['Sample'] == 'Primary sludge + Secondary sludge', 'Sample'] = '1S + 2S'
final.loc[final['Sample'] == 'Primary Sludge', 'Sample'] = '1S'
final.loc[final['Sample'] == 'Secondary Sludge', 'Sample'] = '2S'

season =[]
for i in range(0, len(final)):
    season.append(final.Season.iloc[i].split(' ')[0])
final['Season'] = season

met_final = final[final['Sample'] == 'M']

##nier sample
path_nier = 'NIER.xlsx'
nier = pd.read_excel(path_nier, index_col = 0)
nier.dropna(inplace = True)

##concat met_final, nier --> whole sample 
whole = pd.concat([met_final, nier])
whole['IA/PA'] = whole['IA']/whole['PA']
whole['TVFA + EtOH'] = whole['TVFA + EtOH'] * 1000
whole['TVFA + EtOH/TA'] = whole['TVFA + EtOH']/whole['TA']
whole = whole.drop(np.where(whole['IA/PA'] == float('inf'))[0])
whole = whole.drop(np.where(whole['TVFA + EtOH/TA'] == float('inf'))[0])

#whole.drop(['Sample'], axis = 1, inplace = True)
whole = pd.get_dummies(whole, dtype = 'float')
whole.drop(whole.iloc[:, 29:39], axis = 1, inplace = True)


##target, features
target_nom = ['IA/PA', 'TA', 'IA', 'PA', 'TVFA', 'TVFA + EtOH', 'TVFA + EtOH/TA', ]

    #1) only methanogenic reactor (stability indicator)
target_attributes = 'TVFA + EtOH/TA'

    #2) overall reactors (alkalinity prediction)
target_attributes = 'IA'

target = whole[target_attributes]
features = whole.drop(target_nom, axis = 1)

 

## feature selection
    #1) based on correlation matrix
corr_matrix = pd.concat([features, target], axis = 1).corr()
cm = corr_matrix[target_attributes]
print(cm.sort_values(ascending=False))
feature_attributes =cm[(abs(cm)>float(0.2))&(cm.index!=target_attributes)].index
feature_attributes = feature_attributes.tolist()
print("feature attributes : ", feature_attributes)
features = whole[feature_attributes]

    #2) appending season
season =[]
for i in range(0, len(features)):
    season.append(features.Season.iloc[i].split(' ')[0])
features['season'] = season
features.drop(['Season', 'site'], axis = 1, inplace = True)
features = pd.get_dummies(features, dtype = 'float')


##data visualization
import seaborn as sns
final["Alk_cut"] = pd.cut(final['Alkalinity'], bins=[500, 1000, 2000, 3000, 4000, 5000, 10000,15000,20000,np.inf], )
attributes = feature_attributes + ['Alk_cut']
final1 = final[attributes]
sns.pairplot(final1, hue = 'Alk_cut', palette = 'rocket_r')
for i in ['Lipid', 'EtOH', 'HBu','iHBu', 'iHCa', 'HCa']:
    feature_attributes.remove(i)



##model selection
from model import ML, build_model
from functions import comparison, evaluate, feature_importance_plot, permutation_importance_plot
from functions_2 import display_scores, comparison2, rmse, mape, split_data, plot_learning_curves, plot_learning_curves_epoch
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = split_data(features, target, 0.3)

    #1) DNN
from tensorflow import keras
model = keras.models.Sequential()
model.add(keras.layers.Input(shape = [len(features.columns)]))
model.add(keras.layers.Dense(40, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(80, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(1))
model.compile(loss='mean_squared_error', metrics=[keras.metrics.RootMeanSquaredError(name='rmse')] )
history = model.fit(X_train, y_train, epochs = 500, verbose = 0, validation_split = 0.3)
rmse = pd.DataFrame(history.history)[['rmse', 'val_rmse']]
rmse.plot(figsize = (8,5))
import matplotlib.pyplot as plt
plt.title('Deep neural network')


    #2) Ensemble model
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import xgboost as xgb
forest_reg = RandomForestRegressor()
xgb_reg= xgb.XGBRFRegressor()

xgb_reg = xgb.XGBRFRegressor(reg_alpha = 0.05, eta = 0.05, max_depth = 8, gamma = 0)
ada_reg = AdaBoostRegressor()
##parameter

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)
dtrain = xgb.DMatrix(X_train, y_train, feature_names = features.columns)
dval = xgb.DMatrix(X_val, y_val)
dtest = xgb.DMatrix(X_test, y_test)
params = xgb_reg.get_xgb_params()
params['learning_rate'] = 0.01
params['tree_method'] = 'auto'
xgb_train = xgb.train(params, dtrain, num_boost_round =200, evals = [(dtrain,'train'), (dval,'eval')] ,early_stopping_rounds=30, verbose_eval = 20)
xgb_cv = xgb.cv(params, dtrain, num_boost_round = 200,  early_stopping_rounds = 35, verbose_eval = None )


    #3) regression model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
lin_reg = LinearRegression()
ridge_reg = Ridge()
lasso_reg = Lasso()



## model fitting, visualization
model =xgb_reg
ml = ML(model, features, target)
ml.model_fit(X_train, y_train, early_stopping=(True))
compare, pred = ml.predict(X_train, y_train)
ml.cross_val(X_train, y_train)
importance = ml.permutation_importance_plot(X_train, y_train)
plot_learning_curves(model, X_train, y_train)


## feature importance (SHAP)
import shap
explainer = shap.TreeExplainer(model, )
shap_val = explainer(X_train)
    #1) summary plot : mean shap value of each features
shap.summary_plot(shap_val, X_train, plot_type = 'bar')
    #2) whether each feature impact positively or negatively on the model output
shap.plots.waterfall(shap_val[0])
    #3) to see each individual feature on the general model output (SHAP determines the color automatically, that has the most irrelevant impact on the feature)
'''
feature = 'Na', 'NH4', 'COD', 'K'
'''
feature = 'Na'
shap.plots.scatter(shap_val[:, feature], color = shap_val) 
    #*4) to see the positive/negative impact of degree of feature value on SHAP  (RECOMMENDED!)
shap.plots.beeswarm(shap_val)
## new features
new_feature_attributes = importance[importance > float(0.01)].index.tolist()
features = features[new_feature_attributes]












#hyperparameter tuning
#RandomizedSearch CV
from sklearn.model_selection import RandomizedSearchCV

booster = ['gbtree', 'dart']
eta = [0.1,0.2, 0.3, 0.4,0.5]
max_depth = [2,4,6]
gamma = [0, 2, 4, 6]
min_child_weight = [1,3,5,7,10]
max_delta_step = [0,1,2,3,5]
reg_alpha = [0, 0.05, 0.5, 5, 10]
tree_method = ['auto', 'exact', 'approx',]
random_grid = {'eta':eta, 'gamma':gamma, 'reg_alpha':reg_alpha, 'tree_method': tree_method}
xgb_random = RandomizedSearchCV(estimator = xgb_reg, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, n_jobs = -1)
xgb_random.fit(X_train, y_train, verbose = True)
print(xgb_random.best_params_)
random_xgb = xgb_random.best_estimator_
random_accuracy = evaluate(random_xgb, X_train, y_train)
display_scores(random_xgb, X_train, y_train)
plot_learning_curves(random_xgb, X_train, y_train)
permutation_importance_plot(random_xgb, X_train, y_train, features)

#early stopping


# Create the parameter grid based on the results of random search 
from sklearn.model_selection import GridSearchCV
param_grid = {
    'reg_alpha': [0.02,0.05,0.08,0.1],
    'max_depth': [4,6,8,10],
    'gamma': [0,1,2,3,4],
    'eta': [0.05,0.1,0.2,0.3],
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = xgb_reg, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_xgb = grid_search.best_estimator_
grid_accuracy = evaluate(grid_xgb, X_train, y_train)

grid_pred = grid_xgb.predict(X_train)
comparison2(grid_pred, y_train, title = 'Alkalinity prediction(grid search)')
comparison(grid_pred, y_train, title = 'Alkalinity prediction(grid search)')
display_scores(grid_xgb, X_train, y_train)
plot_learning_curves(grid_xgb, X_train, y_train)

model = grid_xgb



# parameter tuning
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 300, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#evaluate with new model
base_forest = RandomForestRegressor()
base_forest.fit(X_train, y_train)
base_accuracy = evaluate(forest_reg, X_train, y_train)
best_forest = rf_random.best_estimator_
best_accuracy = evaluate(best_forest, X_train, y_train)

#cross validation
print("cross validation score of random grid model\n")



from functions_2 import comparison2, plot_learning_curves
best_pred = rf_random.best_estimator_.predict(X_train)
comparison2(best_pred, y_train, title = 'Alkalinity prediction(random search)')
comparison(best_pred, y_train, title = 'Alkalinity prediction(random search)')


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [15,20,25,30],
    'max_features': ['sqrt'],
    'min_samples_leaf': [3,4,5],
    'min_samples_split': [7,10,12],
    'n_estimators': [80,90,100,110,120,130]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_forest = grid_search.best_estimator_
grid_accuracy = evaluate(grid_forest, X_train, y_train)

grid_pred = grid_forest.predict(X_train)
comparison2(grid_pred, y_train, title = 'Alkalinity prediction(grid search)')
compare = comparison(grid_pred, y_train, title = 'Alkalinity prediction(grid search)')

#cross validation
print("cross validation score of gridsearch model\n")
display_scores(grid_forest, X_train, y_train)










model =xgb_reg
test_pred = model.predict(X_test)
comparison2(test_pred, y_test, title = 'Alkalinity prediction (grid search)')
comparison(test_pred, y_test, title = 'Alkalinity prediction (grid search)')
print("final model r2 :", model.score(X_test, y_test))
print("feature importances ", model.feature_importances_)
feature_importance_plot(model, features)
