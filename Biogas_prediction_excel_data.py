# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 01:08:37 2020

@author: sujineeda
"""
#-*-Encoding: UTF-8 -*-#

import pandas as pd
import os
import numpy as np

#import data
from tkinter import filedialog
filename = filedialog.askopenfilename()
sheet_name = []
while True:
    new_sheet_name = input('sheet name : ')
    if new_sheet_name == '':
        break
    else: 
        sheet_name.append(new_sheet_name)

'''
SBK data 이용 
엑셀 데이터에서 바로 가져와서 엑셀 데이터 column, sheet name 확인해야 됨
'''
        
usecols = 'E,I,L,M,S,T,X,Y,Z,AC,AD,AM,AN,AO,BC,BD'    
data = pd.read_excel(filename, sheet_name = sheet_name, usecols= usecols,
                     header = 2, index_col = 0, 
                     na_values= ['',' - ',0] )

'''
column 이름 한글로 된거 Unnamed 로 표시되어 바꿔줌 
'''

# Converting each sheet to dataframe
for i in sheet_name:
    
    data[i].drop(data[i].index[range(0,5)], inplace = True)
    data[i].rename(columns={'DIG A': 'PS_vol', 'TS(%).1': 'PS_TS(%)', 'VS(%).1':'PS_VS(%)',
                            'Unnamed: 23':'Treated FW', 'Unnamed: 24':'Grit ratio', 'DIG A.1':'FWW_vol', 
                            'TS(%).2':'FWW_TS(%)', 'VS(%).2':'FWW_VS(%)', 'TS(%).3':'Dig_TS(%)','VS(%).3':'Dig_VS(%)',
                            'Nm3':'Biogas'}, inplace = True)
    data[i].dropna(inplace=True)
    data[i].drop_duplicates(['PS_TS(%)'], inplace=True)
    
data_2019 = data['Process Daily 2019']
data_2018 = data['Process Daily 2018']    

# Merge 2018, 2019 data
data1 = pd.concat([data_2019, data_2018], ignore_index =True)

print(data1.columns)

# Set the target attribute
target_attributes=input('target_attribute:')

corr_matrix = data1.corr()
cm = corr_matrix[target_attributes]
print(cm.sort_values(ascending=False))

#data visualization
import matplotlib.pyplot as plt
data1.hist(bins=50, figsize=(20,15))
plt.show()

# Set the features attribute by giving lower limit on correlation value
corr_value=input('corr_value:')

feature_attributes =cm[(cm>float(corr_value))&(cm.index!=target_attributes)].index
feature_attributes = feature_attributes.tolist()

# Pairplot between features
import seaborn as sns
data1['target_cut'] = pd.cut(data1[target_attributes], bins=[0,25000,30000,35000,np.inf], labels=[2.5,3,3.5,4])
attributes = feature_attributes + ['target_cut']
data2 = data1[attributes]
sns.pairplot(data2, hue = 'target_cut', palette = 'rocket_r')

print('feature_attributes : ', feature_attributes)

# Add more features or Delete more features
while True:
    added_features = input('add more features? :')
    if added_features == '':
        break
    else:
        feature_attributes += [added_features]

print('feature_attributes : ', feature_attributes)

while True:
    del_features = input('delete features? :')
    if del_features == '':
        break
    else:
        feature_attributes.remove(del_features)
        

print('feature_attributes : ', feature_attributes)
  
