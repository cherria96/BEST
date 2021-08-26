# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:53:05 2021

@author: sujin
"""

import pandas as pd
import numpy as np
import sys

path = 'C:/Users/cherr/Desktop/git/study/www/NIER_FW Sites.xlsx'
data = pd.read_excel(path, sheet_name = list(range(2,11)), header = 1)
item = ["pH, Alk", "COD", "Protein", "Solids", "TC", "Lipid", "VFAs", "Anion", "Cation"]

for i, name in zip(range(2,11), item):
    data[name] = data.pop(i)
    data[name].dropna(subset = ['Site', 'Sample'], inplace = True)
    #data[name].set_index([data[name].Site, data[name].Sample], inplace = True, drop = True)


#1) pH, Alk
data['pH'] = data['pH, Alk'].iloc[:, 2:5]
data['Alkalinity'] = data['pH, Alk'].iloc[:,10:13]
data['pH, Alk'] = pd.concat([data['pH'], data['Alkalinity']], axis = 1)
data['pH, Alk'].rename(columns = {"Unnamed: 4":"pH"}, inplace = True)
for i in ['pH', 'Alkalinity']:
    del data[i]

#2) COD
data['COD'] = data['COD'].iloc[:, [10,18]]
data['COD'].rename(columns = {'Ave (g/L)' : 'COD', 'Ave (g/L).1': 'sCOD'}, inplace = True)

#3) Protein
data['Protein'] = data['Protein'].iloc[:, [10,18,24]]
data['Protein'].rename(columns = {'Ave (g/L)' : 'TKN', 'Ave (g/L).1': 'TAN', 'Ave (g/L).2' : 'Protein'}, inplace = True)

#4) Solids
data['Solids'] = data['Solids'].iloc[:, [18,20,22,38,40,42]]
data['Solids'].rename(columns = {'TS Ave.' : 'TS', 'VS Ave.': 'VS', 'FS Ave.' : 'FS', 'TS Ave..1' : 'TSS', 
                                 'VS Ave..1': 'VSS', 'FS Ave..1' : 'FSS'}, inplace = True)

#5) TC
data['TC'] = data['TC'].iloc[:, [10]]
data['TC'].rename(columns = {'Ave (g/L)' : 'TC'}, inplace = True)

#6) Lipid 
data['Lipid'] = data['Lipid'].iloc[:,[12]]
data['Lipid'].rename(columns = {'Ave. (g/L)' : 'Lipid'}, inplace = True)

#7) VFAs
data['VFAs'] = data['VFAs'].iloc[:,[50,52,54,56,58,60,62,64,66,68,70]]
data['VFAs'].rename(columns = {'TVFA.2' : 'TVFA', 'TVFA + EtOH.2' : 'TVFA + EtOH'}, inplace = True)

#8) Anion 
data['Anion'] = data['Anion'].iloc[:, [10,16,22,28,34,40]]
data['Anion'].rename(columns = {'Ave (g/L)' : 'F', 'Ave (g/L).1': 'Cl', 'Ave (g/L).2' : 'NO2', 
                                'Ave (g/L).3' : 'NO3', 'Ave (g/L).4': 'PO4', 'Ave (g/L).5' : 'SO4'}, inplace = True)

#9) Cation
data['Cation'] = data['Cation'].iloc[:, [10,16,22,28,34]]
data['Cation'].rename(columns = {'Ave (g/L)' : 'Na', 'Ave (g/L).1': 'NH4', 'Ave (g/L).2' : 'K', 
                                'Ave (g/L).3' : 'Mg', 'Ave (g/L).4': 'Ca'}, inplace = True)


item_list = []
for name in item:
    item_list.append(data[name])
nier = pd.concat(item_list, axis =1, join = 'inner')
nier['IA/PA'] = nier['IA']/nier['PA']
nier.to_excel('nier.xlsx')
