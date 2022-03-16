#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 21:19:20 2022

@author: xinxing
"""


import pandas as pd
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, wilcoxon
from chord import Chord
import operator
#sklearn imports
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler #used for 'Feature Scaling'
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Basic imports
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.model_selection import StratifiedKFold


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


data=pd.read_excel('/Users/xinxing/Documents/MDD/New data/MDD_test/01_31_2022/data/MDD_data_new.xlsx')
data=data.iloc[7:,:]
data=data.reset_index(drop=True)
drop_list=['7TID','XNAT','Status','Height','Weight','BMI','Race','Ethnicity','Gender',
           'Education','EmploymentStatus','HouseholdIncome','Age']


scorelist=['masq_gd_score','masq_ad_score','masq_aa_score','lsc_score','ce_tleq','ctq_total',
           'oc_tleq','rrs_total','shaps_score_2','bss_total',
           'sticsa_somatic','sticsa_cognitive','pss_score']

y=data.Status
X=data.drop(drop_list+scorelist,axis=1)

numFeature=X.shape[1]
#X=X.as_matrix()
y=y.ravel()
X_1=X
print(X_1.columns)

#normalized standardize features
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)


#clf=RandomForestClassifier(n_estimators=100,criterion='entropy')#,class_weight='balanced')
#clf=svm.SVC(kernel='rbf',C=1000, gamma=0.01,coef0=0.01,probability=True)
#cv = StratifiedKFold(n_splits=3)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 250)
result= []
importance=np.zeros((500,numFeature))
res=[]
f1=[]

i = 0
j=0 
for i in range(100):
    clf = RandomForestClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=8, max_features='auto',max_depth=50)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    for train, test in cv.split(X, y):
    
        #sm= SMOTEENN(random_state=44)
        #X_res,y_res=sm.fit_sample(X[train],y[train])
        #probas_ = clf.fit(X_res, y_res).predict_proba(X[test])#The smote upsampling
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test]) #original
        accuracy=clf.predict(X[test])
        res.append(clf.score(X[test],y[test]))
        #print(('accuracy: ')+str(clf.score(X[test],y[test])))
        y_pred=clf.predict(X[test])
        f1.append(f1_score(y[test], y_pred, average='weighted'))
        #print(('F1 socre: ')+str(f1_score(y[test], y_pred, average='weighted')))
        importance[j,:]=clf.feature_importances_
        
        j += 1
    i += 1
b=np.mean(importance, axis=0)
feature_importances = pd.DataFrame(b,index = X_1.columns, columns=['importance']).sort_values('importance',ascending=False)    
print("The acc is: "+str(np.mean(res)))
print("The f1 is: " +str(np.mean(f1)))
#print("The top 30 feature ranking is:")
#print(feature_importances[:30])
#print("The top 40 feature ranking is:")
#print(feature_importances[:40])



'''
###############################################################################
#top 30
###############################################################################
top30=feature_importances[:30].T.columns.to_list()
Xtop30=data[top30]
Xtop30_1=Xtop30
ytop30=data.Status

numFeature=Xtop30.shape[1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(Xtop30)
Xtop30=scaler.transform(Xtop30)



tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 250)
result= []
importancetop30=np.zeros((500,numFeature))
res=[]
f1=[]


i=0
j=0 
for i in range(100):
    clf=RandomForestClassifier(n_estimators=300, criterion='gini', min_samples_leaf=2, max_depth=10, bootstrap=True,max_features='auto',min_samples_split=10)
    #clf = GradientBoostingClassifier(n_estimators=3600, min_samples_split=5, min_samples_leaf=8, max_features='auto',max_depth=50)
    cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=i)
    for train, test in cv.split(Xtop30, ytop30):
    
        #sm= SMOTEENN(random_state=44)
        #X_res,y_res=sm.fit_sample(X[train],y[train])
        #probas_ = clf.fit(X_res, y_res).predict_proba(X[test])#The smote upsampling
        probas_ = clf.fit(Xtop30[train], ytop30[train]).predict_proba(Xtop30[test]) #original
        accuracy=clf.predict(Xtop30[test])
        res.append(clf.score(Xtop30[test],ytop30[test]))
        #print(('accuracy: ')+str(clf.score(X[test],y[test])))
        y_pred=clf.predict(Xtop30[test])
        f1.append(f1_score(ytop30[test], y_pred, average='weighted'))
        #print(('F1 socre: ')+str(f1_score(y[test], y_pred, average='weighted')))
        importancetop30[j,:]=clf.feature_importances_
        
        j += 1
    i += 1
b=np.mean(importancetop30, axis=0)
feature_importancestop30 = pd.DataFrame(b,index = Xtop30_1.columns, columns=['importance']).sort_values('importance',ascending=False)    
print("Top 30:")
print("The acc is: "+str(np.mean(res)))
print("The f1 is: " +str(np.mean(f1)))
print("The feature ranking is:")
print(feature_importancestop30)
print(" ")


###############################################################################
#top 40
###############################################################################
top40=feature_importances[:40].T.columns.to_list()
Xtop40=data[top40]
Xtop40_1=Xtop40
ytop40=data.Status

numFeature=Xtop40.shape[1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(Xtop40)
Xtop40=scaler.transform(Xtop40)



tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 250)
result= []
importancetop40=np.zeros((500,numFeature))
res=[]
f1=[]


i=0
j=0 
for i in range(100):
    clf=RandomForestClassifier(n_estimators=300, criterion='gini', min_samples_leaf=2, max_depth=10, bootstrap=True,max_features='auto',min_samples_split=10)
    #clf = GradientBoostingClassifier(n_estimators=3600, min_samples_split=5, min_samples_leaf=8, max_features='auto',max_depth=50)
    cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=i)
    for train, test in cv.split(Xtop40, ytop40):
    
        #sm= SMOTEENN(random_state=44)
        #X_res,y_res=sm.fit_sample(X[train],y[train])
        #probas_ = clf.fit(X_res, y_res).predict_proba(X[test])#The smote upsampling
        probas_ = clf.fit(Xtop40[train], ytop40[train]).predict_proba(Xtop40[test]) #original
        accuracy=clf.predict(Xtop40[test])
        res.append(clf.score(Xtop40[test],ytop40[test]))
        #print(('accuracy: ')+str(clf.score(X[test],y[test])))
        y_pred=clf.predict(Xtop40[test])
        f1.append(f1_score(ytop40[test], y_pred, average='weighted'))
        #print(('F1 socre: ')+str(f1_score(y[test], y_pred, average='weighted')))
        importancetop40[j,:]=clf.feature_importances_
        
        j += 1
    i += 1
b=np.mean(importancetop40, axis=0)
feature_importancestop40 = pd.DataFrame(b,index = Xtop40_1.columns, columns=['importance']).sort_values('importance',ascending=False)    
print("Top 40:")
print("The acc is: "+str(np.mean(res)))
print("The f1 is: " +str(np.mean(f1)))
print("The feature ranking is:")
print(feature_importancestop40)
print(" ")

###############################################################################
#top 50
###############################################################################
top50=feature_importances[:50].T.columns.to_list()
Xtop50=data[top50]
Xtop50_1=Xtop50
ytop50=data.Status

numFeature=Xtop50.shape[1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(Xtop50)
Xtop50=scaler.transform(Xtop50)



tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 250)
result= []
importancetop50=np.zeros((500,numFeature))
res=[]
f1=[]


i=0
j=0 
for i in range(100):
    clf=RandomForestClassifier(n_estimators=300, criterion='gini', min_samples_leaf=2, max_depth=10, bootstrap=True,max_features='auto',min_samples_split=10)
    #clf = GradientBoostingClassifier(n_estimators=3600, min_samples_split=5, min_samples_leaf=8, max_features='auto',max_depth=50)
    cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=i)
    for train, test in cv.split(Xtop50, ytop50):
    
        #sm= SMOTEENN(random_state=44)
        #X_res,y_res=sm.fit_sample(X[train],y[train])
        #probas_ = clf.fit(X_res, y_res).predict_proba(X[test])#The smote upsampling
        probas_ = clf.fit(Xtop50[train], ytop50[train]).predict_proba(Xtop50[test]) #original
        accuracy=clf.predict(Xtop50[test])
        res.append(clf.score(Xtop50[test],ytop50[test]))
        #print(('accuracy: ')+str(clf.score(X[test],y[test])))
        y_pred=clf.predict(Xtop50[test])
        f1.append(f1_score(ytop50[test], y_pred, average='weighted'))
        #print(('F1 socre: ')+str(f1_score(y[test], y_pred, average='weighted')))
        importancetop50[j,:]=clf.feature_importances_
        
        j += 1
    i += 1
b=np.mean(importancetop50, axis=0)
feature_importancestop50 = pd.DataFrame(b,index = Xtop50_1.columns, columns=['importance']).sort_values('importance',ascending=False)    
print("Top 50:")
print("The acc is: "+str(np.mean(res)))
print("The f1 is: " +str(np.mean(f1)))
print("The feature ranking is:")
print(feature_importancestop50)
print(" ")

###############################################################################
#top 60
###############################################################################
top60=feature_importances[:60].T.columns.to_list()
Xtop60=data[top60]
Xtop60_1=Xtop60
ytop60=data.Status

numFeature=Xtop60.shape[1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(Xtop60)
Xtop60=scaler.transform(Xtop60)



tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 250)
result= []
importancetop60=np.zeros((500,numFeature))
res=[]
f1=[]


i=0
j=0 
for i in range(100):
    clf=RandomForestClassifier(n_estimators=300, criterion='gini', min_samples_leaf=2, max_depth=10, bootstrap=True,max_features='auto',min_samples_split=10)
    #clf = GradientBoostingClassifier(n_estimators=3600, min_samples_split=5, min_samples_leaf=8, max_features='auto',max_depth=50)
    cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=i)
    for train, test in cv.split(Xtop60, ytop60):
    
        #sm= SMOTEENN(random_state=44)
        #X_res,y_res=sm.fit_sample(X[train],y[train])
        #probas_ = clf.fit(X_res, y_res).predict_proba(X[test])#The smote upsampling
        probas_ = clf.fit(Xtop60[train], ytop60[train]).predict_proba(Xtop60[test]) #original
        accuracy=clf.predict(Xtop60[test])
        res.append(clf.score(Xtop60[test],ytop60[test]))
        #print(('accuracy: ')+str(clf.score(X[test],y[test])))
        y_pred=clf.predict(Xtop60[test])
        f1.append(f1_score(ytop60[test], y_pred, average='weighted'))
        #print(('F1 socre: ')+str(f1_score(y[test], y_pred, average='weighted')))
        importancetop60[j,:]=clf.feature_importances_
        
        j += 1
    i += 1
b=np.mean(importancetop60, axis=0)
feature_importancestop60 = pd.DataFrame(b,index = Xtop60_1.columns, columns=['importance']).sort_values('importance',ascending=False)    
print("Top 60:")
print("The acc is: "+str(np.mean(res)))
print("The f1 is: " +str(np.mean(f1)))
print("The feature ranking is:")
print(feature_importancestop60)
print(" ")
'''
###############################################################################
#top 70
###############################################################################
top70=feature_importances[:70].T.columns.to_list()
Xtop70=data[top70]
Xtop70_1=Xtop70
ytop70=data.Status

numFeature=Xtop70.shape[1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(Xtop70)
Xtop70=scaler.transform(Xtop70)



tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 250)
result= []
importancetop70=np.zeros((500,numFeature))
res=[]
f1=[]


i=0
j=0 
for i in range(100):
    clf=RandomForestClassifier(n_estimators=300, criterion='gini', min_samples_leaf=2, max_depth=10, bootstrap=True,max_features='auto',min_samples_split=10)
    #clf = GradientBoostingClassifier(n_estimators=3600, min_samples_split=5, min_samples_leaf=8, max_features='auto',max_depth=50)
    cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=i)
    for train, test in cv.split(Xtop70, ytop70):
    
        #sm= SMOTEENN(random_state=44)
        #X_res,y_res=sm.fit_sample(X[train],y[train])
        #probas_ = clf.fit(X_res, y_res).predict_proba(X[test])#The smote upsampling
        probas_ = clf.fit(Xtop70[train], ytop70[train]).predict_proba(Xtop70[test]) #original
        accuracy=clf.predict(Xtop70[test])
        res.append(clf.score(Xtop70[test],ytop70[test]))
        #print(('accuracy: ')+str(clf.score(X[test],y[test])))
        y_pred=clf.predict(Xtop70[test])
        f1.append(f1_score(ytop70[test], y_pred, average='weighted'))
        #print(('F1 socre: ')+str(f1_score(y[test], y_pred, average='weighted')))
        importancetop70[j,:]=clf.feature_importances_
        
        j += 1
    i += 1
b=np.mean(importancetop70, axis=0)
feature_importancestop70 = pd.DataFrame(b,index = Xtop70_1.columns, columns=['importance']).sort_values('importance',ascending=False)    
print("Top 70:")
print("The acc is: "+str(np.mean(res)))
print("The f1 is: " +str(np.mean(f1)))
print("The feature ranking is:")
print(feature_importancestop70)
print(" ")