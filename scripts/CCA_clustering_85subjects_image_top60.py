#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 06:15:55 2022

@author: xinxing
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:28:09 2022

@author: xinxing
"""

import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
import numpy as np
 
from sklearn.cluster import AgglomerativeClustering 


#data=pd.read_excel('/Users/xinxing/Documents/MDD/New data/MDD_test/01_31_2022/data/MDD_update_data.xlsx')
data_t=pd.read_excel('/Users/xinxing/Documents/MDD/New data/MDD_test/01_31_2022/data/MDD_data_new.xlsx')
data_t=data_t.iloc[7:,:]
top60=['N_Right-Thalamus-Proper',
 'Rpostcentral',
 'N_Left-AN_CCumbens-area',
 'N_Right-AN_CCumbens-area',
 'Linferiorparietal',
 'Lcaudalanteriorcingulate',
 'Rparacentral',
 'Linsula',
 'Lentorhinal',
 'Rsuperiorfrontal',
 'Lsuperiorparietal',
 'Lprecentral',
 'Lprecuneus',
 'Lposteriorcingulate',
 'Lbankssts',
 'Lfusiform',
 'Rcuneus',
 'N_Left-Thalamus-Proper',
 'Lmedialorbitofrontal',
 'N_Right-Pallidum',
 'Rcaudalanteriorcingulate',
 'Lfrontalpole',
 'Rrostralmiddlefrontal',
 'Lpostcentral',
 'Lcaudalmiddlefrontal',
 'Lparsorbitalis',
 'Rrostralanteriorcingulate',
 'Brain-Stem',
 'Linferiortemporal',
 'Lrostralanteriorcingulate',
 'Rmedialorbitofrontal',
 'Llateraloccipital',
 'N_Right-Putamen',
 'Listhmuscingulate',
 'Rparahippocampal',
 'Rtransversetemporal',
 'N_Right-Caudate',
 'N_Right-Amygdala',
 'Rparsorbitalis',
 'Rsuperiortemporal',
 'N_Left-Amygdala',
 'Rprecentral',
 'N_Left-Putamen',
 'Rinferiorparietal',
 'N_Right-choroid-plexus',
 'N_Right-VentralDC',
 'Rlateralorbitofrontal',
 'Rpericalcarine',
 'Rtemporalpole',
 'Lpericalcarine',
 'Lmiddletemporal',
 'Lsuperiortemporal',
 'Rbankssts',
 'N_Left-Caudate',
 'Lparsopercularis',
 'Rfusiform',
 'N_Left-VentralDC',
 'Rparstriangularis',
 'Rentorhinal',
 'Rprecuneus']

data=data_t[data_t.Status==0]

datalist=["XNAT","Age","Gender"]
'''
scorelist=['madrs_sum','lsc_score','ce_tleq','oc_tleq',
           'qids_score','rrs_total','shaps_score_1','shaps_score_2','bss_total',
           'sticsa_somatic','sticsa_cognitive']
'''

scorelist=['masq_gd_score','masq_ad_score','masq_aa_score','lsc_score','ce_tleq','ctq_total',
           'oc_tleq','rrs_total','shaps_score_2','bss_total',
           'sticsa_somatic','sticsa_cognitive','pss_score']

drop_list=['7TID','XNAT','Status','Age', 'Gender','Height','Weight','BMI','Race','Ethnicity',
           'Education','EmploymentStatus','HouseholdIncome']

classification_data=data.drop(datalist + scorelist+ drop_list, axis=1)
#classification_data=data[top60]

item1='N_Left-VentralDC'
item2='Rfusiform'

dfnew=data[datalist + scorelist]  
dfnew=dfnew.reset_index(drop=True)

newlist=[item1,item2]

datanew=data[newlist]

#y_new=datanew.Status
 
X_new=datanew


#normalized standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_new)
X_new=scaler.transform(X_new)
X_new=pd.DataFrame(X_new,columns=newlist)


#pca = PCA(n_components = 2) 
#X_principal = pca.fit_transform(X_new)
# Calculate the distance between each sample
#Z = hierarchy.linkage(X_principal, 'ward')
Z = hierarchy.linkage(X_new, 'ward')
 
# Set the colour of the cluster here:
hierarchy.set_link_color_palette(['r', 'b'])
 
# Make the dendrogram and give the colour above threshold
hierarchy.dendrogram(Z, color_threshold=6.5, above_threshold_color='grey')
 
# Add horizontal line.
plt.axhline(y=6.5, c='black', lw=2, linestyle='dashed')

#from scipy.cluster.hierarchy import fcluster
#d=shc.linkage(X_principal, method ='ward')

ac2 = AgglomerativeClustering(n_clusters = 2,compute_full_tree=True)

# Visualizing the clustering 
# Visualizing the clustering 
fig = plt.figure(figsize =(8, 8)) 

import mpl_toolkits.axisartist as axisartist
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

blue_circle = Line2D([0], [0], marker='o', color='w', label='Cluster 1',
                        markerfacecolor='b', markersize=15)
red_circle = Line2D([0], [0], marker='o', color='w', label='Cluster 2',
                        markerfacecolor='r', markersize=15)

#pop_a = mpatches.Circle((0,5),0.1, color='b', label='Cluster 1')
#pop_b = mpatches.Circle((2,5),0.1, color='r', label='Cluster 2')

ax=axisartist.Subplot(fig, 1,1,1)
fig.add_axes(ax)

ax.axis[:].set_visible(False)

ax.axis["x"]=ax.new_floating_axis(0,0)

ax.axis["y"]=ax.new_floating_axis(1,0)



labels=['cluster1', 'cluster2']
color=['b','r']

datat=X_new.to_numpy()

for i in range(datat.shape[0]):
    ax.scatter(datat[i,0], datat[i,1],  
           c = color[ac2.fit_predict(X_new)[i]], cmap ='rainbow')#,label=labels[ac2.fit_predict(X_new)[i]]) 

plt.axhline(y=0, xmin=-4, xmax=4,color='k')
plt.axvline(x=0,ymin=-2, ymax=6, color='k')

y=[-3,-2,-1,0,1,2,3]
yvalues = ['-3', '', '', '','', '','3'] 
ax.set_ylim(-3,3)

plt.yticks(y,yvalues)


x=[-3,-2,-1,0,1,2,3]
xvalues = ['-3', '', '', '','', '','3'] 
ax.set_xlim(-3,3)
plt.xticks(x,xvalues)
plt.axis('off')
ax.text(-3.5, 0.5, item1,fontsize=15)
ax.text(-0.5, 3.25, item2,fontsize=15)
#plt.xlabel(item1)
#plt.ylabel(item2)
plt.legend(handles=[blue_circle,red_circle])
plt.show()
    
#for i in range (42):
    
#    plt.annotate(z[i],(X_principal[i, 0], X_principal[i, 1]))





labels=ac2.fit_predict(X_new)

DX=pd.DataFrame(labels,columns=["label"])
df_concat = pd.concat([dfnew, DX], axis=1)

classification_t=classification_data.reset_index(drop=True)
df_classification = pd.concat([classification_t, DX], axis=1)

print(("total samples is: ") + str(df_concat.shape[0]))

print(("group 0 number is:  ") + str((df_concat.label==0).sum()))

print(("group 1 number is:  ") + str((df_concat.label==1).sum()))

#print(("group 2 number is:  ") + str((df_concat.label==2).sum()))

print(("Group 0 avg. age is: ") + str(df_concat[df_concat.label==0]["Age"].mean()))

print(("Group 1 avg. age is: ") + str(df_concat[df_concat.label==1]["Age"].mean()))

#print(("Group 2 avg. age is: ") + str(df_concat[df_concat.label==2]["Age"].mean()))



#print(("Group 0 avg. score is: ") + str(df_concat[df_concat.label==0][item].mean()))

#print(("Group 1 avg. score is: ") + str(df_concat[df_concat.label==1][item].mean()))

from scipy.stats import ttest_ind
stat, p = ttest_ind(df_concat[df_concat.label==0].Age,df_concat[df_concat.label==1].Age)
print("Age p-value is: "+str(p))

#stat_score, p_score = ttest_ind(df_concat[df_concat.label==0][item],df_concat[df_concat.label==1][item])
#print("Score p-value is:" +str(p_score) )

#stat, p = ttest_ind(df_concat[df_concat.label==0].Gender,df_concat[df_concat.label==1].Gender)
#print("Gender p-value is: "+ str(p))

import seaborn as sns

lut = dict(zip(set(labels), "brg"))

row_colors=pd.DataFrame(labels)[0].map(lut)

sns.clustermap(X_new,method='ward',row_colors=row_colors)
#plt.axvline(x=2,c='black', lw=2, linestyle='dashed')
plt.show() 

cluster0=df_concat[df_concat.label==0]
cluster1=df_concat[df_concat.label==1]


cluster0_score=cluster0[scorelist]
cluster1_score=cluster1[scorelist]

data1=pd.read_excel('/Users/xinxing/Documents/MDD/New data/MDD_test/01_31_2022/data/MDD_data_new.xlsx')

control=data1[data1.Status==1]
control=control[scorelist]
cnlabels = np.ones(48)+1
cn_DX=pd.DataFrame(cnlabels,columns=["label"])
#control_n=pd.concat([control, cn_DX], axis=1)

result0=cluster0_score.mean()-control.mean()
result1=cluster1_score.mean()-control.mean()

#cluster0.to_csv('/Users/xinxing/Documents/MDD/New data/MDD_test/May_20_2021/clusterlist_42/'+item+'_cluster0.csv',index=False)
#cluster1.to_csv('/Users/xinxing/Documents/MDD/New data/MDD_test/May_20_2021/clusterlist_42/'+item+'_cluster1.csv',index=False)

mri0=df_classification[df_classification.label==0]
mri1=df_classification[df_classification.label==1]

mri0=mri0.drop(['label'],axis=1)
mri1=mri1.drop(['label'],axis=1)

controlmri=data1[data1.Status==1]
controlmri=controlmri[classification_data.columns.to_list()]

mri_result0=mri0.mean()-controlmri.mean()
mri_result1=mri1.mean()-controlmri.mean()


barWidth = 0.25
fig = plt.subplots(figsize =(100, 50))

br1 = np.arange(len(mri_result0))
br2 = [x + barWidth for x in br1]
plt.bar(br1, mri_result0, color ='b', width = barWidth,
        edgecolor ='grey', label ='Cluster1')
plt.bar(br2, mri_result1, color ='r', width = barWidth,
        edgecolor ='grey', label ='Cluster2')

plt.ylabel('Imaging value difference', fontweight ='bold', fontsize = 35)
plt.xticks([r + barWidth for r in range(len(mri_result0))],classification_data.columns.to_list(),rotation = 75,fontweight ='bold', fontsize = 25)
plt.legend(fontsize=35)
plt.show()

######################################################################
#classification 
######################################################################


######################################################################
#classification Symptom Scores

######################################################################
#classification 
######################################################################


######################################################################
#classification Symptom Scores
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from scipy import interp
from sklearn.preprocessing import StandardScaler

symptom = df_concat.drop(['XNAT', 'Age', 'Gender'], axis=1)

X_symptom = symptom.drop('label',axis=1)
X_symptom_1=X_symptom
y_symptom = symptom.label

numFeature=X_symptom.shape[1]

scaler = StandardScaler()
scaler.fit(X_symptom)
X_symptom=scaler.transform(X_symptom)

importance=np.zeros((5,numFeature))
res=[]
f1=[]

j=0

clf=RandomForestClassifier(n_estimators=300, criterion='gini', min_samples_leaf=2, max_depth=10, bootstrap=True,max_features='auto',min_samples_split=10)
#clf = GradientBoostingClassifier(n_estimators=3600, min_samples_split=5, min_samples_leaf=8, max_features='auto',max_depth=50)
cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=9)
for train, test in cv.split(X_symptom, y_symptom):
    #sm= SMOTEENN(random_state=44)
    #X_res,y_res=sm.fit_sample(X[train],y[train])
    #probas_ = clf.fit(X_res, y_res).predict_proba(X[test])#The smote upsampling
    probas_ = clf.fit(X_symptom[train], y_symptom[train]).predict_proba(X_symptom[test]) #original
    accuracy=clf.predict(X_symptom[test])
    res.append(clf.score(X_symptom[test],y_symptom[test]))
    #print(('accuracy: ')+str(clf.score(X[test],y[test])))
    y_pred=clf.predict(X_symptom[test])
    f1.append(f1_score(y_symptom[test], y_pred, average='weighted'))
    #print(('F1 socre: ')+str(f1_score(y[test], y_pred, average='weighted')))
    importance[j,:]=clf.feature_importances_
        
    j += 1
    
b=np.mean(importance, axis=0)
feature_importances = pd.DataFrame(b,index = X_symptom_1.columns, columns=['importance']).sort_values('importance',ascending=False)    
print("The acc is: "+str(np.mean(res)))
print("The f1 is: " +str(np.mean(f1)))
print("The feature ranking is:")
print(feature_importances)


######################################################################
#classification MRI image
#normalized standardize features
X=df_classification.drop('label',axis=1)
X_1=X
y=df_classification.label

numFeature=X.shape[1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)



tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 250)
result= []
importance=np.zeros((500,numFeature))
res=[]
f1=[]


i=0
j=0 
for i in range(100):
    clf=RandomForestClassifier(n_estimators=300, criterion='gini', min_samples_leaf=2, max_depth=10, bootstrap=True,max_features='auto',min_samples_split=10)
    #clf = GradientBoostingClassifier(n_estimators=3600, min_samples_split=5, min_samples_leaf=8, max_features='auto',max_depth=50)
    cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=i)
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
print("The feature ranking is:")
print(feature_importances[:20])

top20=feature_importances[:20].T.columns.to_list()

X=df_classification.drop('label',axis=1)
X=X[top20]
X_1=X
y=df_classification.label

numFeature=X.shape[1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)



tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 250)
result= []
importance=np.zeros((500,numFeature))
res=[]
f1=[]


i=0
j=0 
for i in range(100):
    clf=RandomForestClassifier(n_estimators=300, criterion='gini', min_samples_leaf=2, max_depth=10, bootstrap=True,max_features='auto',min_samples_split=10)
    #clf = GradientBoostingClassifier(n_estimators=3600, min_samples_split=5, min_samples_leaf=8, max_features='auto',max_depth=50)
    cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=i)
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
print("The feature ranking is:")
print(feature_importances)