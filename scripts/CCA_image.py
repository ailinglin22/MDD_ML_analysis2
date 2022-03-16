#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 00:11:54 2022

@author: xinxing
"""

import pandas as pd
import boto


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import statsmodels.api as sm
import seaborn as sns


data=pd.read_excel('/Users/xinxing/Documents/MDD/New data/MDD_test/01_31_2022/data/MDD_data_new.xlsx')


data=data.iloc[7:,:]
#data_t=data[data.Status==0]
#data_t=data_t.iloc[7:,:]




drop_list=['7TID','XNAT','Status','Age', 'Gender','Height','Weight','BMI','Race','Ethnicity',
           'Education','EmploymentStatus','HouseholdIncome']


scorelist=['masq_gd_score','masq_ad_score','masq_aa_score','lsc_score','ce_tleq',
           'ctq_total','oc_tleq','rrs_total','shaps_score_2','bss_total',
           'sticsa_somatic','sticsa_cognitive','pss_score']

df=data.drop(drop_list,axis=1)
#df=data_t.drop(drop_list,axis=1)

top20=['N_Right-Thalamus-Proper', 'N_Left-AN_CCumbens-area', 'Rpostcentral', 'N_Right-AN_CCumbens-area',
 'Linferiorparietal', 'Linsula', 'Rparacentral', 'Lentorhinal',
 'N_Right-Pallidum', 'Rsuperiorfrontal','Lprecuneus','Lsuperiorparietal',
 'N_Left-Thalamus-Proper', 'Lcaudalanteriorcingulate', 'Lprecentral','Rcuneus',
 'Lparsorbitalis','Lcaudalmiddlefrontal','Lfusiform','Rrostralmiddlefrontal']

top30=['N_Right-Thalamus-Proper','N_Left-AN_CCumbens-area', 'Rpostcentral','N_Right-AN_CCumbens-area',
 'Linferiorparietal','Linsula','Rparacentral','Lentorhinal',
 'N_Right-Pallidum','Rsuperiorfrontal','Lprecuneus','Lsuperiorparietal',
 'N_Left-Thalamus-Proper','Lcaudalanteriorcingulate','Lprecentral','Rcuneus',
 'Lparsorbitalis','Lcaudalmiddlefrontal','Lfusiform','Rrostralmiddlefrontal',
 'Lmedialorbitofrontal','Lbankssts','Lpostcentral','N_Left-Caudate',
 'N_Left-VentralDC','Rmedialorbitofrontal','Rparahippocampal','N_Left-Amygdala',
 'N_Right-Caudate','Rcaudalanteriorcingulate']


top40=['N_Right-Thalamus-Proper', 'Rpostcentral', 'N_Left-AN_CCumbens-area', 'N_Right-AN_CCumbens-area',
 'Linferiorparietal', 'Lcaudalanteriorcingulate', 'Rparacentral','Linsula',
 'Lentorhinal', 'Rsuperiorfrontal','Lsuperiorparietal','Lprecentral',
 'Lprecuneus', 'Lposteriorcingulate', 'Lbankssts', 'Lfusiform',
 'Rcuneus', 'N_Left-Thalamus-Proper', 'Lmedialorbitofrontal','N_Right-Pallidum',
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
 'Rsuperiortemporal']

top50=['N_Right-Thalamus-Proper',
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
 'Lpericalcarine']

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

Y=df[scorelist]
Y_mc = (Y-Y.mean())/(Y.std())

#X=df.drop(scorelist, axis=1)
X=df[top30]
X_mc = (X-X.mean())/(X.std())


index=X.columns


from sklearn.cross_decomposition import CCA

# Instantiate the Canonical Correlation Analysis with 2 components
my_cca = CCA(n_components=2)

# Fit the model
my_cca.fit(X_mc, Y_mc)

X_c, Y_c = my_cca.transform(X_mc, Y_mc)


cc_res_X = pd.DataFrame({"CCX_1":X_c[:, 0],
                       "CCX_2":X_c[:, 1]
                      })
X_mc_1=X_mc.reset_index(drop=True)

cc_res_X_data=pd.concat([cc_res_X, X_mc_1], axis=1)

cc_res_Y = pd.DataFrame({
                       "CCY_1":Y_c[:, 0],
                       "CCY_2":Y_c[:, 1],
                       "masq_gd_score":Y_mc.masq_gd_score,
                       "masq_ad_score":Y_mc.masq_ad_score,
                       "masq_aa_score":Y_mc.masq_aa_score,
                       "lsc_score":Y_mc.lsc_score,
                       "ce_tleq":Y_mc.ce_tleq,
                       "ctq_total":Y_mc.ctq_total,
                       "oc_tleq":Y_mc.oc_tleq,
                       "rrs_total":Y_mc.rrs_total,
                       "shaps_score_2":Y_mc.shaps_score_2,
                       "bss_total":Y_mc.bss_total,
                       "sticsa_somatic":Y_mc.sticsa_somatic,
                       "sticsa_cognitive":Y_mc.sticsa_cognitive,
                       "pss_score":Y_mc.pss_score
                      })

corr_Y_df= cc_res_Y.corr(method='pearson')
corr_X_df= cc_res_X_data.corr(method='pearson')
# Get lower triangular correlation matrix
Y_df_lt = corr_Y_df.where(np.tril(np.ones(corr_Y_df.shape)).astype(np.bool))

#plt.rcParams.update({'font.size': 24})



# make a lower triangular correlation heatmap with Seaborn
plt.figure(figsize=(20,20))
sns.set(font_scale=2)
sns.heatmap(corr_Y_df,cmap="coolwarm",annot=True,fmt='.1g')
plt.tight_layout()
plt.savefig("Heatmap_Canonical_Correlates_from_Y_and_data.jpg",
                    format='jpeg',
                    dpi=100)

# make a lower triangular correlation heatmap with Seaborn
plt.figure(figsize=(60,40))
sns.set(font_scale=2)
sns.heatmap(corr_X_df,cmap="coolwarm",annot=True,fmt='.1g')
plt.tight_layout()
'''
plt.savefig("Heatmap_Canonical_Correlates_from_Y_and_data.jpg",
                    format='jpeg',
                    dpi=100)
'''


cc_res = pd.DataFrame({"CCX_1":X_c[:, 0],
                       "CCY_1":Y_c[:, 0],
                       "CCX_2":X_c[:, 1],
                       "CCY_2":Y_c[:, 1],
                      })

sns.set_context("talk", font_scale=1.2)
plt.figure(figsize=(10,8))
sns.scatterplot(x="CCX_1",
                y="CCY_1", 
                data=cc_res)
plt.title('Comp. 1, corr = %.2f' %
         np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1])


plt.figure(figsize=(10,8))
sns.scatterplot(x="CCX_2",
                y="CCY_2", 
                data=cc_res)
plt.title('Comp. 1, corr = %.2f' %
         np.corrcoef(X_c[:, 1], Y_c[:, 1])[0, 1])


# Obtain the loading values
xrot = my_cca.x_loadings_
maxvalues=np.argmax(abs(my_cca.x_loadings_),axis=0)
print(index[maxvalues[0]])
print(index[maxvalues[1]])
yrot = my_cca.y_loadings_
maxvalues=np.argmax(abs(my_cca.y_loadings_),axis=0)
print(scorelist[maxvalues[0]])
print(scorelist[maxvalues[1]])


plt.figure(figsize=(25, 25))


# Plot an arrow and a text label for each variable
for var_i in range(13):
  x = yrot[var_i,0]
  y = yrot[var_i,1]

  plt.arrow(0,0,x,y, color='blue')
  plt.text(x,y, scorelist[var_i], color='blue')

plt.title("Loading Value On Symptom Score")
plt.show()


plt.figure(figsize=(80, 80))
# Plot an arrow and a text label for each variable
#for var_i in range(87):
for var_i in range(30):
  x = xrot[var_i,0]
  y = xrot[var_i,1]

  plt.arrow(0,0,x,y)
  plt.text(x,y, index[var_i], color='blue',fontsize=60)

plt.title("Loading Value On MRI Image",fontsize=60)
plt.show()


scorelist_t=['masq_ad_score','masq_aa_score','lsc_score','ce_tleq',
           'oc_tleq','rrs_total','shaps_score_2','bss_total',
           'sticsa_somatic','sticsa_cognitive','pss_score']
df_t=df.drop(scorelist_t, axis=1)

# Obtain the rotation matrices
xrot = my_cca.x_rotations_
yrot = my_cca.y_rotations_
yrot_t=np.vstack((yrot[0,:],yrot[5,:]))
# Put them together in a numpy matrix
xyrot = np.vstack((yrot_t,xrot))

nvariables = xyrot.shape[0]

plt.figure(figsize=(100, 100))
plt.xlim((-1,1))
plt.ylim((-1,1))



# Plot an arrow and a text label for each variable
for var_i in range(nvariables):
  x = xyrot[var_i,0]
  y = xyrot[var_i,1]

  plt.arrow(0,0,x,y)
  plt.text(x,y, df_t.columns[var_i], color='red' if var_i <2 else 'blue', fontsize=15)

plt.show()


