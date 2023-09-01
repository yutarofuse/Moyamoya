#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import csv
import matplotlib as mpl

from scipy.stats import norm
from numpy import sqrt
from numpy import argmax
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from pandas.plotting import scatter_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn_pandas import DataFrameMapper
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import RocCurveDisplay

# SHAP
import shap
shap.initjs()
# Permutation Importance
from sklearn.inspection import permutation_importance


# In[41]:


df = pd.read_csv("Moya_internal_dataset.csv")
X = df.drop("Results",axis=1)
y = df["Results"]
df2 = pd.read_csv("Moya_external_dataset.csv")
X2 = df2.drop("Results",axis=1)
y2 = df2["Results"]


# In[180]:


#SVM#
tprs = []
aucs = []
y_preds = []
y_tests = []
param = []
mean_fpr = np.linspace(0, 1, 100)
svcparam = {'C': [1],  'gamma' : [0.1], 'probability' : [True]}
svc_model = SVC()
model_name = 'SVM'

df_results = pd.DataFrame(columns=['df2_value', 'row_num', 'result', 'pos_neg', 'probability'])
df_results2 = pd.DataFrame(columns=['Model','percentile','True positives','True negatives', 'False positives', 'False negatives', 'Accuracy','Sensitivity', 'Specificity', 'F1 score', 'ROC AUC'])

X_train = X
X_test = X2
y_train = y
y_test = y2
select = SelectPercentile(percentile=15)
select.fit(X_train, y_train)
X_train1 = select.transform(X_train)
X_test1 = select.transform(X_test)

selected_features = X.columns.values[select.get_support()]
X1 = pd.DataFrame(X_train1, columns=selected_features)


grid_search = GridSearchCV(estimator=svc_model, param_grid=svcparam, cv=5, scoring='roc_auc')
grid_search.fit(X_train1, y_train)

#best_svc_model = grid_search.best_estimator_
#coefs = best_svc_model.coef_.flatten()
#factor_names = X1.columns.values

y_pred = grid_search.predict(X_test1)
if hasattr(grid_search, 'predict_proba'):
    y_score2 = grid_search.predict_proba(X_test1)[:, 1]
else:
    y_score2 = y_pred
    
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score2)

# get the best threshold
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]

y_prob_pred = (y_score2 >= best_thresh).astype(bool)
#y_prob_pred = (y_score2 >= 0.5).astype(bool)
tprs.append(np.interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
aucs.append(roc_auc_score(y_test, y_score2))
y_preds.extend(y_prob_pred)
y_tests.extend(y_test)

f1 = f1_score(y_test, y_prob_pred)

print(f'Percentile: {15}')
print(f'Accuracy Score: {accuracy_score(y_tests, y_preds)}')
print(f'AUC Score: {np.mean(aucs)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_tests, y_preds)}\n')
sns.heatmap(confusion_matrix(y_tests, y_preds), annot=True, cmap='Blues')
plt.title(f'Confusion Matrix for {model_name} (Percentile: {15})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

cm = confusion_matrix(y_test, y_prob_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

for idx, (actual, predicted, score) in enumerate(zip(y_test, y_prob_pred, y_score2)):
    if actual == 1 and predicted == 1:
        result = 'Correct'
        pos_neg = 'TP'
        prob = score
    elif actual == 0 and predicted == 0:
        result = 'Correct'
        pos_neg = 'TN'
        prob = score
    elif actual == 0 and predicted == 1:
        result = 'Incorrect'
        pos_neg = 'FP'
        prob = score
    else:
        result = 'Incorrect'
        pos_neg = 'FN'
        prob = score
        df_results = df_results.append({'percentile': str(100),'df2_value': X_test.iloc[idx],'row_num': idx,'result': result,'pos_neg': pos_neg,'probability': prob}, ignore_index=True)

df_results2 = df_results2.append({'Model': 'SVM', 'percentile': str(100),'True positives': tp,'True negatives': tn,'False positives': fp,'False negatives': fn,'Accuracy': accuracy_score(y_tests, y_preds),
                            'Sensitivity': sensitivity,'Specificity': specificity, 'F1 score': f1,'ROC AUC': roc_auc_score(y_test, y_score2)}, ignore_index=True)

fpr2 = fpr
tpr2 = tpr


# In[146]:


#RF#
tprs = []
aucs = []
y_preds = []
y_tests = []
param = []
rfparam = {'class_weight': [None], 'criterion': ['gini'], 'max_depth': [200], 'max_features': ['sqrt'], 'min_samples_leaf': [5], 'min_samples_split': [2], 'n_estimators': [1000]}
rf_model = RandomForestClassifier()
model_name = 'RF'

df_results = pd.DataFrame(columns=['df2_value', 'row_num', 'result', 'pos_neg', 'probability'])
df_results2 = pd.DataFrame(columns=['percentile','True positives','True negatives', 'False positives', 'False negatives', 'Accuracy','Sensitivity', 'Specificity', 'ROC AUC'])

X_train = X
X_test = X2
y_train = y
y_test = y2
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train1 = select.transform(X_train)
X_test1 = select.transform(X_test)
X_train1 = X_train.loc[:, select.get_support()]
X_test1 = X_test.loc[:, select.get_support()]
selected_features = X.columns.values[select.get_support()]
X1 = pd.DataFrame(X_train1, columns=selected_features)


grid_search = GridSearchCV(estimator=rf_model, param_grid=rfparam, cv=5, scoring='roc_auc')
grid_search.fit(X_train1, y_train)
grid_search
#best_svc_model = grid_search.best_estimator_
#coefs = best_svc_model.coef_.flatten()
#factor_names = X1.columns.values

y_pred = grid_search.predict(X_test1)
if hasattr(grid_search, 'predict_proba'):
    y_score3 = grid_search.predict_proba(X_test1)[:, 1]
else:
    y_score3 = y_pred
    
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score3)

# get the best threshold
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]

y_prob_pred = (y_score3 >= best_thresh).astype(bool)
#y_prob_pred = (y_score3 >= 0.5).astype(bool)
tprs.append(np.interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
aucs.append(roc_auc_score(y_test, y_score3))
y_preds.extend(y_prob_pred)
y_tests.extend(y_test)
f1 = f1_score(y_test, y_prob_pred)
print(f'Percentile: {50}')
print(f'Accuracy Score: {accuracy_score(y_tests, y_preds)}')
print(f'AUC Score: {np.mean(aucs)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_tests, y_preds)}\n')
sns.heatmap(confusion_matrix(y_tests, y_preds), annot=True, cmap='Blues')
plt.title(f'Confusion Matrix for {model_name} (Percentile: {50})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

cm = confusion_matrix(y_test, y_prob_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

for idx, (actual, predicted, score) in enumerate(zip(y_test, y_prob_pred, y_score3)):
    if actual == 1 and predicted == 1:
        result = 'Correct'
        pos_neg = 'TP'
        prob = score
    elif actual == 0 and predicted == 0:
        result = 'Correct'
        pos_neg = 'TN'
        prob = score
    elif actual == 0 and predicted == 1:
        result = 'Incorrect'
        pos_neg = 'FP'
        prob = score
    else:
        result = 'Incorrect'
        pos_neg = 'FN'
        prob = score

    df_results = df_results.append({'percentile': str(100),'df2_value': X_test.iloc[idx],'row_num': idx,'result': result,'pos_neg': pos_neg,'probability': prob}, ignore_index=True)

df_results2 = df_results2.append({'Model': 'Random forest', 'percentile': str(100),'True positives': tp,'True negatives': tn,'False positives': fp,'False negatives': fn,'Accuracy': accuracy_score(y_tests, y_preds),
                            'Sensitivity': sensitivity,'Specificity': specificity, 'F1 score': f1,'ROC AUC': roc_auc_score(y_test, y_score3)}, ignore_index=True)

fpr3 = fpr
tpr3 = tpr


# In[153]:


feature_importancesRF = grid_search.best_estimator_.feature_importances_

feature_importancesRF

X_train1 = pd.DataFrame(X_train1)
fi = feature_importancesRF  
fi_df = pd.DataFrame({'feature': list(X_train1.columns), 'feature importance': fi[:]}).sort_values('feature importance', ascending = False)

n = int(len(fi_df) * 0.2)

fi_df_top20 = fi_df.sort_values('feature importance', ascending=False)[:n]

sns.barplot(x='feature importance', y='feature', data=fi_df_top20, orient='h', color='gray')

plt.savefig("featureimportanceRF.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')

fi_df.to_csv("RF_fi.csv")


# In[ ]:


plt.close(fig)
#LGBM#
tprs = []
aucs = []
y_preds = []
y_tests = []
param = []
lgbmparam = {'num_leaves': [7], 'learning_rate': [0.01], 'feature_fraction': [0.5],'bagging_fraction': [0.8], 'bagging_freq': [3]} 
lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', verbose = -1,  metric='auc', random_state = 0)
model_name = 'LGBM'

df_results = pd.DataFrame(columns=['df2_value', 'row_num', 'result', 'pos_neg', 'probability'])
#df_results2 = pd.DataFrame(columns=['percentile','True positives','True negatives', 'False positives', 'False negatives', 'Accuracy','Sensitivity', 'Specificity', 'ROC AUC'])

X_train = X
X_test = X2
y_train = y
y_test = y2
select = SelectPercentile(percentile=5)
select.fit(X_train, y_train)
X_train1 = select.transform(X_train)
X_test1 = select.transform(X_test)
X_train1 = X_train.loc[:, select.get_support()]
X_test1 = X_test.loc[:, select.get_support()]
selected_features = X.columns.values[select.get_support()]
X1 = pd.DataFrame(X_train1, columns=selected_features)


grid_search = GridSearchCV(estimator=lgb_estimator, param_grid=lgbmparam, cv=5, scoring='roc_auc')
grid_search.fit(X_train1, y_train)

y_pred = grid_search.predict(X_test1)
if hasattr(grid_search, 'predict_proba'):
    y_score4 = grid_search.predict_proba(X_test1)[:, 1]
else:
    y_score4 = y_pred
    
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score4)

# get the best threshold
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]

y_prob_pred = (y_score4 >= best_thresh).astype(bool)
#y_prob_pred = (y_score4 >= 0.5).astype(bool)
tprs.append(np.interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
aucs.append(roc_auc_score(y_test, y_score4))
y_preds.extend(y_prob_pred)
y_tests.extend(y_test)
f1 = f1_score(y_test, y_prob_pred)
print(f'Percentile: {5}')
print(f'Accuracy Score: {accuracy_score(y_tests, y_preds)}')
print(f'AUC Score: {np.mean(aucs)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_tests, y_preds)}\n')
sns.heatmap(confusion_matrix(y_tests, y_preds), annot=True, cmap='Blues')
plt.title(f'Confusion Matrix for {model_name} (Percentile: {5})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

cm = confusion_matrix(y_test, y_prob_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

for idx, (actual, predicted, score) in enumerate(zip(y_test, y_prob_pred, y_score4)):
    if actual == 1 and predicted == 1:
        result = 'Correct'
        pos_neg = 'TP'
        prob = score
    elif actual == 0 and predicted == 0:
        result = 'Correct'
        pos_neg = 'TN'
        prob = score
    elif actual == 0 and predicted == 1:
        result = 'Incorrect'
        pos_neg = 'FP'
        prob = score
    else:
        result = 'Incorrect'
        pos_neg = 'FN'
        prob = score

    df_results = df_results.append({'percentile': str(100),'df2_value': X_test.iloc[idx],'row_num': idx,'result': result,'pos_neg': pos_neg,'probability': prob}, ignore_index=True)

df_results2 = df_results2.append({'Model': 'Light GBM', 'percentile': str(100),'True positives': tp,'True negatives': tn,'False positives': fp,'False negatives': fn,'Accuracy': accuracy_score(y_tests, y_preds),
                            'Sensitivity': sensitivity,'Specificity': specificity, 'F1 score': f1,'ROC AUC': roc_auc_score(y_test, y_score4)}, ignore_index=True)

feature_importancesLGBM = grid_search.best_estimator_.feature_importances_

X_train1 = pd.DataFrame(X_train1)
fi2 = feature_importancesLGBM  
fi_df2 = pd.DataFrame({'feature': list(X_train1.columns), 'feature importance': fi2[:]}).sort_values('feature importance', ascending = False)
n = int(len(fi_df) * 0.2)
fi_df_top202 = fi_df2.sort_values('feature importance', ascending=False)[:n]
# バープロットを作成
sns.barplot(x='feature importance', y='feature', data=fi_df_top202, orient='h', color='gray')
plt.savefig("featureimportanceLGBM.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')

fi_df.to_csv("LGBM_fi.csv")

fpr4 = fpr
tpr4 = tpr


# In[173]:


plt.close(fig)
plt.figure()
# SHAPの算出
model2 = grid_search.best_estimator_
explainer2 = shap.TreeExplainer(model2)
shap_values2 = explainer2.shap_values(X_test1)[1]
shap.summary_plot(shap_values2, X_test1, show= False)
plt.savefig("SHAP_LGBM2.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')


# In[175]:


#shap.summary_plot(shap_values, X_test1)
plt.figure()
fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # Adjust the figure size as needed
shap.summary_plot(shap_values2,  X_test1, plot_type="bar", show=False)

# Save the figure
plt.savefig("SHAP_LGBM_bar2.tif", format= "tiff", dpi = 600, bbox_inches = 'tight')
plt.close(fig)  # Close the figure


# In[ ]:


plt.close(fig)
plt.figure()
fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # Adjust the figure size as needed
# shap_valuesの値を変更する。今回は、3としてあり、上段0から数えるため、Patient number 4 となる。同時に、X_test1.ilocも変更が必要。
shap.force_plot(explainer2.expected_value[1], shap_values2[3], X_test1.iloc[3,:], show =False,matplotlib=True)
plt.savefig("SHAP_LGBM_force_plot2.tif", format= "tiff", dpi = 600, bbox_inches="tight")
plt.close(fig)  # Close the figure


# In[ ]:


base_value = explainer2.expected_value
base_value


# In[ ]:


plt.close(fig)
plt.figure()
#fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # Adjust the figure size as needed
# shap_valuesの値を変更する。今回は、3としてあり、上段0から数えるため、Patient number 4 となる。同時に、X_test1.ilocも変更が必要。
shap.force_plot(explainer2.expected_value[1], shap_values2[4], X_test1.iloc[4,:], show =False,matplotlib=True)
plt.savefig("SHAP_LGBM_force_plot3.tif", format= "tiff", dpi = 600, bbox_inches="tight")
plt.close(fig)  # Close the figure


# In[163]:


df_results2.to_csv('Results.csv', index=False)


# In[165]:


import seaborn as sns
sns.set_palette("Accent") # カラーパレットを設定
plt.plot(fpr2,tpr2,label='SVM (AUC= %0.3f)' % round(auc(fpr2, tpr2), 3), color=sns.color_palette()[2])
plt.plot(fpr3,tpr3,label='RF (AUC= %0.3f)' % round(auc(fpr3, tpr3), 3), color=sns.color_palette()[4])
plt.plot(fpr4,tpr4,label='LGBM (AUC= %0.3f)' % round(auc(fpr4, tpr4), 3), color=sns.color_palette()[5])
plt.plot([0,0,1], [0,1,1], linestyle='--', color = 'gray')
plt.plot([0, 1], [0, 1], linestyle='--', color = 'gray')
plt.legend()
plt.xlabel('false positive rate (FPR)')
plt.ylabel('true positive rate (TPR)')
plt.savefig("ROC2.tif", format= "tiff", dpi = 600, bbox_inches = 'tight')
plt.show()

