#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from numpy import sqrt
from numpy import argmax
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
#from sklearn.metrics import plot_roc_curve
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from pandas.plotting import scatter_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn_pandas import DataFrameMapper
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.metrics import RocCurveDisplay


# In[16]:


df = pd.read_csv("Moya_internal_dataset.csv")
X = df.drop("Results",axis=1)
y = df["Results"]
X = X.drop(X.columns[[0]],axis =1)


# In[18]:


#SelectPercentile

skf = StratifiedKFold(n_splits= 5, shuffle = True, random_state = 0)
svcparam = {'C': [0.001, 0.01, 0.1, 1, 10, 100],  'gamma' : [0.001, 0.01, 0.1, 1, 10, 100], 'probability' : [True]}
rfparam = {'criterion'   : ['gini'],'n_estimators': [1000],'max_features': ['sqrt'], 'min_samples_leaf': [2, 5, 10], 'min_samples_split': [2, 5], 'max_depth': [7, 63, 200], 'class_weight': ['balanced', None]}
lgbmparam = {'num_leaves': [7, 15, 31], 'learning_rate': [0.1, 0.01, 0.005], 'feature_fraction': [0.5, 0.8],'bagging_fraction': [0.8], 'bagging_freq': [1, 3]} 
lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', verbose = -1,  metric='auc', random_state = 0)
a_list = [SVC(), RandomForestClassifier(), lgb_estimator]
b_list = [[svcparam], [rfparam], [lgbmparam]
c_list = ['Support vector machine', 'Random forest', 'Light gradient boosting machine']

for (h,p,r) in zip(a_list,b_list,c_list):
    s = [5,10,15,20,50,100]
    pm = []
    parameter = []
    for i in list(s):

        tprs = []
        aucs = []
        pm1 = []
        param = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()

        param_grid = p
        for j, (train_index, test_index) in enumerate(skf.split(X,y)):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(X_train)
            scaler.transform(X_train)
            X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

            grid_search = GridSearchCV(estimator= h ,  param_grid= param_grid, cv=5, scoring = 'roc_auc')

            select = SelectPercentile(percentile = i)
            select.fit(X_train, y_train)
            X_train1 = select.transform(X_train)
            X_test1 = select.transform(X_test)
            grid_search.fit(X_train1, y_train)
            Y_pred = grid_search.predict(X_test1)
            Y_score = grid_search.predict_proba(X_test1)[:,1]
            fpr, tpr, thresholds = roc_curve(y_true=y_test,y_score=Y_score)

            # get the best threshold
            J = tpr - fpr
            ix = argmax(J)
            best_thresh = thresholds[ix]
            y_prob_pred = (grid_search.predict_proba(X_test1)[:,1]>= best_thresh).astype(bool)

            def specificity_score(y_test, y_prob_pred):
                tn, fp, fn, tp = confusion_matrix(y_test, y_prob_pred).flatten()
                return tn / (tn + fp)


            accuracy1 = format(accuracy_score(y_true = y_test , y_pred = y_prob_pred))
            f1_1 = format(f1_score(y_true = y_test , y_pred = y_prob_pred))
            sensitivity1 = format(recall_score(y_true = y_test , y_pred = y_prob_pred))
            specificity1 = format(specificity_score(y_test , y_prob_pred))
            PPV1 = format(precision_score(y_true = y_test , y_pred = y_prob_pred))
            AUC1 = format(roc_auc_score(y_true=y_test,y_score=Y_score))
            bestparameters1 = format(grid_search.best_params_)
            performance = (format(i), accuracy1, f1_1, sensitivity1, specificity1, PPV1, AUC1)
            pm1.append(performance)
            param.append(bestparameters1)
            #accu.append(accuracy1)
            viz = RocCurveDisplay.from_estimator(
            grid_search,
            X_test1,
            y_test,
            name="ROC fold {}".format(j),
            alpha=0.3,
            lw=1,
            ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        pm1 = np.asarray(pm1, dtype=np.float32)
        mean_pm1 = np.mean(pm1, axis=0)
        std_pm1 = np.std(pm1, axis=0)
        pm1= np.vstack([pm1, mean_pm1])
        pm1= np.vstack([pm1, std_pm1])

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title= r,
            xlabel = 'False positive rate (FPR)',
            ylabel = 'True positive rate (TPR)'

        )
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="gray", alpha=0.8)
        ax.legend(loc="lower right")
        plt.savefig(str(i)+str(h)+"ROC.tif", format = "tiff", dpi = 1200, bbox_inches = 'tight')
        plt.show()
        print('confusion matrix = \n', confusion_matrix(y_true = y_test, y_pred = y_prob_pred))
        pm1
        pm1= pd.DataFrame(pm1)
        f = open('1.csv', 'a')
        pm1.to_csv('1.csv', mode = 'a')
        f.write("\n")
        f.close()
        f2 = open('2.csv', 'a')
        param = pd.DataFrame(param)
        param.to_csv('2.csv', mode = 'a')
        f2.write("\n")
        f2.close()


# In[19]:


#SelectPercentile

skf = StratifiedKFold(n_splits= 5, shuffle = True, random_state = 0)
svcparam = {'C': [0.001, 0.01, 0.1, 1, 10, 100],  'gamma' : [0.001, 0.01, 0.1, 1, 10, 100], 'probability' : [True]}
rfparam = {'criterion'   : ['gini'],'n_estimators': [1000],'max_features': ['sqrt'], 'min_samples_leaf': [2, 5, 10], 'min_samples_split': [2, 5], 'max_depth': [7, 63, 200], 'class_weight': ['balanced', None]}
lgbmparam = {'num_leaves': [7, 15, 31], 'learning_rate': [0.1, 0.01, 0.005], 'feature_fraction': [0.5, 0.8],'bagging_fraction': [0.8], 'bagging_freq': [1, 3]} 
lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', verbose = -1,  metric='auc', random_state = 0)
a_list = [SVC(), RandomForestClassifier(), lgb_estimator]
b_list = [[svcparam], [rfparam], [lgbmparam]
c_list = ['Support vector machine', 'Random forest', 'Light gradient boosting machine']
for (h,p,r) in zip(a_list,b_list,c_list):
    s = [5,10,15,20,50,100]
    pm = []
    parameter = []
    for i in list(s):

        tprs = []
        aucs = []
        pm1 = []
        param = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()

        param_grid = p
        for j, (train_index, test_index) in enumerate(skf.split(X,y)):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(X_train)
            scaler.transform(X_train)
            X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

            grid_search = GridSearchCV(estimator= h ,  param_grid= param_grid, cv=5, scoring = 'roc_auc')

            select = SelectPercentile(percentile = i)
            select.fit(X_train, y_train)
            X_train1 = select.transform(X_train)
            X_test1 = select.transform(X_test)
            sm = SMOTE(random_state=0)
            x_resampled, y_resampled = sm.fit_sample(X_train1, y_train)
            grid_search.fit(x_resampled, y_resampled)
            Y_pred = grid_search.predict(X_test1)
            Y_score = grid_search.predict_proba(X_test1)[:,1]
            fpr, tpr, thresholds = roc_curve(y_true=y_test,y_score=Y_score)

            # get the best threshold
            J = tpr - fpr
            ix = argmax(J)
            best_thresh = thresholds[ix]
            y_prob_pred = (grid_search.predict_proba(X_test1)[:,1]>= best_thresh).astype(bool)

            def specificity_score(y_test, y_prob_pred):
                tn, fp, fn, tp = confusion_matrix(y_test, y_prob_pred).flatten()
                return tn / (tn + fp)


            accuracy1 = format(accuracy_score(y_true = y_test , y_pred = y_prob_pred))
            f1_1 = format(f1_score(y_true = y_test , y_pred = y_prob_pred))
            sensitivity1 = format(recall_score(y_true = y_test , y_pred = y_prob_pred))
            specificity1 = format(specificity_score(y_test , y_prob_pred))
            PPV1 = format(precision_score(y_true = y_test , y_pred = y_prob_pred))
            AUC1 = format(roc_auc_score(y_true=y_test,y_score=Y_score))
            bestparameters1 = format(grid_search.best_params_)
            performance = (format(i), accuracy1, f1_1, sensitivity1, specificity1, PPV1, AUC1)
            pm1.append(performance)
            param.append(bestparameters1)
            #accu.append(accuracy1)
            viz = RocCurveDisplay.from_estimator(
            grid_search,
            X_test1,
            y_test,
            name="ROC fold {}".format(j),
            alpha=0.3,
            lw=1,
            ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        pm1 = np.asarray(pm1, dtype=np.float32)
        mean_pm1 = np.mean(pm1, axis=0)
        std_pm1 = np.std(pm1, axis=0)
        pm1= np.vstack([pm1, mean_pm1])
        pm1= np.vstack([pm1, std_pm1])

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title= r,
            xlabel = 'False positive rate (FPR)',
            ylabel = 'True positive rate (TPR)'

        )
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="gray", alpha=0.8)
        ax.legend(loc="lower right")
        plt.savefig(str(i)+str(h)+"SMOTE_ROC.tif", format = "tiff", dpi = 1200, bbox_inches = 'tight')
        plt.show()
        print('confusion matrix = \n', confusion_matrix(y_true = y_test, y_pred = y_prob_pred))
        pm1
        pm1= pd.DataFrame(pm1)
        f = open('1.csv', 'a')
        pm1.to_csv('1.csv', mode = 'a')
        f.write("\n")
        f.close()
        f2 = open('2.csv', 'a')
        param = pd.DataFrame(param)
        param.to_csv('2.csv', mode = 'a')
        f2.write("\n")
        f2.close()


# In[ ]:




