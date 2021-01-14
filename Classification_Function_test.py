#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn import svm
from sklearn.model_selection import TimeSeriesSplit
import warnings
import Feature_Creation
from Feature_Creation import create_features
warnings.filterwarnings('ignore')


# In[60]:


#%run Feature_Creation.ipynb
#%run Auxillary_Functions.ipynb


# In[61]:


def daily_prediction_analysis(filename,CV= 5,verbose=False,do_forest=False,do_rf =True,do_logreg =True,do_svm=True,do_xgb=True,do_stacking=True):
    ## Read data
    data = pd.read_csv(filename)
    data_feat = create_features(data)
    if verbose==True:
        print(data_feat.head())
    
    
    RF_dict = {}
    logreg_poly_dict = {}
    SVM_poly_dict = {}
    XGB_dict = {}
    ### Normalizing Features and creating test train split and time series cross-validation
    y = data_feat['target'].astype(int)
    X = data_feat.drop(['Date','Adj Close','High','Low','Close','target'],axis=1)
    
    ### Continuous features
    continuous = ['1day_pct', '2day_pct', '3day_pct', '4day_pct', '5day_pct', '7day_pct',
                  '1day_pct_cs','ewma_7', 'ewma_50', 'ewma_200', 'RSI', 'MACD','Volume','day_var','open_close','open_prev_close','high_close']
    ### Discrete features
    discrete = ['prev_hot_streak','prev_cold_streak', 'current_hot_streak', 'current_cold_streak',
                'RSI_overbought','RSI_oversold',
                #'7g(50&200)','7l(50&200)','7g50','7g200',
                'prev_current_hot', 'prev_current_cold','current_hot_prev_cold','current_cold_prev_hot',
                'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
                'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
                'dayowk_0', 'dayowk_1', 'dayowk_2', 'dayowk_3', 'dayowk_4',
               ]
    ### Scale continuos features
    scaler = StandardScaler()
    X_cont = pd.DataFrame(scaler.fit_transform(X[continuous]),columns=continuous)
    
    ### Discerete Features
    X_disc = X[discrete]

    X_cont.reset_index(drop=True,inplace=True)

    X_disc.reset_index(drop=True,inplace=True)
    
    ### Combining
    X = pd.concat([X_cont,X_disc],axis=1)

    if CV=='tscv':    
        train_size = X.shape[0]*4//5
        X_test = X.iloc[train_size:]
        X_train = X.iloc[0:train_size]
        y_test = y.iloc[train_size:]
        y_train = y.iloc[0:train_size]

        ### Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=4)
    else:
        #print('here')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    ### Naive Prediction
    y_hat_test_naive = np.ones(len(y_test))
    naive_dict = {'model':'Naive','precision':precision_score(y_hat_test_naive,y_test),
                    'recall':recall_score(y_hat_test_naive,y_test),
                    'accuracy':accuracy_score(y_hat_test_naive,y_test),
                    'f1':f1_score(y_hat_test_naive,y_test),
                 'return':np.nansum((np.array(X_test['1day_pct'].shift(-1))*(np.array(y_hat_test_naive))))}
    if verbose==True:
        print('naive return:',np.nansum((np.array(X_test['1day_pct'].shift(-1))*(np.array(y_hat_test_naive)))))

    ### Large Forest
    if do_forest == True:
            forest = RandomForestClassifier(n_estimators=3000, max_depth= 10)
            forest.fit(X_train, y_train)
            #plot_feature_importances(forest,n_features=50)
    
    
        
    if do_rf == True:
        rf_clf = RandomForestClassifier()
        rf_param_grid = {
            'n_estimators': [100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 2, 3, 5,10],
            'min_samples_split': [5,10,15],
            'min_samples_leaf': [3,5,9,13]
        }
        rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=CV)
        rf_grid_search.fit(X_train, y_train)
        if verbose==True:
            print(f"Training Accuracy: {rf_grid_search.best_score_ :.2%}")
            print("")
            print(f"Optimal Parameters: {rf_grid_search.best_params_}")
        best_rf = rf_grid_search.best_params_

        y_hat_test_RF = rf_grid_search.predict(X_test)
        RF_dict = {'model':'RF','precision':precision_score(y_hat_test_RF,y_test),'recall':recall_score(y_hat_test_RF,y_test),
           'accuracy':accuracy_score(y_hat_test_RF,y_test),'f1':f1_score(y_hat_test_RF,y_test),
           'return':np.nansum((np.array(X_test['1day_pct'].shift(-1))*((np.array(y_hat_test_RF)-1/2)*2))),
            'information gain':accuracy_score(y_hat_test_RF,y_test)-accuracy_score(y_hat_test_naive,y_test)}
        if verbose==True:
            print(RF_dict)
    
    ### Logistic Regression
    if do_logreg == True:
        logreg_clf = LogisticRegression()
        logreg_param_grid = {
            'fit_intercept': [True,False],
            'solver':['liblinear'],
            'C': np.logspace(0,4,5),
            'penalty': ['l2'],
        }
        logreg_grid_search = GridSearchCV(logreg_clf, logreg_param_grid, cv=CV)
        logreg_grid_search.fit(X_train, y_train)
        y_hat_test_log = logreg_grid_search.predict(X_test)
        logreg_poly_dict = {'model':'Logistic','precision':precision_score(y_hat_test_log,y_test),
                    'recall':recall_score(y_hat_test_log,y_test),
                   'accuracy':accuracy_score(y_hat_test_log,y_test),
                    'f1':f1_score(y_hat_test_log,y_test),
                   'return':np.nansum((np.array(X_test['1day_pct'].shift(-1))*((np.array(y_hat_test_log)-1/2)*2))),
                    'information gain':accuracy_score(y_hat_test_log,y_test)-accuracy_score(y_hat_test_naive,y_test)}
        if verbose==True:
            print(logreg_poly_dict)
        
    
    ### SVM poly
    if do_svm==True:
        svm_clf_poly = svm.SVC(kernel='poly')
        r_range =  np.array([0.25,0.5, 1,2,4])
        gamma_range =  np.array([0.0001,0.001, 0.01,0.1])
        d_range = np.array([2,3, 4])
        param_grid = dict(gamma=gamma_range, degree=d_range, coef0=r_range)
        svm_grid_search_poly = GridSearchCV(svm_clf_poly, param_grid, cv=CV)
        svm_grid_search_poly.fit(X_train, y_train)
        best_svm = svm_grid_search_poly.best_params_
        y_hat_test_svm_poly = svm_grid_search_poly.predict(X_test)
        SVM_poly_dict = {'model':'SVM_poly','precision':precision_score(y_hat_test_svm_poly,y_test),'recall':recall_score(y_hat_test_svm_poly,y_test),
               'accuracy':accuracy_score(y_hat_test_svm_poly,y_test),'f1':f1_score(y_hat_test_svm_poly,y_test),
                'return':np.nansum((np.array(X_test['1day_pct'].shift(-1))*((np.array(y_hat_test_svm_poly)-1/2)*2))),
                'information gain':accuracy_score(y_hat_test_svm_poly,y_test)-accuracy_score(y_hat_test_naive,y_test)}
        if verbose==True:
            print(SVM_poly_dict)
    
    ##XGB
    if do_xgb==True:
        estimator = XGBClassifier(
        objective= 'binary:logistic',
        nthread=2,
        seed=42)
        parameters = {
            'max_depth': range (2, 10, 1),
            'n_estimators': range(20, 120, 10),
            'learning_rate': [0.001,0.003,0.01, 0.03, 0.1]
        }

        xgb_grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=parameters,
            n_jobs = 10,
            cv = CV,
            verbose=False
        )
        xgb_grid_search.fit(X_train, y_train)
        if verbose==True:
            print(f"Training Accuracy: {xgb_grid_search.best_score_ :.2%}")
            print("")
            print(f"Optimal Parameters: {xgb_grid_search.best_params_}")
        xgb_best = xgb_grid_search.best_params_
        y_hat_test_XGB = xgb_grid_search.predict(X_test)
        XGB_dict = {'model':'XGB','precision':precision_score(y_hat_test_XGB,y_test),'recall':recall_score(y_hat_test_XGB,y_test),
                   'accuracy':accuracy_score(y_hat_test_XGB,y_test),'f1':f1_score(y_hat_test_XGB,y_test),
                   'return':np.nansum((np.array(X_test['1day_pct'].shift(-1))*((np.array(y_hat_test_XGB)-1/2)*2))),
                   'information gain':accuracy_score(y_hat_test_XGB,y_test)-accuracy_score(y_hat_test_naive,y_test)}
        if verbose==True:
            print(XGB_dict)
        
    
    if do_stacking==True:
        rf_base = RandomForestClassifier(n_estimators = best_rf['n_estimators'],
                                               criterion = best_rf['criterion'],max_depth=best_rf['max_depth'],
                                               min_samples_split = best_rf['min_samples_split'],
                                              min_samples_leaf= best_rf['min_samples_leaf'])
        xgb_base = XGBClassifier(n_estimators = xgb_best['n_estimators'],
                                 max_depth = xgb_best['max_depth'],
                                 learning_rate = xgb_best['learning_rate'],
                                objective= 'binary:logistic',
                                nthread=2,
                                seed=42)
        base_models = [('random_forest', rf_base),
               ('xgb', xgb_base)]          
        stack_clf = StackingClassifier(estimators = base_models,final_estimator = LogisticRegression(),
                                           cv = 5)
        stack_clf.fit(X_train, y_train)
        y_hat_test_stack = stack_clf.predict(X_test)
        stack_dict = {'model':'stack','precision':precision_score(y_hat_test_stack,y_test),'recall':recall_score(y_hat_test_stack,y_test),
                   'accuracy':accuracy_score(y_hat_test_stack,y_test),'f1':f1_score(y_hat_test_stack,y_test),
                   'return':np.nansum((np.array(X_test['1day_pct'].shift(-1))*((np.array(y_hat_test_stack)-1/2)*2))),
                     'information gain':accuracy_score(y_hat_test_stack,y_test)-accuracy_score(y_hat_test_naive,y_test)}
        if verbose==True:
            print(stack_dict)
    return naive_dict, RF_dict, logreg_poly_dict, SVM_poly_dict, XGB_dict, stack_dict
    


# In[ ]:




