#!/usr/bin/env python
# coding: utf-8

# # Classifying whether an ETF goes up or down
# ## Here we will use the data of the ETF we are prediciting and not any other data

# #### Importing required libraries

# In[6]:


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
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn import svm
from sklearn.model_selection import TimeSeriesSplit
import warnings
from multiprocess import Pool #Note multiprocess and not multiprocessinh
import datetime
import classification
warnings.filterwarnings('ignore')


# In[7]:


get_ipython().run_line_magic('run', 'Feature_Creation.ipynb')
get_ipython().run_line_magic('run', 'Auxillary_Functions.ipynb')
#%run Classification_Function.ipynb


# ## Importing data

# #### Importing data and generating features

# In[4]:


SPY = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()),'data','largecap','SPY.csv'))

SPY_feat = create_features(SPY)

SPY_feat.head()


# #### Normalizing features

# In[5]:


y = SPY_feat['target'].astype(int)
X = SPY_feat.drop(['Date','Adj Close','High','Low','Close','target'],axis=1)

continuous = ['1day_pct', '2day_pct', '3day_pct', '4day_pct', '5day_pct', '7day_pct',
              '1day_pct_cs',
              'ewma_7', 'ewma_50', 'ewma_200', 'RSI', 'MACD','Volume','day_var','open_close','open_prev_close','high_close']
discrete = ['prev_hot_streak','prev_cold_streak', 'current_hot_streak', 'current_cold_streak',
            'RSI_overbought','RSI_oversold',
            #'7g(50&200)','7l(50&200)','7g50','7g200',
            'prev_current_hot', 'prev_current_cold','current_hot_prev_cold','current_cold_prev_hot',
            'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
            'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
            'dayowk_0', 'dayowk_1', 'dayowk_2', 'dayowk_3', 'dayowk_4',
           ]
scaler = StandardScaler()
X_cont = pd.DataFrame(scaler.fit_transform(X[continuous]),columns=continuous)

X_disc = X[discrete]

X_cont.reset_index(drop=True,inplace=True)

X_disc.reset_index(drop=True,inplace=True)

X = pd.concat([X_cont,X_disc],axis=1)


X_test = X.iloc[1000:]
X_train = X.iloc[0:1000]
y_test = y.iloc[1000:]
y_train = y.iloc[0:1000]

tscv = TimeSeriesSplit(n_splits=4)
print(tscv.split(X_train))
for train, val in tscv.split(X_train):
    print("%s %s" % (train, val))


# # Analysis

# # SPY

# ### Naive Prediction. Up everyday

# In[6]:


y_hat_test_naive = np.ones(len(y_test))
print('precision score' ,precision_score(y_hat_test_naive,y_test))
print('recall score' ,recall_score(y_hat_test_naive,y_test))
print('accuracy score' ,accuracy_score(y_hat_test_naive,y_test))
print('f1 score' ,f1_score(y_hat_test_naive,y_test))


# ### Random Forests not tuned

# In[7]:


forest = RandomForestClassifier(n_estimators=3000, max_depth= 10)
forest.fit(X_train, y_train)
print(forest.score(X_test, y_test))

plot_feature_importances(forest,n_features=50)


# In[8]:


y_hat_test = forest.predict(X_test)
print('precision score' ,precision_score(y_hat_test,y_test))
print('recall score' ,recall_score(y_hat_test,y_test))
print('accuracy score' ,accuracy_score(y_hat_test,y_test))
print('f1 score' ,f1_score(y_hat_test,y_test))
print(y_test.value_counts()/y_test.value_counts().sum())


# ### Random Forest tuned using GridSearch

# In[26]:


rf_clf = RandomForestClassifier()
rf_param_grid = {
    'n_estimators': [5,7,10,30,50],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 3, 5,10],
    'min_samples_split': [5,10,15],
    'min_samples_leaf': [3,5,9,13]
}
rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=tscv)
rf_grid_search.fit(X_train, y_train)

print(f"Training Accuracy: {rf_grid_search.best_score_ :.2%}")
print("")
print(f"Optimal Parameters: {rf_grid_search.best_params_}")
best_rf = rf_grid_search.best_params_
y_hat_test_RF = rf_grid_search.predict(X_test)
RF_dict = {'model':'RF','precision':precision_score(y_hat_test_RF,y_test),'recall':recall_score(y_hat_test_RF,y_test),
           'accuracy':accuracy_score(y_hat_test_RF,y_test),'f1':f1_score(y_hat_test_RF,y_test),
           'return':np.nansum((np.array(X_test['1day_pct'].shift(-1))*((np.array(y_hat_test_RF)-1/2)*2)))}

print(RF_dict)

#print(np.nansum((np.array(X_test['1day_pct'].shift(-1))*((np.array(y_hat_test_RF)-1/2)*2))))

#print(np.nansum((np.array(X_test['1day_pct'].shift(-1))*(np.array(y_hat_test_naive)))))


# In[27]:


best_rf['n_estimators']


# In[35]:


RandomForestClassifier(n_estimators=best_rf['n_estimators'],
                       criterion = rf_grid_search.best_params_['criterion'],
                      max_depth= rf_grid_search.best_params_['max_depth']).fit(X_train,y_train)


# ### Logistic Regression

# In[15]:


logreg_clf = LogisticRegression()
logreg_param_grid = {
    'fit_intercept': [True,False],
    'solver':['liblinear'],
    'C': np.logspace(0,4,5),
    'penalty': ['l2'],
}
logreg_grid_search = GridSearchCV(logreg_clf, logreg_param_grid, cv=tscv)
logreg_grid_search.fit(X_train, y_train)

print(f"Training Accuracy: {logreg_grid_search.best_score_ :.2%}")
print("")
print(f"Optimal Parameters: {logreg_grid_search.best_params_}")


# In[16]:


y_hat_test_log = logreg_grid_search.predict(X_test)
logreg_poly_dict = {'model':'SVM_poly','precision':precision_score(y_hat_test_log,y_test),
                    'recall':recall_score(y_hat_test_log,y_test),
                   'accuracy':accuracy_score(y_hat_test_log,y_test),
                    'f1':f1_score(y_hat_test_log,y_test),
                   'return':np.nansum((np.array(X_test['1day_pct'].shift(-1))*((np.array(y_hat_test_RF)-1/2)*2)))}


# In[17]:


logreg_poly_dict


# ## SVM

# In[18]:


clf = svm.SVC(kernel='linear')
clf.fit(X_train,y_train)


# In[19]:


svm_clf = svm.SVC()
C_range = np.array([50., 100., 200., 500.])
gamma_range = np.array([0.0001,3*0.0001,0.001,3*0.001])
svm_param_grid = dict(gamma=gamma_range, C=C_range)

svm_grid_search = GridSearchCV(svm_clf, svm_param_grid, cv=tscv)
svm_grid_search.fit(X_train, y_train)

print(f"Training Accuracy: {svm_grid_search.best_score_ :.2%}")
print("")
print(f"Optimal Parameters: {svm_grid_search.best_params_}")


# In[20]:


y_hat_test_svm = svm_grid_search.predict(X_test)
SVM_dict = {'model':'SVM','precision':precision_score(y_hat_test_svm,y_test),'recall':recall_score(y_hat_test_svm,y_test),
           'accuracy':accuracy_score(y_hat_test_svm,y_test),'f1':f1_score(y_hat_test_svm,y_test)}


# In[21]:


SVM_dict


# In[22]:


svm_clf_poly = svm.SVC(kernel='poly')
r_range =  np.array([0.25,0.5, 1,2,4])
gamma_range =  np.array([0.0001,0.001, 0.01,0.1])
d_range = np.array([2,3, 4])
param_grid = dict(gamma=gamma_range, degree=d_range, coef0=r_range)
svm_grid_search_poly = GridSearchCV(svm_clf_poly, param_grid, cv=tscv)
svm_grid_search_poly.fit(X_train, y_train)

print(f"Training Accuracy: {svm_grid_search_poly.best_score_ :.2%}")
print("")
print(f"Optimal Parameters: {svm_grid_search_poly.best_params_}")


# In[23]:


y_hat_test_svm_poly = svm_grid_search_poly.predict(X_test)
SVM_poly_dict = {'model':'SVM_poly','precision':precision_score(y_hat_test_svm_poly,y_test),'recall':recall_score(y_hat_test_svm_poly,y_test),
           'accuracy':accuracy_score(y_hat_test_svm_poly,y_test),'f1':f1_score(y_hat_test_svm_poly,y_test)}


# In[24]:


SVM_poly_dict


# In[25]:


np.nansum((np.array(X_test['1day_pct'].shift(-1))*((np.array(y_hat_test_svm)-1/2)*2)))


# ## Gradient Boosting Classifier

# In[26]:


model = XGBClassifier()


# In[27]:


model.fit(X_train,y_train)


# In[28]:


model.score(X_test,y_test)


# In[29]:


estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=2,
    seed=42
)
parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(20, 120, 10),
    'learning_rate': [0.001,0.003,0.01, 0.03, 0.1]
}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    n_jobs = 10,
    cv = tscv,
    verbose=True
)
grid_search.fit(X_train, y_train)
print(f"Training Accuracy: {grid_search.best_score_ :.2%}")
print("")
print(f"Optimal Parameters: {grid_search.best_params_}")

y_hat_test_XGB = grid_search.predict(X_test)
XGB_dict = {'model':'XGB','precision':precision_score(y_hat_test_XGB,y_test),'recall':recall_score(y_hat_test_XGB,y_test),
           'accuracy':accuracy_score(y_hat_test_XGB,y_test),'f1':f1_score(y_hat_test_XGB,y_test)}

print(XGB_dict)

print(np.nansum((np.array(X_test['1day_pct'].shift(-1))*((np.array(y_hat_test_XGB)-1/2)*2))))

print(np.nansum((np.array(X_test['1day_pct'].shift(-1))*np.ones(len(y_test)))))


# ## Voting

# In[32]:


voted_y = []
for i in range(len(y_test)):
    l = np.array([y_hat_test_RF[i],y_hat_test_log[i],y_hat_test_svm[i],y_hat_test_XGB[i]])
    counts = np.bincount(l)
    voted_y.append(np.argmax(counts))
voted_y = np.array(voted_y)
voted_dict = {'model':'Stacked','precision':precision_score(voted_y,y_test),'recall':recall_score(voted_y,y_test),
           'accuracy':accuracy_score(voted_y,y_test),'f1':f1_score(voted_y,y_test)}


# In[33]:


voted_dict


# ## Stacked Ananlysis

# In[ ]:





# # Gold Predictions XGB

# In[64]:


GLD = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()),'data','other','gold.csv'))

GLD_feat = create_features(GLD)

GLD_feat.head()


# In[29]:


y = GLD_feat['target'].astype(int)
X = GLD_feat.drop(['Date','Adj Close','High','Low','Close','target'],axis=1)

continuous = ['1day_pct', '2day_pct', '3day_pct', '4day_pct', '5day_pct', '7day_pct',
              '1day_pct_cs',
              'ewma_7', 'ewma_50', 'ewma_200', 'RSI', 'MACD','Volume','day_var','open_close','open_prev_close','high_close']
discrete = ['prev_hot_streak','prev_cold_streak', 'current_hot_streak', 'current_cold_streak',
            'RSI_overbought','RSI_oversold',
            #'7g(50&200)','7l(50&200)','7g50','7g200',
            'prev_current_hot', 'prev_current_cold','current_hot_prev_cold','current_cold_prev_hot',
            'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
            'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
            'dayowk_0', 'dayowk_1', 'dayowk_2', 'dayowk_3', 'dayowk_4',
           ]
scaler = StandardScaler()
X_cont = pd.DataFrame(scaler.fit_transform(X[continuous]),columns=continuous)

X_disc = X[discrete]

X_cont.reset_index(drop=True,inplace=True)

X_disc.reset_index(drop=True,inplace=True)

X = pd.concat([X_cont,X_disc],axis=1)

train_size = X.shape[0]*4//5
X_test = X.iloc[train_size:]
X_train = X.iloc[0:train_size]
y_test = y.iloc[train_size:]
y_train = y.iloc[0:train_size]

tscv = TimeSeriesSplit(n_splits=4)
print(tscv.split(X_train))
for train, val in tscv.split(X_train):
    print("%s %s" % (train, val))


# ### Naive Prediction. Trade based on previous day only

# In[30]:


y_hat_test_naive = np.ones(len(y_test))
print('precision score' ,precision_score(y_hat_test_naive,y_test))
print('recall score' ,recall_score(y_hat_test_naive,y_test))
print('accuracy score' ,accuracy_score(y_hat_test_naive,y_test))
print('f1 score' ,f1_score(y_hat_test_naive,y_test))


# ## Random Forest Grid Search

# In[31]:


rf_clf = RandomForestClassifier()
rf_param_grid = {
    'n_estimators': [100,200,300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 3, 5,10,12,15],
    'min_samples_split': [5,10,15],
    'min_samples_leaf': [2,3,5,9,13]
}
rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=tscv)
rf_grid_search.fit(X_train, y_train)

print(f"Training Accuracy: {rf_grid_search.best_score_ :.2%}")
print("")
print(f"Optimal Parameters: {rf_grid_search.best_params_}")

y_hat_test_RF = rf_grid_search.predict(X_test)
RF_dict = {'model':'RF','precision':precision_score(y_hat_test_RF,y_test),'recall':recall_score(y_hat_test_RF,y_test),
           'accuracy':accuracy_score(y_hat_test_RF,y_test),'f1':f1_score(y_hat_test_RF,y_test)}

print(RF_dict)

print(np.nansum((np.array(X_test['1day_pct'].shift(-1))*((np.array(y_hat_test_RF)-1/2)*2))))

print(np.nansum((np.array(X_test['1day_pct'].shift(-1))*(np.array(y_hat_test_naive)))))


# ## XGB Classifier

# In[32]:


estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=2,
    seed=42
)
parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(20, 120, 10),
    'learning_rate': [0.001,0.003,0.01, 0.03, 0.1]
}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    n_jobs = 10,
    cv = tscv,
    verbose=True
)
grid_search.fit(X_train, y_train)
print(f"Training Accuracy: {grid_search.best_score_ :.2%}")
print("")
print(f"Optimal Parameters: {grid_search.best_params_}")

y_hat_test_XGB = grid_search.predict(X_test)
XGB_dict = {'model':'XGB','precision':precision_score(y_hat_test_XGB,y_test),'recall':recall_score(y_hat_test_XGB,y_test),
           'accuracy':accuracy_score(y_hat_test_XGB,y_test),'f1':f1_score(y_hat_test_XGB,y_test)}

print(XGB_dict)

print(np.nansum((np.array(X_test['1day_pct'].shift(-1))*((np.array(y_hat_test_XGB)-1/2)*2))))

print(np.nansum((np.array(X_test['1day_pct'].shift(-1))*np.ones(len(y_test)))))


# # Random Predictions

# In[15]:


RND = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()),'data','largecap','random.csv'))
RND.drop('Unnamed: 0',axis=1,inplace=True)
#RND.columns
RND_feat = create_features(RND)

RND_feat.head()


# In[24]:


RND_feat.columns


# In[25]:


y = RND_feat['target'].astype(int)
X = RND_feat.drop(['Date','Adj Close','High','Low','Close','target'],axis=1)

continuous = ['1day_pct', '2day_pct', '3day_pct', '4day_pct', '5day_pct', '7day_pct',
              '1day_pct_cs',
              'ewma_7', 'ewma_50', 'ewma_200', 'RSI', 'MACD','Volume','day_var','open_close','open_prev_close','high_close']
discrete = ['prev_hot_streak','prev_cold_streak', 'current_hot_streak', 'current_cold_streak',
            'RSI_overbought','RSI_oversold',
            #'7g(50&200)','7l(50&200)','7g50','7g200',
            'prev_current_hot', 'prev_current_cold','current_hot_prev_cold','current_cold_prev_hot',
            'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
            'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
            'dayowk_0', 'dayowk_1', 'dayowk_2', 'dayowk_3', 'dayowk_4',
           ]
scaler = StandardScaler()
X_cont = pd.DataFrame(scaler.fit_transform(X[continuous]),columns=continuous)

X_disc = X[discrete]

X_cont.reset_index(drop=True,inplace=True)

X_disc.reset_index(drop=True,inplace=True)

X = pd.concat([X_cont,X_disc],axis=1)

train_size = X.shape[0]*4//5
X_test = X.iloc[train_size:]
X_train = X.iloc[0:train_size]
y_test = y.iloc[train_size:]
y_train = y.iloc[0:train_size]

tscv = TimeSeriesSplit(n_splits=4)
print(tscv.split(X_train))
for train, val in tscv.split(X_train):
    print("%s %s" % (train, val))


# In[26]:


y_hat_test_naive = np.ones(len(y_test))
print('precision score' ,precision_score(y_hat_test_naive,y_test))
print('recall score' ,recall_score(y_hat_test_naive,y_test))
print('accuracy score' ,accuracy_score(y_hat_test_naive,y_test))
print('f1 score' ,f1_score(y_hat_test_naive,y_test))


# In[44]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','largecap','SPY.csv')
daily_prediction_analysis(fname)


# In[45]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','largecap','random.csv')
daily_prediction_analysis(fname)


# In[61]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','largecap','VOO.csv')
daily_prediction_analysis(fname)


# In[66]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','largecap','IVV.csv')
daily_prediction_analysis(fname)


# In[67]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','energy','XLE.csv')
daily_prediction_analysis(fname)


# In[68]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','health','XLV.csv')
daily_prediction_analysis(fname)


# In[14]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','GLD.csv')
daily_prediction_analysis(fname)


# In[15]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','random1.csv')
daily_prediction_analysis(fname)


# In[16]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','random2.csv')
daily_prediction_analysis(fname)


# In[17]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','random3.csv')
daily_prediction_analysis(fname)


# In[18]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','random4.csv')
daily_prediction_analysis(fname)


# In[20]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','random5.csv')
daily_prediction_analysis(fname)


# In[27]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','LQD.csv')
daily_prediction_analysis(fname)


# In[29]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','GLD.csv')
daily_prediction_analysis(fname)


# In[30]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','TIP.csv')
daily_prediction_analysis(fname)


# In[31]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','FXE.csv')
daily_prediction_analysis(fname)


# In[32]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','EEM.csv')
daily_prediction_analysis(fname)


# In[33]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','random1.csv')
daily_prediction_analysis(fname)


# In[34]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','random2.csv')
daily_prediction_analysis(fname)


# In[35]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','random3.csv')
daily_prediction_analysis(fname)


# In[45]:


start = datetime.datetime.now()
fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','random4.csv')
naive_list = []
RF_list = []
Log_list = []
SVM_list = []
stack_list = []
for i in range(0,40):
    print(i)
    naive_res,RF_res,Log_res,SVM_res,XGB_res,stack_res = daily_prediction_analysis(fname,do_forest=False)
    naive_list.append(naive_res)
    RF_list.append(RF_res)
    Log_list.append(Log_res)
    SVM_list.append(SVM_res)
    stack_list.append(stack_res)

end = datetime.datetime.now()
print(end-start)
  


# In[47]:


RF_ig_rand4 = []
Log_ig_rand4 = []
SVM_ig_rand4 = []
stack_ig_rand4 = []

for i in range(len(RF_list)):
    RF_ig_rand4.append(RF_list[i]['information gain'])
    Log_ig_rand4.append(Log_list[i]['information gain'])
    SVM_ig_rand4.append(SVM_list[i]['information gain'])
    stack_ig_rand4.append(stack_list[i]['information gain'])


# In[25]:


RF_ig_rand1 = []
Log_ig_rand1 = []
SVM_ig_rand1 = []
stack_ig_rand1 = []

for i in range(len(RF_list)):
    RF_ig_rand1.append(RF_list[i]['information gain'])
    Log_ig_rand1.append(Log_list[i]['information gain'])
    SVM_ig_rand1.append(SVM_list[i]['information gain'])
    stack_ig_rand1.append(stack_list[i]['information gain'])


# In[39]:


RF_ig_rand3 = []
Log_ig_rand3 = []
SVM_ig_rand3 = []
stack_ig_rand3 = []

for i in range(len(RF_list)):
    RF_ig_rand3.append(RF_list[i]['information gain'])
    Log_ig_rand3.append(Log_list[i]['information gain'])
    SVM_ig_rand3.append(SVM_list[i]['information gain'])
    stack_ig_rand3.append(stack_list[i]['information gain'])


# In[55]:


dictionary = {'RF1': RF_ig_rand1, 'RF2': RF_ig_rand2, 'RF3': RF_ig_rand3, 'RF4': RF_ig_rand4, 'RF5': RF_ig_rand5,
             'LOG1': Log_ig_rand1, 'LOG2': Log_ig_rand2,'LOG3': Log_ig_rand3,'LOG4': Log_ig_rand4,'LOG5': Log_ig_rand5,
             'SVM1':SVM_ig_rand1,'SVM2':SVM_ig_rand2,'SVM3':SVM_ig_rand3,'SVM4':SVM_ig_rand4,'SVM5':SVM_ig_rand5,
             'STACK1':stack_ig_rand1,'STACK2':stack_ig_rand2,'STACK3':stack_ig_rand3,'STACK4':stack_ig_rand4,
            'STACK5':stack_ig_rand5} 


# In[57]:


rand_df = pd.DataFrame(dictionary)


# In[58]:


rand_df.to_csv('rand_data.csv')  


# In[48]:


plt.hist([RF_ig_rand1,RF_ig_rand2,RF_ig_rand3,RF_ig_rand4,RF_ig_rand5])


# In[52]:


sns.displot([RF_ig_rand1,RF_ig_rand2,RF_ig_rand3,RF_ig_rand4,RF_ig_rand5],kde=True)


# In[13]:


filename = []
for i in range(0,16):
    filename.append(fname)


# In[14]:


start = datetime.datetime.now()
pool = Pool(8)
result = pool.map(daily_prediction_analysis, filename)
end = datetime.datetime.now()
print(end-start)


# In[16]:


res_24 = res_8.append(result)


# In[23]:


res_16 = res_8[-1]


# In[26]:


del res_8[-1]


# In[27]:


len(res_8)


# In[34]:


result_24 = res_8+res_16


# In[47]:


RF_ig = []
for i in range(24):
    RF_ig.append(result_24[i][1]['information gain'])


# In[48]:


RF_ig


# In[49]:


result_24[0][2]


# In[59]:


start = datetime.datetime.now()
fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','SPY.csv')
naive_list = []
RF_list = []
Log_list = []
SVM_list = []
stack_list = []
for i in range(0,40):
    print(i)
    naive_res,RF_res,Log_res,SVM_res,XGB_res,stack_res = daily_prediction_analysis(fname,do_forest=False)
    naive_list.append(naive_res)
    RF_list.append(RF_res)
    Log_list.append(Log_res)
    SVM_list.append(SVM_res)
    stack_list.append(stack_res)

end = datetime.datetime.now()
print(end-start)


# In[60]:


RF_ig_SPY = []
Log_ig_SPY = []
SVM_ig_SPY = []
stack_ig_SPY = []

for i in range(len(RF_list)):
    RF_ig_SPY.append(RF_list[i]['information gain'])
    Log_ig_SPY.append(Log_list[i]['information gain'])
    SVM_ig_SPY.append(SVM_list[i]['information gain'])
    stack_ig_SPY.append(stack_list[i]['information gain'])


# In[68]:


sns.displot([SVM_ig_rand5,SVM_ig_SPY],kde=True)


# In[2]:


import multiprocessing  
import random
from multiprocessing import Pool
import defs



if __name__ == '__main__':
    pool = Pool()
    to_factor = [ random.randint(100000, 50000000) for i in range(20)]
    results = pool.map(defs.prime_factor, to_factor)
    for value, factors in zip(to_factor, results):
        print("The factors of {} are {}".format(value, factors))


# In[9]:


fname = os.path.join(os.path.abspath(os.getcwd()),'data','10_ETF','random1.csv')


# In[3]:


import Classification_Function_test


# In[10]:


if __name__ == '__main__':
    pool = Pool()
    filename = []
    for i in range(0,2):
        filename.append(fname)
    results = pool.map(Classification_Function_test.daily_prediction_analysis, filename)
    for value, factors in zip(to_factor, results):
        print("The factors of {} are {}".format(value, factors))


# In[ ]:




