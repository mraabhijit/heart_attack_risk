# Import Libraries
import pickle
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Import Data
url = 'data/train.csv'

df = pd.read_csv(url)

# Parameters

output_file = f'model.bin'

# Data Preparation

df.columns = df.columns.str.replace(" ", '_').str.lower()

## Replace " " with "_" and lowercase the strings in categorical features

categorical = [x for x in df.columns if df[x].dtype == 'O']
for c in categorical:
    df[c] = df[c].str.replace(" ", '_').str.lower()

# Map Diabetes
diabetes_values = {
    0: 'yes',
    1: 'no'
}
df.diabetes = df.diabetes.map(diabetes_values)

# Map Family History
family_values = {
    1: 'yes',
    0: 'no'
}
df.family_history = df.family_history.map(family_values)

# Map Smoking
smoking_values = {
    1: 'yes',
    0: 'no'
}
df.smoking = df.smoking.map(smoking_values)

# Map Obesity
obesity_values = {
    1: 'yes',
    0: 'no'
}
df.obesity = df.obesity.map(obesity_values)

# Map Alcohol Consumption
alcohol_values = {
    1: 'yes',
    0: 'no'
}

df.alcohol_consumption = df.alcohol_consumption.map(alcohol_values)

# Map Previous Heart Problems
values = {
    1: 'yes',
    0: 'no'
}
df.previous_heart_problems = df.previous_heart_problems.map(values)

# Map Medication Use
values = {
    1: 'yes',
    0: 'no'
}
df.medication_use = df.medication_use.map(values)
df.medication_use.head()

# Separate Out Systolic and Diastolic Blood Pressure Values
df[['systolic_bp', 'diastolic_bp']] = df['blood_pressure'].str.split('/', expand=True).astype(int)
df.drop(columns = 'blood_pressure', inplace=True)

# Drop unwanted Features
df.drop(columns = ['continent', 'hemisphere', 'patient_id'], inplace=True)

# Rrename `sex` to `gender`
df.rename(columns = {'sex': 'gender'}, inplace=True)


categorical = [x for x in df.columns if df[x].dtype == 'O']
numerical = [x for x in df.columns if x not in categorical]
numerical.remove('heart_attack_risk')


## Setting up Validation Framework

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=12)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=12)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = df_train.heart_attack_risk.values
y_val = df_val.heart_attack_risk.values
y_test = df_test.heart_attack_risk.values

del df_train['heart_attack_risk']
del df_val['heart_attack_risk']
del df_test['heart_attack_risk']


## Model Building and Cross Validation

def train(df_train, y_train, model):
    dicts = df_train[categorical+numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model.fit(X_train, y_train)
    return dv, model


def predict(df_val, dv, model):
    dicts = df_val[categorical+numerical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    y_pred = model.predict_proba(X_val)[:, 1]

    return y_pred


# Logistic Regression

lr = LogisticRegression(C=0.001, solver='newton-cholesky', penalty='l2', max_iter=10)

# Evaluation
scores_cv = []
df_full = df_full_train.copy()
kfold = KFold(n_splits = 10, shuffle=True, random_state=1)
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full.iloc[train_idx]
    df_val = df_full.iloc[val_idx]

    y_train = df_train.heart_attack_risk.values
    y_val = df_val.heart_attack_risk.values
    
    del df_train['heart_attack_risk']
    del df_val['heart_attack_risk']
    
    dv, model = train(df_train, y_train, lr)
    
    y_pred = predict(df_train, dv, lr)
    auc_train = roc_auc_score(y_train, y_pred)

    y_pred = predict(df_val, dv, lr)
    auc_val = roc_auc_score(y_val, y_pred)
    
    scores_cv.append((auc_train, auc_val))

df_scores_cv = pd.DataFrame(scores_cv, columns = ['Train_AUC', 'Val_AUC'])
print("Logistic Regression")
print(f"\tTrain: {round(df_scores_cv.Train_AUC.mean(), 3)} +- {round(df_scores_cv.Train_AUC.std(), 3)}")
print(f"\tVal: {round(df_scores_cv.Val_AUC.mean(), 3)} +- {round(df_scores_cv.Val_AUC.std(), 3)}")
print()


# Decision Trees

dt = DecisionTreeClassifier(criterion='gini',
                       max_depth=4,
                       min_samples_leaf=15,
                       random_state=12)

# Evaluation
scores_cv = []
df_full = df_full_train.copy()
kfold = KFold(n_splits = 10, shuffle=True, random_state=1)
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full.iloc[train_idx]
    df_val = df_full.iloc[val_idx]

    y_train = df_train.heart_attack_risk.values
    y_val = df_val.heart_attack_risk.values
    
    del df_train['heart_attack_risk']
    del df_val['heart_attack_risk']
    
    dv, model = train(df_train, y_train, dt)
    
    y_pred = predict(df_train, dv, dt)
    auc_train = roc_auc_score(y_train, y_pred)

    y_pred = predict(df_val, dv, dt)
    auc_val = roc_auc_score(y_val, y_pred)
    
    scores_cv.append((auc_train, auc_val))

df_scores_cv = pd.DataFrame(scores_cv, columns = ['Train_AUC', 'Val_AUC'])
print("Decision Trees")
print(f"\tTrain: {round(df_scores_cv.Train_AUC.mean(), 3)} +- {round(df_scores_cv.Train_AUC.std(), 3)}")
print(f"\tVal: {round(df_scores_cv.Val_AUC.mean(), 3)} +- {round(df_scores_cv.Val_AUC.std(), 3)}")
print()


# Random Forest

rf = RandomForestClassifier(n_estimators=9, 
                            max_depth=2,
                            min_samples_leaf=20,
                            max_features=35,
                            random_state=1)

# Evaluation
scores_cv = []
df_full = df_full_train.copy()
kfold = KFold(n_splits = 10, shuffle=True, random_state=1)
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full.iloc[train_idx]
    df_val = df_full.iloc[val_idx]

    y_train = df_train.heart_attack_risk.values
    y_val = df_val.heart_attack_risk.values
    
    del df_train['heart_attack_risk']
    del df_val['heart_attack_risk']
    
    dv, model = train(df_train, y_train, rf)
    
    y_pred = predict(df_train, dv, rf)
    auc_train = roc_auc_score(y_train, y_pred)

    y_pred = predict(df_val, dv, rf)
    auc_val = roc_auc_score(y_val, y_pred)
    
    scores_cv.append((auc_train, auc_val))

df_scores_cv = pd.DataFrame(scores_cv, columns = ['Train_AUC', 'Val_AUC'])
print("Random Forest")
print(f"\tTrain: {round(df_scores_cv.Train_AUC.mean(), 3)} +- {round(df_scores_cv.Train_AUC.std(), 3)}")
print(f"\tVal: {round(df_scores_cv.Val_AUC.mean(), 3)} +- {round(df_scores_cv.Val_AUC.std(), 3)}")
print()


# XGBoost

xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 2,
    'seed': 1,
    'verbosity': 1
}

# evaluation
scores_cv = []
df_full = df_full_train.copy()
kfold = KFold(n_splits = 10, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full.iloc[train_idx]
    df_val = df_full.iloc[val_idx]

    y_train = df_train.heart_attack_risk.values
    y_val = df_val.heart_attack_risk.values

    dv = DictVectorizer(sparse=False)
    
    dicts_train = df_train[categorical+numerical].to_dict(orient='records')
    X_train = dv.fit_transform(dicts_train)

    dicts_val = df_val[categorical+numerical].to_dict(orient='records')
    X_val = dv.transform(dicts_val)

    dtrain = xgb.DMatrix(X_train, label=y_train,
                   feature_names=list(dv.get_feature_names_out()))

    dval = xgb.DMatrix(X_val, 
                    feature_names=list(dv.get_feature_names_out()))

    model = xgb.train(xgb_params, dtrain, num_boost_round=13)
    
    y_pred = model.predict(dtrain)
    auc_train = roc_auc_score(y_train, y_pred)

    y_pred = model.predict(dval)
    auc_val = roc_auc_score(y_val, y_pred)
    
    scores_cv.append((auc_train, auc_val))

df_scores_cv = pd.DataFrame(scores_cv, columns = ['Train_AUC', 'Val_AUC'])
print("XGBoost")
print(f"\tTrain: {round(df_scores_cv.Train_AUC.mean(), 3)} +- {round(df_scores_cv.Train_AUC.std(), 3)}")
print(f"\tVal: {round(df_scores_cv.Val_AUC.mean(), 3)} +- {round(df_scores_cv.Val_AUC.std(), 3)}")
print()

# Training the final model

print('Training the final model')

lr = LogisticRegression(C = 0.001, 
                        solver = 'newton-cholesky', 
                        penalty = 'l2', 
                        max_iter = 10)

# dt = DecisionTreeClassifier(criterion='gini',
#                        max_depth=4,
#                        min_samples_leaf=15,
#                        random_state=12)

# rf = RandomForestClassifier(n_estimators=9, 
#                             max_depth=2,
#                             min_samples_leaf=20,
#                             max_features=35,
#                             random_state=1)

# dv = DictVectorizer(sparse=False)

# dicts_train = df_full_train[categorical+numerical].to_dict(orient='records')
# X_train = dv.fit_transform(dicts_train)

# dicts_test = df_test[categorical+numerical].to_dict(orient='records')
# X_test = dv.transform(dicts_test)

# dtrain = xgb.DMatrix(X_train, label=df_full_train['heart_attack_risk'].values,
#                 feature_names=list(dv.get_feature_names_out()))

# dtest = xgb.DMatrix(X_test, 
#                 feature_names=list(dv.get_feature_names_out()))

# model = xgb.train(xgb_params, dtrain, num_boost_round=13)

# y_pred = model.predict(dtest)
# auc_val = roc_auc_score(y_test, y_pred)


dv, model = train(df_full_train, df_full_train.heart_attack_risk.values, lr)
y_pred = predict(df_test, dv, lr)

auc = roc_auc_score(y_test, y_pred)

print(f'auc = {round(auc, 3)}')


# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved to {output_file}')