#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[156]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")


# ## Import Data
# 
# Dataset url: https://github.com/abhijitchak103/heart_attack_risk/blob/main/data/train.csv

# In[2]:


url = 'data/train.csv'


# In[3]:


df = pd.read_csv(url)


# In[4]:


df.head().T


# ## Data Description

# The features for preparing the predictors:
# 
# - `Patient ID`: Unique identifier for each patienttack risk (1: Yes, 0: No)ack risk (1: Yes, 0: No)

# - `Age`: Age of the patient

# - `Sex`: Gender of the patient (Male/Female)

# - `Cholesterol`: Cholesterol levels of the patient

# - `Blood Pressure`: Blood pressure of the patient (systolic/diastolic)  

# - `Heart Rate`: Heart rate of the patient

# - `Diabetes`: Whether the patient has diabetes (Yes/No)

# - `Family History`: Family history of heart-related problems (1: Yes, 0: No)

# - `Smoking`: Smoking status of the patient (1: Smoker, 0: Non-smoker)

# - `Obesity`: Obesity status of the patient (1: Obese, 0: Not obese)

# - `Alcohol Consumption`: Patient consumes alcohol (Yes/No)

# - `Exercise Hours Per Week`: Number of exercise hours per week

# - `Diet`: Dietary habits of the patient (Healthy/Average/Unhealthy)

# - `Previous Heart Problems`: Previous heart problems of the patient (1: Yes, 0: No)

# - `Medication Use`: Medication usage by the patient (1: Yes, 0: No)

# - `Stress Level`: Stress level reported by the patient (1-10)

# - `Sedentary Hours Per Day`: Hours of sedentary activity per day

# - `Income`: Income level of the patient

# - `BMI`: Body Mass Index (BMI) of the patient

# - `Triglycerides`: Triglyceride levels of the patient

# - `Physical Activity Days Per Week`: Days of physical activity per week

# - `Sleep Hours Per Day`: Hours of sleep per day

# - `Country`: Country of the patient

# - `Continent`: Continent where the patient resides

# - `Hemisphere`: Hemisphere where the patient resides

# Target Feature:
# 
# `Heart Attack Risk`: Presence of heart attack risk (1: Yes, 0: No)

# ## Data Preparation
# 
# - Rename Columns to remove spaces and capital letters
# - Lowercase string entries in rows and replace spaces with underscores
# - Convert numeric categoric entries to stings

# In[5]:


df.columns = df.columns.str.replace(" ", '_').str.lower()


# In[6]:


categorical = [x for x in df.columns if df[x].dtype == 'O']


# In[7]:


for c in categorical:
    df[c] = df[c].str.replace(" ", '_').str.lower()


# In[8]:


df.head().T


# In[9]:


df.diabetes.value_counts()


# In[10]:


# Diabetes: Whether the patient has diabetes (Yes/No)

diabetes_values = {
    0: 'yes',
    1: 'no'
}
df.diabetes = df.diabetes.map(diabetes_values)
df.diabetes.head()


# In[11]:


df.family_history.value_counts()


# In[12]:


# Family History: Family history of heart-related problems (1: Yes, 0: No)

family_values = {
    1: 'yes',
    0: 'no'
}
df.family_history = df.family_history.map(family_values)
df.family_history.head()


# In[13]:


df.smoking.value_counts()


# In[14]:


# Smoking: Smoking status of the patient (1: Smoker, 0: Non-smoker)

smoking_values = {
    1: 'yes',
    0: 'no'
}
df.smoking = df.smoking.map(smoking_values)
df.smoking.head()


# In[15]:


df.obesity.value_counts()


# In[16]:


# Obesity: Obesity status of the patient (1: Obese, 0: Not obese)

obesity_values = {
    1: 'yes',
    0: 'no'
}
df.obesity = df.obesity.map(obesity_values)
df.obesity.head()


# In[17]:


df.alcohol_consumption.value_counts()


# In[18]:


# Alcohol Consumption: Patient consumes alcohol (Yes/No)

alcohol_values = {
    1: 'yes',
    0: 'no'
}

df.alcohol_consumption = df.alcohol_consumption.map(alcohol_values)
df.alcohol_consumption.head()


# In[19]:


df.previous_heart_problems.value_counts()


# In[20]:


# Previous Heart Problems: Previous heart problems of the patient (1: Yes, 0: No)

values = {
    1: 'yes',
    0: 'no'
}
df.previous_heart_problems = df.previous_heart_problems.map(values)
df.previous_heart_problems.head()


# In[21]:


df.medication_use.value_counts()


# In[22]:


# Medication Use: Medication usage by the patient (1: Yes, 0: No)

values = {
    1: 'yes',
    0: 'no'
}
df.medication_use = df.medication_use.map(values)
df.medication_use.head()


# In[23]:


df.isnull().sum()


# We do not have null values

# In[24]:


df.info()


# We see that `blood_pressure` is of type `object` whereas it should be `numeric`. We can correct it by splitting the `systolic` and `diastolic` to different columns.

# In[25]:


df[['systolic_bp', 'diastolic_bp']] = df['blood_pressure'].str.split('/', expand=True).astype(int)


# In[26]:


df.head(2)


# In[27]:


df.drop(columns = 'blood_pressure', inplace=True)


# In[28]:


df.info()


# We can get a fairly good idea about `continent` and `hemisphere` from `country`. So we can drop the two features. Also, we can drop `patient_id`.

# In[29]:


df.drop(columns = ['continent', 'hemisphere', 'patient_id'], inplace=True)


# We can also rename `sex` to `gender`

# In[30]:


df.rename(columns = {'sex': 'gender'}, inplace=True)


# In[31]:


categorical = [x for x in df.columns if df[x].dtype == 'O']
numerical = [x for x in df.columns if x not in categorical]

assert len(categorical) + len(numerical) == len(df.columns)


# ## Setting up Validation Framework

# In[172]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=12)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=12)


# In[173]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[174]:


y_train = df_train.heart_attack_risk.values
y_val = df_val.heart_attack_risk.values
y_test = df_test.heart_attack_risk.values


# In[175]:


del df_train['heart_attack_risk']
del df_val['heart_attack_risk']
del df_test['heart_attack_risk']


# In[36]:


len(df_train), len(df_val), len(df_test)


# In[37]:


len(y_train), len(y_val), len(y_test)


# ## Exploratory Data Analysis
# 
# - Mising Values
# - Distribution of Target Variable
# - Feature Importance - Mutual Inforamtion
# - Feature Importance - Correlation

# In[38]:


df_full_train = df_full_train.reset_index(drop=True)


# In[39]:


df_full_train.isnull().sum()


# In[40]:


df_full_train.heart_attack_risk.value_counts(normalize=True)


# We can calculate the global heart attack risk rate. `Global` here refers to the entire dataset.

# In[41]:


global_heart_attack_risk_rate = df_full_train.heart_attack_risk.mean()
round(global_heart_attack_risk_rate, 4)


# In[42]:


numerical.remove('heart_attack_risk')
numerical


# ### Feature Importance - Risk and Rate

# We can check the risk among individual groups now and see how groups influence the risk rate.

# In[43]:


df_full_train[categorical].nunique()


# In[44]:


from IPython.display import display


# In[45]:


categories = categorical.copy()
categories.remove('country')
for c in categories:
    print(c)
    df_group = df_full_train.groupby(c).heart_attack_risk.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_heart_attack_risk_rate
    df_group['risk'] = df_group['mean'] / global_heart_attack_risk_rate
    display(df_group)
    print()


# We cannot clearly see a demarcation on which group influences the risk of heart attacks among the various categorical features.

# ### Feature Importance - Mutual Information
# 
# `Mutual Information` measures how much information we can gather by studying another variable.

# In[46]:


def mutual_score(series):
    return mutual_info_score(series, df_full_train.heart_attack_risk)


# In[47]:


mi = df_full_train[categorical].apply(mutual_score)
mi_sorted = mi.sort_values(ascending=False)


# In[48]:


mi_5 = mi_sorted.index[:5]
mi_5


# The top 5 categorical features for providing information about `heart attack risk` are:
# `country`, `diabetes`, `diet`, `obesity` and `alcohol_consumption`

# ### Feature Importance - Correlation
# 
# For numerical features, we can use correlation values.

# In[49]:


corr = df_full_train[numerical].corrwith(df_full_train.heart_attack_risk).abs()
corr_sorted = corr.sort_values(ascending=False)


# In[50]:


corr_sorted


# In[51]:


corr_5 = corr_sorted.index[:5]
corr_5


# The top 5 numerical features for providing information about `heart attack risk` are:
# `cholesterol`, `systolic_bp`, `sleep_hours_per_day`, `triglycerides` and `income`

# ## Encoding Variables

# In[52]:


dv = DictVectorizer(sparse=False)


# In[53]:


features_top_10 = list(corr_5) + list(mi_5)


# In[54]:


# Training with Top 5 features of Categorical and Numerical

train_dicts_10 = df_train[features_top_10].to_dict(orient='records')
X_train_10 = dv.fit_transform(train_dicts_10)

val_dicts_10 = df_val[features_top_10].to_dict(orient='records')
X_val_10 = dv.transform(val_dicts_10)


# In[55]:


# Training all the features

train_dicts = df_train[categorical +numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)


# ## Logistic Regression

# In[56]:


score = {}


# In[57]:


lr = LogisticRegression()


# In[58]:


lr.fit(X_train_10, y_train)


# In[59]:


y_pred = lr.predict_proba(X_val_10)[:, 1]


# In[60]:


score['log_reg_10'] = roc_auc_score(y_val, y_pred)
score['log_reg_10']


# In[61]:


lr.fit(X_train, y_train)
y_pred = lr.predict_proba(X_val)[:, 1]


# In[62]:


score['log_reg'] = roc_auc_score(y_val, y_pred)
score['log_reg']


# The model with full set of features does around 2% better than only selected features

# ### Tuning Logistic Regression Model

# - Solvers

# In[63]:


solvers = ['liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga', 'lbfgs']


# In[64]:


for s in solvers:
    lr = LogisticRegression(solver=s)
    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f'solver: {s} --> auc: {auc}')
    if auc > score['log_reg']:
        score['log_reg'] = auc    


# So, the `newton-cholesky` does the best in increasing auc scores.
# 
# We can try, with using `None` penalty as by default `l2` is being used.

# In[65]:


for p in ['l2', None]:
    lr = LogisticRegression(solver='newton-cholesky', penalty=p)
    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f'penalty: {p} --> auc: {auc}')
    if auc > score['log_reg']:
        score['log_reg'] = auc


# ### Logistic Regression with Feature Scaling

# In[66]:


def scale_numeric(df_in, numeric_features, categoric_features, scaler = StandardScaler()):
    numeric_df = df_in[numeric_features]
    categoric_df = df_in[categoric_features]
    train_data = scaler.fit(numeric_df)
    train_data = scaler.transform(numeric_df)
    numeric_df = pd.DataFrame(train_data, columns = numeric_features)
    df_out = pd.concat([numeric_df, categoric_df], axis=1)
    return df_out, scaler


# In[67]:


df_train_scale, scaler = scale_numeric(df_train, numerical, categorical)


# In[68]:


def scale_val(df_in, numeric_features, categoric_features, scaler):
    numeric_df = df_in[numeric_features]
    categoric_df = df_in[categoric_features]
    numeric_df = pd.DataFrame(scaler.transform(numeric_df), columns = numeric_features)
    df_out = pd.concat([numeric_df, categoric_df], axis=1)
    return df_out


# In[69]:


df_val_scale = scale_val(df_val, numerical, categorical, scaler)


# In[70]:


dv = DictVectorizer(sparse=False)
train_dicts_scale = df_train_scale[categorical +numerical].to_dict(orient='records')
X_train_scale = dv.fit_transform(train_dicts_scale)

val_dicts_scale = df_val_scale[categorical + numerical].to_dict(orient='records')
X_val_scale = dv.transform(val_dicts_scale)


# In[71]:


lr = LogisticRegression(solver='newton-cholesky', penalty='l2')


# In[72]:


lr.fit(X_train_scale, y_train)
y_pred = lr.predict_proba(X_val_scale)[:, 1]
auc = roc_auc_score(y_val, y_pred)
auc


# Feature Scaling does not improve the model performance. So we stick with the data without Scaling.

# - `C`

# In[73]:


for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    lr =  LogisticRegression(C=c, solver='newton-cholesky', penalty='l2')
    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_pred)
    y_pred = lr.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, y_pred)
    print(f"C: {c:>4},Train auc: {round(auc_train, 3)}, Val auc: {round(auc_val, 3)}")


# - `max_iter`

# In[74]:


for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    print()
    print(f"C={c}")
    for m in [10, 100, 1000, 10000, 100000]:
        lr =  LogisticRegression(C=c, solver='newton-cholesky', penalty='l2', max_iter=m)
        lr.fit(X_train, y_train)
        y_pred = lr.predict_proba(X_train)[:, 1]
        auc_train = roc_auc_score(y_train, y_pred)
        y_pred = lr.predict_proba(X_val)[:, 1]
        auc_val = roc_auc_score(y_val, y_pred)
        print(f"Max iterations: {m:>6}, Train auc: {round(auc_train, 3)}, Val auc: {round(auc_val, 3)}")
    print()


# For `C`=0.001, the model yields the best performance. hence, we fix the Logistic Regression model at:
# ```python
# LogisticRegression(C=0.001, solver='newton-cholesky', penalty='l2', max_iter=10)
# ```

# ## Decision Trees

# In[75]:


dt_10 = DecisionTreeClassifier()
dt = DecisionTreeClassifier()


# In[76]:


dt_10.fit(X_train_10, y_train)
y_pred = dt_10.predict_proba(X_val_10)[:, 1]
auc = roc_auc_score(y_val, y_pred)
auc


# In[77]:


dt.fit(X_train, y_train)
y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
auc


# We can see that the Decision Tree with all features work slightly better. We can further tune the model.

# ## Parameter Tuning for Decision Trees

# - `Criterion`: [`gini`, `entropy`, `log_loss`]

# In[78]:


criteria = ['gini', 'entropy', 'log_loss']


# In[79]:


for c in criteria:
    dt = DecisionTreeClassifier(criterion=c,
                                random_state=12)
    dt.fit(X_train, y_train)
    y_pred = dt.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"Criterion: {c:>8}\t-->\tAUC: {round(auc,4)}")


# We can stick with `gini` as the criteria.

# - `max_depth`: [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, None]

# In[80]:


depth = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, None]


# In[81]:


for d in depth:
    dt = DecisionTreeClassifier(criterion='gini',
                                max_depth=d,
                                random_state=12)
    dt.fit(X_train, y_train)
    y_pred = dt.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"Max_depth: {d}\t-->\tAUC: {round(auc,4)}")


# `max_depth` = 4

# - `min_samples_leaf`: [1, 2, 3, 4, 5, 10]

# In[82]:


leaves = [1, 2, 3, 4, 5, 10, 50, 100, 200, 500]


# In[83]:


for l in leaves:
    dt = DecisionTreeClassifier(criterion='gini',
                                max_depth=4,
                                min_samples_leaf=l,
                                random_state=12)
    dt.fit(X_train, y_train)
    y_pred = dt.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"Min_sampples_leaf: {l:>3}\t-->\tAUC: {round(auc,4)}")


# `min_sample_leaf` of 1, 2 and 3 have similar scores. We can go with 2.

# Further we can combine and check for combinations of `max_depth` and `min_samples_leaf`.

# In[84]:


dt_scores = []
for d in [2,3,4,5,6,7, 8, 9, 10, 20, None]:
    for l in [1,2,3,4,5,10,15,20,30,40,50]:
        dt = DecisionTreeClassifier(criterion='gini',
                                max_depth=d,
                                min_samples_leaf=l,
                                random_state=12)
        dt.fit(X_train, y_train)
        y_pred = dt.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, y_pred)
        dt_scores.append((d, l, round(auc, 4)))
        # print(f"Min_sampples_leaf: {l:>3}\t-->\tAUC: {round(auc,4)}")

df_dt = pd.DataFrame(dt_scores, columns = ['max_depth', 'min_samples_leaf', 'auc'])
df_dt.sort_values(by='auc', ascending=False).head()


# In[85]:


dt_pivot = df_dt.pivot(index='min_samples_leaf', columns='max_depth', values='auc')
dt_pivot


# In[86]:


plt.figure(figsize =(10,10))
sns.heatmap(dt_pivot, annot=True, fmt='.3f')


# As evident from the heatmap, the `max_depth` of 4 and `min_samples_leaf` of 15 yields the best model out of the ranges tested. We finalize our `DecisionTreeClassifier` model as:
# 
# ```python
# DecisionTreeClassifier(criterion='gini',
#                        max_depth=4,
#                        min_samples_leaf=15,
#                        random_state=12)
# ```

# In[87]:


dt = DecisionTreeClassifier(criterion='gini',
                       max_depth=4,
                       min_samples_leaf=15,
                       random_state=12)
dt.fit(X_train, y_train)
y_pred = dt.predict_proba(X_train)[:, 1]
auc_train = roc_auc_score(y_train, y_pred)
y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
score['dt'] = auc
print(f"auc_train: {round(auc_train, 3)}, auc_val: {round(auc, 3)}")


# ## Random Forest Classifier

# In[88]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_train)[:, 1]
auc_train = roc_auc_score(y_train, y_pred)
y_pred = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print(f"auc_train: {round(auc_train, 3)}, auc_val: {round(auc, 3)}")


# ### Random Forest Tuning

# - `n_estimators`

# In[89]:


estimators = range(10, 201, 10)


# In[90]:


scores_rf = []
for n in estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_pred)
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"n_estimators: {n:>3}, auc_train: {round(auc_train, 3)}, auc_val: {round(auc, 3)}")
    scores_rf.append((n, auc))

df_scores = pd.DataFrame(scores_rf, columns = ['n_estimators', 'auc'])
display(df_scores.sort_values(by='auc', ascending=False))

plt.plot(df_scores.n_estimators, df_scores.auc)
plt.xlabel("# Trees")
plt.ylabel("AUC")
plt.show()


# We can refine the `n_estimators` by magnifying the scope between 10 and 30.

# In[91]:


estimators = range(10, 31, 2)


# In[92]:


scores_rf = []
for n in estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_pred)
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"n_estimators: {n:>3}, auc_train: {round(auc_train, 3)}, auc_val: {round(auc, 3)}")
    scores_rf.append((n, auc))
df_scores = pd.DataFrame(scores_rf, columns = ['n_estimators', 'auc'])
display(df_scores)

plt.plot(df_scores.n_estimators, df_scores.auc)
plt.xlabel("# Trees")
plt.ylabel("AUC")
plt.show()


# In[93]:


estimators = range(20, 25, 1)


# In[94]:


scores_rf = []
for n in estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_pred)
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"n_estimators: {n:>3}, auc_train: {round(auc_train, 3)}, auc_val: {round(auc, 3)}")
    scores_rf.append((n, auc))
df_scores = pd.DataFrame(scores_rf, columns = ['n_estimators', 'auc'])
display(df_scores)

plt.plot(df_scores.n_estimators, df_scores.auc)
plt.xlabel("# Trees")
plt.ylabel("AUC")
plt.show()


# In[95]:


estimators = range(5, 10, 1)


# In[96]:


scores_rf = []
for n in estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_pred)
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"n_estimators: {n:>3}, auc_train: {round(auc_train, 3)}, auc_val: {round(auc, 3)}")
    scores_rf.append((n, auc))
df_scores = pd.DataFrame(scores_rf, columns = ['n_estimators', 'auc'])
display(df_scores)

plt.plot(df_scores.n_estimators, df_scores.auc)
plt.xlabel("# Trees")
plt.ylabel("AUC")
plt.show()


# `n_estimators`: 21

# - `max_depth`

# In[97]:


depth = [2, 3, 4, 5, 10, 15, 20]


# In[98]:


scores_rf = []
for d in depth:
    for n in estimators:
        rf = RandomForestClassifier(n_estimators=n, 
                                    max_depth=d,
                                    random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_train)[:, 1]
        auc_train = roc_auc_score(y_train, y_pred)
        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        # print(f"Max_depth: {d:>2}, auc_train: {round(auc_train, 3)}, auc_val: {round(auc, 3)}")
        scores_rf.append((d, n, auc_train, auc))
df_scores = pd.DataFrame(scores_rf, columns = ['max_depth', 'n_estimators', 'train_auc', 'val_auc'])
display(df_scores.sort_values(by=['val_auc', 'train_auc'], ascending=False))


# `max_depth`: 2, `n_estimators`: 9

# - `min_sample_leaf`

# In[99]:


leaves = [5, 10, 15, 20, 25]


# In[100]:


scores_rf = []
for s in leaves:
    rf = RandomForestClassifier(n_estimators=9, 
                                max_depth=2,
                                min_samples_leaf=s,
                                random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_pred)
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores_rf.append((s, n, auc_train, auc))
df_scores = pd.DataFrame(scores_rf, columns = ['min_samples_leaf', 'n_estimators', 'train_auc', 'val_auc'])
display(df_scores.sort_values(by=['val_auc', 'train_auc'], ascending=False))


# `min_samples_leaf`: 20

# - `max_features`

# In[101]:


num_features = list(range(5, len(dv.get_feature_names_out()) + 1, 5)) + [None]


# In[102]:


scores_rf = []
for f in num_features:
    rf = RandomForestClassifier(n_estimators=9, 
                                max_depth=2,
                                min_samples_leaf=20,
                                max_features=f,
                                random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_pred)
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores_rf.append((f, auc_train, auc))
df_scores = pd.DataFrame(scores_rf, columns = ['max_features', 'train_auc', 'val_auc'])
display(df_scores.sort_values(by=['val_auc', 'train_auc'], ascending=False))


# In[103]:


num_features = range(25, 51, 1)


# In[104]:


scores_rf = []
for f in num_features:
    rf = RandomForestClassifier(n_estimators=9, 
                                max_depth=2,
                                min_samples_leaf=20,
                                max_features=f,
                                random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_pred)
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores_rf.append((f, auc_train, auc))
df_scores = pd.DataFrame(scores_rf, columns = ['max_features', 'train_auc', 'val_auc'])
display(df_scores.sort_values(by=['val_auc', 'train_auc'], ascending=False).head())


# `max_features`: 35

# Final Random Forest model:
# ```python
#     rf = RandomForestClassifier(n_estimators=9, 
#                                 max_depth=2,
#                                 min_samples_leaf=20,
#                                 max_features=35,
#                                 random_state=1)
# ```

# In[105]:


rf = RandomForestClassifier(n_estimators=9, 
                            max_depth=2,
                            min_samples_leaf=20,
                            max_features=35,
                            random_state=1)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_train)[:, 1]
auc_train = roc_auc_score(y_train, y_pred)
y_pred = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
score['rf'] = auc
print(f"auc_train: {round(auc_train, 3)}, auc_val: {round(auc, 3)}")


# In[106]:


score


# ## XGBoost

# In[107]:


features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)


# In[108]:


watchlist = [(dtrain, 'train'), (dval, 'val')]


# In[109]:


def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        # print(line)
        it_line, train_line, val_line = line.split('\t')
        it = int(it_line.strip('[]'))
        train_auc = float(train_line.split(':')[1])
        val_auc = float(val_line.split(':')[1])

        results.append((it, train_auc, val_auc))

    out_df = pd.DataFrame(results, columns = ['iter_num', 'train_auc', 'val_auc'])

    return out_df


# In[110]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3,\n    'max_depth': 4,\n    'min_child_weight': 1,\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 2,\n    'seed': 1,\n    'verbosity': 1\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                 evals = watchlist)\n")


# In[111]:


scores_xgb = parse_xgb_output(output)


# In[112]:


plt.plot(scores_xgb.iter_num, scores_xgb.train_auc, label='train')
plt.plot(scores_xgb.iter_num, scores_xgb.val_auc, label='val')
plt.legend()
plt.show()


# ### XGBoost Parameter Tuning

# - `eta`

# In[113]:


learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]


# In[114]:


get_ipython().run_cell_magic('capture', 'output', "scores = {}\n\nfor e in learning_rates:\n    xgb_params = {\n        'eta': e,\n        'max_depth': 4,\n        'min_child_weight': 1,\n        'objective': 'binary:logistic',\n        'eval_metric': 'auc',\n        'nthread': 2,\n        'seed': 1,\n        'verbosity': 1\n    }\n    \n    model = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                     evals = watchlist)\n    key = 'eta=%s' % (xgb_params['eta'])\n    scores[key] = parse_xgb_output(output)\n")


# In[115]:


scores.keys()


# In[116]:


for key, df_score in scores.items():
    plt.plot(df_score.iter_num, df_score.train_auc, label=key)
plt.legend()
plt.show()


# In[117]:


for key, df_score in scores.items():
    plt.plot(df_score.iter_num, df_score.val_auc, label=key)
plt.legend()
plt.show()


# As there is no significant change in performance among different learning rates, we can select 0.1 as a learning rate, which is neither too slow nor too fast.

# - `max_depth`

# In[118]:


scores = {}


# In[119]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1,\n    'max_depth': 10,\n    'min_child_weight': 1,\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 2,\n    'seed': 1,\n    'verbosity': 1\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                 evals = watchlist)\n")


# In[120]:


key = 'max_depth=%s' % (xgb_params['max_depth'])
scores[key] = parse_xgb_output(output)


# In[121]:


for key, df_score in scores.items():
    plt.plot(df_score.iter_num, df_score.train_auc, label=key)
plt.legend()
plt.show()


# In[122]:


for key, df_score in scores.items():
    plt.plot(df_score.iter_num, df_score.val_auc, label=key)
plt.legend()
plt.show()


# It looks like at `max_depth=3`, the training auc is not too high while the validation auc is within acceptable ranges. However, to get a better picture, we can limit the boost rounds to 25 as the model reaches its peak within 25 rounds.

# In[123]:


get_ipython().run_cell_magic('capture', 'output', "scores = {}\n\nxgb_params = {\n    'eta': 0.1,\n    'max_depth': 3,\n    'min_child_weight': 1,\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 2,\n    'seed': 1,\n    'verbosity': 1\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=25,\n                 evals = watchlist)\n")


# In[124]:


key = 'max_depth=%s' % (xgb_params['max_depth'])
scores[key] = parse_xgb_output(output)
for key, df_score in scores.items():
    plt.plot(df_score.iter_num, df_score.train_auc, label='training')
    plt.plot(df_score.iter_num, df_score.val_auc, label='validation')
plt.legend()
plt.show()


# It looks like at boost_rounds=13, we get the best validation_auc score.

# - `min_child_weight`

# In[125]:


scores = {}


# In[126]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1,\n    'max_depth': 3,\n    'min_child_weight': 30,\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 2,\n    'seed': 1,\n    'verbosity': 1\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=20,\n                 evals = watchlist)\n")


# In[127]:


key = 'min_child_weight=%s' % (xgb_params['min_child_weight'])
scores[key] = parse_xgb_output(output)


# In[128]:


for key, df_score in scores.items():
    plt.plot(df_score.iter_num, df_score.train_auc, label=key)
plt.legend()
plt.show()


# In[129]:


for key, df_score in scores.items():
    plt.plot(df_score.iter_num, df_score.val_auc, label=key)
plt.legend()
plt.show()


# `min_child_weights=1` can be used with `num_boost_round=13` and `max_depth=3`
# 
# ```python
# xgb_params = {
#     'eta': 0.1,
#     'max_depth': 3,
#     'min_child_weight': 1,
#     'objective': 'binary:logistic',
#     'eval_metric': 'auc',
#     'nthread': 2,
#     'seed': 1,
#     'verbosity': 1
# }
# 
# model = xgb.train(xgb_params, dtrain, num_boost_round=13)
# ```

# In[130]:


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

model = xgb.train(xgb_params, dtrain, num_boost_round=13)


# In[131]:


y_pred = model.predict(dtrain)
auc_train = roc_auc_score(y_train, y_pred)
y_pred = model.predict(dval)
auc = roc_auc_score(y_val, y_pred)
score['xgb']=auc
print(f"Train AUC: {auc_train}, Val AUC: {auc}")


# ## Cross Validation

# In[176]:


def train(df_train, y_train, model):
    dicts = df_train[categorical+numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model.fit(X_train, y_train)
    return dv, model


# In[177]:


def predict(df_val, dv, model):
    dicts = df_val[categorical+numerical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    y_pred = model.predict_proba(X_val)[:, 1]

    return y_pred


# - `Logistic Regression`

# In[178]:


lr = LogisticRegression(C=0.001, solver='newton-cholesky', penalty='l2', max_iter=10)


# In[188]:


scores_cv = []
df_full = df_full_train.copy()
kfold = KFold(n_splits = 10, shuffle=True, random_state=1)
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full.iloc[train_idx]
    df_val = df_full.iloc[val_idx]

    y_train = df_train.heart_attack_risk.values
    y_val = df_val.heart_attack_risk.values

    # display(df_train.columns)
    # display(df_val.columns)
    
    del df_train['heart_attack_risk']
    del df_val['heart_attack_risk']
    
    dv, model = train(df_train, y_train, lr)
    
    y_pred = predict(df_train, dv, lr)
    auc_train = roc_auc_score(y_train, y_pred)

    y_pred = predict(df_val, dv, lr)
    auc_val = roc_auc_score(y_val, y_pred)
    
    scores_cv.append((auc_train, auc_val))

df_scores_cv = pd.DataFrame(scores_cv, columns = ['Train_AUC', 'Val_AUC'])
print(f"Train: {df_scores_cv.Train_AUC.mean()} +- {df_scores_cv.Train_AUC.std()}")
print(f"Val: {df_scores_cv.Val_AUC.mean()} +- {df_scores_cv.Val_AUC.std()}")


# - `Decision Trees`

# In[186]:


dt = DecisionTreeClassifier(criterion='gini',
                       max_depth=4,
                       min_samples_leaf=15,
                       random_state=12)


# In[189]:


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
print(f"Train: {df_scores_cv.Train_AUC.mean()} +- {df_scores_cv.Train_AUC.std()}")
print(f"Val: {df_scores_cv.Val_AUC.mean()} +- {df_scores_cv.Val_AUC.std()}")


# - `Random Forest`

# In[191]:


rf = RandomForestClassifier(n_estimators=9, 
                            max_depth=2,
                            min_samples_leaf=20,
                            max_features=35,
                            random_state=1)


# In[192]:


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
print(f"Train: {df_scores_cv.Train_AUC.mean()} +- {df_scores_cv.Train_AUC.std()}")
print(f"Val: {df_scores_cv.Val_AUC.mean()} +- {df_scores_cv.Val_AUC.std()}")


# - `XGBoost`

# In[193]:


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


# In[196]:


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
print(f"Train: {df_scores_cv.Train_AUC.mean()} +- {df_scores_cv.Train_AUC.std()}")
print(f"Val: {df_scores_cv.Val_AUC.mean()} +- {df_scores_cv.Val_AUC.std()}")


# ## Selecting the best Model

# From the cross validation sets we can see that of the four sets of models, `Logistic Regression` performs the best on validation sets with the least Standard Deviation, which implies better generalization.
# 
# Hence, we select `Logistic Regression` as the final model.
# 
# ```python
# LogisticRegression(C=0.001, solver='newton-cholesky', penalty='l2', max_iter=10)
# ```
