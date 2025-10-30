import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline

import pickle


# Parameters
C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'



# Data Preparation

print("Preparing the data.....")

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.churn = (df.churn == 'yes').astype(int)

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']

# Training model

print("Training the model.....")

pipeline = make_pipeline(DictVectorizer(sparse=False), LogisticRegression(solver='liblinear', C=C))

def train(df, y, C=1.0):

    cat = df[categorical + numerical].to_dict(orient='records')
    
    # dv = DictVectorizer(sparse=False)
    # dv.fit(cat)

    # X = dv.transform(cat)

    # model = LogisticRegression(solver='liblinear', C=C)
    # model.fit(X, y)
    pipeline.fit(cat, y)

    return pipeline


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(dicts)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

print("Validating the data.....")


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

aucs = []

fold = 0

for train_idx, val_idx in kfold.split(df_train_full):
    df_train = df_train_full.iloc[train_idx]
    y_train = df_train.churn.values

    df_val = df_train_full.iloc[val_idx]
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    rocauc = roc_auc_score(y_val, y_pred)
    aucs.append(rocauc)

    print(f'auc on fold {fold} is {rocauc}')
    fold = fold + 1

print('validation results')
print('C=%s, auc =  %0.3f ± %0.3f' % (C, np.mean(aucs), np.std(aucs)))

print("Training the final model.....")

y_train = df_train_full.churn.values
dv, model = train(df_train_full, y_train, C=1)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
print(f'auc={auc}')

# To save the model

with open(output_file, 'wb') as f_out:
    # pickle.dump((dv, model), f_out)
    pickle.dump(pipeline, f_out)

print("The model is saved to {output_file}")