# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:59:49.093632Z","iopub.execute_input":"2023-09-24T08:59:49.094302Z","iopub.status.idle":"2023-09-24T08:59:49.548563Z","shell.execute_reply.started":"2023-09-24T08:59:49.094258Z","shell.execute_reply":"2023-09-24T08:59:49.545787Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:59:49.550599Z","iopub.execute_input":"2023-09-24T08:59:49.551112Z","iopub.status.idle":"2023-09-24T08:59:52.113863Z","shell.execute_reply.started":"2023-09-24T08:59:49.551078Z","shell.execute_reply":"2023-09-24T08:59:52.112510Z"}}
import pandas as pd
import numpy as np
from pandas import Series

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:59:52.115380Z","iopub.execute_input":"2023-09-24T08:59:52.116057Z","iopub.status.idle":"2023-09-24T09:00:04.931887Z","shell.execute_reply.started":"2023-09-24T08:59:52.116016Z","shell.execute_reply":"2023-09-24T09:00:04.930575Z"}}
train_df = pd.read_csv('/kaggle/input/layer7/train.csv')
valid_df = pd.read_csv('/kaggle/input/layer7/valid.csv')
test_df = pd.read_csv('/kaggle/input/layer7/test.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:04.934329Z","iopub.execute_input":"2023-09-24T09:00:04.934680Z","iopub.status.idle":"2023-09-24T09:00:04.944332Z","shell.execute_reply.started":"2023-09-24T09:00:04.934651Z","shell.execute_reply":"2023-09-24T09:00:04.941768Z"}}
train_df.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:04.945642Z","iopub.execute_input":"2023-09-24T09:00:04.946494Z","iopub.status.idle":"2023-09-24T09:00:05.005021Z","shell.execute_reply.started":"2023-09-24T09:00:04.946452Z","shell.execute_reply":"2023-09-24T09:00:05.003911Z"}}
train_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:05.006347Z","iopub.execute_input":"2023-09-24T09:00:05.007215Z","iopub.status.idle":"2023-09-24T09:00:05.041195Z","shell.execute_reply.started":"2023-09-24T09:00:05.007184Z","shell.execute_reply":"2023-09-24T09:00:05.040035Z"}}
missing_columns = train_df.columns[train_df.isnull().any()]
missing_counts = train_df[missing_columns].isnull().sum()

print('Missing Columns and Counts')
for column in missing_columns:
    print( str(column) +' : '+ str(missing_counts[column]))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:05.042904Z","iopub.execute_input":"2023-09-24T09:00:05.043245Z","iopub.status.idle":"2023-09-24T09:00:05.121863Z","shell.execute_reply.started":"2023-09-24T09:00:05.043216Z","shell.execute_reply":"2023-09-24T09:00:05.120637Z"}}
train_data = train_df.copy()
valid_data = valid_df.copy()
test_data = test_df.copy()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:05.123211Z","iopub.execute_input":"2023-09-24T09:00:05.123567Z","iopub.status.idle":"2023-09-24T09:00:07.882767Z","shell.execute_reply.started":"2023-09-24T09:00:05.123530Z","shell.execute_reply":"2023-09-24T09:00:07.881649Z"}}
train_df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:07.884069Z","iopub.execute_input":"2023-09-24T09:00:07.884437Z","iopub.status.idle":"2023-09-24T09:00:12.895126Z","shell.execute_reply.started":"2023-09-24T09:00:07.884391Z","shell.execute_reply":"2023-09-24T09:00:12.893795Z"}}
from sklearn.preprocessing import RobustScaler # eliminate outliers

x_train = {}
x_valid = {}
x_test = {}

y_train = {}
y_valid = {}
y_test = {}

#create dictionaries for each label
for target_label in ['label_1','label_2','label_3','label_4']:

  if target_label == "label_2":
    train = train_df[train_df['label_2'].notna()]
    valid = valid_df[valid_df['label_2'].notna()]
  else:
    train = train_df
    valid = valid_df

  test = test_df

  scaler = RobustScaler()

  x_train[target_label] = pd.DataFrame(scaler.fit_transform(train.drop(['label_1','label_2','label_3','label_4'], axis=1)), columns=[f'feature_{i}' for i in range(1,769)])
  y_train[target_label] = train[target_label]

  x_valid[target_label] = pd.DataFrame(scaler.transform(valid.drop(['label_1','label_2','label_3','label_4'], axis=1)), columns=[f'feature_{i}' for i in range(1,769)])
  y_valid  [target_label] = valid[target_label]

  x_test[target_label] = pd.DataFrame(scaler.transform(test.drop(["ID"],axis=1)), columns=[f'feature_{i}' for i in range(1,769)])

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:12.899354Z","iopub.execute_input":"2023-09-24T09:00:12.899728Z","iopub.status.idle":"2023-09-24T09:00:12.979012Z","shell.execute_reply.started":"2023-09-24T09:00:12.899687Z","shell.execute_reply":"2023-09-24T09:00:12.977958Z"}}
x_train_df = x_train['label_3'].copy()
y_train_df = y_train['label_3'].copy()

x_valid_df = x_valid['label_3'].copy()
y_valid_df = y_valid['label_3'].copy()

x_test_df = x_test['label_3'].copy()

# %% [markdown]
# # Cross Validation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:12.980624Z","iopub.execute_input":"2023-09-24T09:00:12.980986Z","iopub.status.idle":"2023-09-24T09:05:06.728675Z","shell.execute_reply.started":"2023-09-24T09:00:12.980958Z","shell.execute_reply":"2023-09-24T09:05:06.726967Z"}}
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold

# Perform cross-validation
scores = cross_val_score(SVC(), x_train_df, y_train_df, cv=5, scoring='accuracy')

mean_accuracy = scores.mean()
std_accuracy = scores.std()
# Print the cross-validation scores
print('Support Vector Machines')
print('\n')
print("Cross-validation scores:", scores)
print(f"Mean Accuracy: {mean_accuracy:.2f}")
print(f"Standard Deviation: {std_accuracy:.2f}")

# %% [markdown]
# # Feature Engineering

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:05:06.730488Z","iopub.execute_input":"2023-09-24T09:05:06.730903Z","iopub.status.idle":"2023-09-24T09:05:10.763175Z","shell.execute_reply.started":"2023-09-24T09:05:06.730873Z","shell.execute_reply":"2023-09-24T09:05:10.762042Z"}}
from sklearn.decomposition import PCA

pca = PCA(n_components=0.975, svd_solver='full')
pca.fit(x_train_df)
x_train_df_pca = pd.DataFrame(pca.transform(x_train_df))
x_valid_df_pca = pd.DataFrame(pca.transform(x_valid_df))
x_test_df_pca = pd.DataFrame(pca.transform(x_test_df))
print('Shape after PCA: ',x_train_df_pca.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:05:10.764914Z","iopub.execute_input":"2023-09-24T09:05:10.765707Z","iopub.status.idle":"2023-09-24T09:05:10.772506Z","shell.execute_reply.started":"2023-09-24T09:05:10.765665Z","shell.execute_reply":"2023-09-24T09:05:10.771012Z"}}
from sklearn import metrics

# %% [markdown]
# # SVM

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:05:10.774443Z","iopub.execute_input":"2023-09-24T09:05:10.775848Z","iopub.status.idle":"2023-09-24T09:05:19.428998Z","shell.execute_reply.started":"2023-09-24T09:05:10.775807Z","shell.execute_reply":"2023-09-24T09:05:19.427811Z"}}
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classifier = svm.SVC(kernel='linear', C=1)

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:05:19.430966Z","iopub.execute_input":"2023-09-24T09:05:19.431412Z","iopub.status.idle":"2023-09-24T09:33:33.751422Z","shell.execute_reply.started":"2023-09-24T09:05:19.431371Z","shell.execute_reply":"2023-09-24T09:33:33.749819Z"}}
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import numpy as np

param_dist = {
    'C': [100,10,1,0,0.1,0.01],
    'kernel': ['rbf','linear','poly','sigmoid'],
    'gamma': ['scale','auto'],
    'degree': [1,2,3,4],
    'class_weight' : ['none','balanced']
}

svm = SVC()

random_search = RandomizedSearchCV(
    svm, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, random_state=42, scoring='accuracy'
)

random_search.fit(x_train_df_pca, y_train_df)

best_params = random_search.best_params_
best_model = random_search.best_estimator_

print("best parameters:", best_params)

# %% [markdown]
# Label 1 
# best parameters: {'kernel': 'rbf', 'gamma': 'scale', 'degree': 4, 'class_weight': 'balanced', 'C': 100}
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:43:20.024998Z","iopub.execute_input":"2023-09-24T09:43:20.025406Z","iopub.status.idle":"2023-09-24T09:44:17.908578Z","shell.execute_reply.started":"2023-09-24T09:43:20.025377Z","shell.execute_reply":"2023-09-24T09:44:17.907053Z"}}
from sklearn import svm

classifier = svm.SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'], degree=best_params['degree'], class_weight=best_params['class_weight'])

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_predict_after_pca = classifier.predict(x_test_df_pca)



# %% [markdown]
# # RandomForest

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:34:31.550903Z","iopub.execute_input":"2023-09-24T09:34:31.551304Z","iopub.status.idle":"2023-09-24T09:36:35.751928Z","shell.execute_reply.started":"2023-09-24T09:34:31.551272Z","shell.execute_reply":"2023-09-24T09:36:35.750442Z"}}
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=42)

classifier.fit(x_train_df, y_train_df)

y_valid_pred = classifier.predict(x_valid_df)

print("accuracy_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_pred = classifier.predict(x_test_df)

# %% [markdown]
# # CSV Creation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:36:35.753262Z","iopub.execute_input":"2023-09-24T09:36:35.753600Z","iopub.status.idle":"2023-09-24T09:36:35.762686Z","shell.execute_reply.started":"2023-09-24T09:36:35.753572Z","shell.execute_reply":"2023-09-24T09:36:35.761245Z"}}
output_df=pd.DataFrame(columns=["ID","label_1","label_2","label_3","label_4"])

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:36:35.764297Z","iopub.execute_input":"2023-09-24T09:36:35.764793Z","iopub.status.idle":"2023-09-24T09:36:35.782304Z","shell.execute_reply.started":"2023-09-24T09:36:35.764749Z","shell.execute_reply":"2023-09-24T09:36:35.781182Z"}}
IDs = list(i for i in range(1, len(test_df)+1))
output_df["ID"] = IDs

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:36:35.784072Z","iopub.execute_input":"2023-09-24T09:36:35.784567Z","iopub.status.idle":"2023-09-24T09:36:35.795345Z","shell.execute_reply.started":"2023-09-24T09:36:35.784526Z","shell.execute_reply":"2023-09-24T09:36:35.794400Z"}}
output_df["label_3"] = y_test_predict_after_pca

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:36:35.796487Z","iopub.execute_input":"2023-09-24T09:36:35.797440Z","iopub.status.idle":"2023-09-24T09:36:35.822538Z","shell.execute_reply.started":"2023-09-24T09:36:35.797409Z","shell.execute_reply":"2023-09-24T09:36:35.821416Z"}}
output_df

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:36:35.824507Z","iopub.execute_input":"2023-09-24T09:36:35.824965Z","iopub.status.idle":"2023-09-24T09:36:35.841703Z","shell.execute_reply.started":"2023-09-24T09:36:35.824914Z","shell.execute_reply":"2023-09-24T09:36:35.840821Z"}}
output_df.to_csv('/kaggle/working/output7_3.csv',index=False)