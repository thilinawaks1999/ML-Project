# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:35:37.661993Z","iopub.execute_input":"2023-09-24T06:35:37.662493Z","iopub.status.idle":"2023-09-24T06:35:38.120649Z","shell.execute_reply.started":"2023-09-24T06:35:37.662442Z","shell.execute_reply":"2023-09-24T06:35:38.119734Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:35:39.751970Z","iopub.execute_input":"2023-09-24T06:35:39.752655Z","iopub.status.idle":"2023-09-24T06:35:40.509164Z","shell.execute_reply.started":"2023-09-24T06:35:39.752614Z","shell.execute_reply":"2023-09-24T06:35:40.507996Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:35:42.418785Z","iopub.execute_input":"2023-09-24T06:35:42.420315Z","iopub.status.idle":"2023-09-24T06:35:54.322790Z","shell.execute_reply.started":"2023-09-24T06:35:42.420211Z","shell.execute_reply":"2023-09-24T06:35:54.321508Z"}}
train_df = pd.read_csv('/kaggle/input/layer7/train.csv')
valid_df = pd.read_csv('/kaggle/input/layer7/valid.csv')
test_df = pd.read_csv('/kaggle/input/layer7/test.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:35:54.325210Z","iopub.execute_input":"2023-09-24T06:35:54.325598Z","iopub.status.idle":"2023-09-24T06:35:54.333860Z","shell.execute_reply.started":"2023-09-24T06:35:54.325565Z","shell.execute_reply":"2023-09-24T06:35:54.332708Z"}}
train_df.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:35:54.335744Z","iopub.execute_input":"2023-09-24T06:35:54.336104Z","iopub.status.idle":"2023-09-24T06:35:54.388703Z","shell.execute_reply.started":"2023-09-24T06:35:54.336073Z","shell.execute_reply":"2023-09-24T06:35:54.387774Z"}}
train_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:35:56.932938Z","iopub.execute_input":"2023-09-24T06:35:56.933361Z","iopub.status.idle":"2023-09-24T06:35:56.964968Z","shell.execute_reply.started":"2023-09-24T06:35:56.933328Z","shell.execute_reply":"2023-09-24T06:35:56.963908Z"}}
missing_columns = train_df.columns[train_df.isnull().any()]
missing_counts = train_df[missing_columns].isnull().sum()

print('Missing Columns and Counts')
for column in missing_columns:
    print( str(column) +' : '+ str(missing_counts[column]))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:35:59.198908Z","iopub.execute_input":"2023-09-24T06:35:59.200052Z","iopub.status.idle":"2023-09-24T06:35:59.279683Z","shell.execute_reply.started":"2023-09-24T06:35:59.199999Z","shell.execute_reply":"2023-09-24T06:35:59.278549Z"}}
train_data = train_df.copy()
valid_data = valid_df.copy()
test_data = test_df.copy()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:36:01.293532Z","iopub.execute_input":"2023-09-24T06:36:01.294003Z","iopub.status.idle":"2023-09-24T06:36:04.124401Z","shell.execute_reply.started":"2023-09-24T06:36:01.293970Z","shell.execute_reply":"2023-09-24T06:36:04.123174Z"}}
train_df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:36:06.499279Z","iopub.execute_input":"2023-09-24T06:36:06.499739Z","iopub.status.idle":"2023-09-24T06:36:11.429776Z","shell.execute_reply.started":"2023-09-24T06:36:06.499703Z","shell.execute_reply":"2023-09-24T06:36:11.428432Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:36:15.701423Z","iopub.execute_input":"2023-09-24T06:36:15.701873Z","iopub.status.idle":"2023-09-24T06:36:15.833932Z","shell.execute_reply.started":"2023-09-24T06:36:15.701838Z","shell.execute_reply":"2023-09-24T06:36:15.832648Z"}}
x_train_df = x_train['label_1'].copy()
y_train_df = y_train['label_1'].copy()

x_valid_df = x_valid['label_1'].copy()
y_valid_df = y_valid['label_1'].copy()

x_test_df = x_test['label_1'].copy()

# %% [markdown]
# # Cross Validation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-23T18:10:32.538905Z","iopub.execute_input":"2023-09-23T18:10:32.539979Z","iopub.status.idle":"2023-09-23T18:29:38.349846Z","shell.execute_reply.started":"2023-09-23T18:10:32.539949Z","shell.execute_reply":"2023-09-23T18:29:38.348873Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:36:34.712321Z","iopub.execute_input":"2023-09-24T06:36:34.716473Z","iopub.status.idle":"2023-09-24T06:36:39.147321Z","shell.execute_reply.started":"2023-09-24T06:36:34.716407Z","shell.execute_reply":"2023-09-24T06:36:39.145803Z"}}
from sklearn.decomposition import PCA

pca = PCA(n_components=0.975, svd_solver='full')
pca.fit(x_train_df)
x_train_df_pca = pd.DataFrame(pca.transform(x_train_df))
x_valid_df_pca = pd.DataFrame(pca.transform(x_valid_df))
x_test_df_pca = pd.DataFrame(pca.transform(x_test_df))
print('Shape after PCA: ',x_train_df_pca.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:38:56.469149Z","iopub.execute_input":"2023-09-24T06:38:56.469969Z","iopub.status.idle":"2023-09-24T06:38:56.476047Z","shell.execute_reply.started":"2023-09-24T06:38:56.469927Z","shell.execute_reply":"2023-09-24T06:38:56.474547Z"}}
from sklearn import metrics

# %% [markdown]
# # SVM

# %% [code] {"execution":{"iopub.status.busy":"2023-09-23T18:29:41.931161Z","iopub.execute_input":"2023-09-23T18:29:41.931881Z","iopub.status.idle":"2023-09-23T18:30:17.138289Z","shell.execute_reply.started":"2023-09-23T18:29:41.931841Z","shell.execute_reply":"2023-09-23T18:30:17.136934Z"}}
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classifier = svm.SVC(kernel='linear', C=1)

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-23T18:30:17.139851Z","iopub.execute_input":"2023-09-23T18:30:17.140634Z","iopub.status.idle":"2023-09-23T19:08:21.619789Z","shell.execute_reply.started":"2023-09-23T18:30:17.140593Z","shell.execute_reply":"2023-09-23T19:08:21.618045Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:39:06.506854Z","iopub.execute_input":"2023-09-24T06:39:06.507363Z","iopub.status.idle":"2023-09-24T06:41:12.117914Z","shell.execute_reply.started":"2023-09-24T06:39:06.507326Z","shell.execute_reply":"2023-09-24T06:41:12.116751Z"}}
from sklearn import svm

classifier = svm.SVC(kernel='rbf', C=100, gamma='scale', degree=4, class_weight='balanced')

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_predict_after_pca = classifier.predict(x_test_df_pca)



# %% [markdown]
# # RandomForest

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T06:45:48.056926Z","iopub.execute_input":"2023-09-24T06:45:48.057417Z","iopub.status.idle":"2023-09-24T06:49:28.495043Z","shell.execute_reply.started":"2023-09-24T06:45:48.057384Z","shell.execute_reply":"2023-09-24T06:49:28.493587Z"}}
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=42)

classifier.fit(x_train_df, y_train_df)

y_valid_pred = classifier.predict(x_valid_df)

print("accuracy_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_pred = classifier.predict(x_test_df)

# %% [markdown]
# # CSV Creation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:12:18.368585Z","iopub.execute_input":"2023-09-24T07:12:18.368974Z","iopub.status.idle":"2023-09-24T07:12:18.376403Z","shell.execute_reply.started":"2023-09-24T07:12:18.368944Z","shell.execute_reply":"2023-09-24T07:12:18.375292Z"}}
output_df=pd.DataFrame(columns=["ID","label_1","label_2","label_3","label_4"])

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:16:00.121541Z","iopub.execute_input":"2023-09-24T07:16:00.121935Z","iopub.status.idle":"2023-09-24T07:16:00.130932Z","shell.execute_reply.started":"2023-09-24T07:16:00.121905Z","shell.execute_reply":"2023-09-24T07:16:00.129465Z"}}
IDs = list(i for i in range(1, len(test_df)+1))
output_df["ID"] = IDs

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:17:13.321995Z","iopub.execute_input":"2023-09-24T07:17:13.322417Z","iopub.status.idle":"2023-09-24T07:17:13.327745Z","shell.execute_reply.started":"2023-09-24T07:17:13.322385Z","shell.execute_reply":"2023-09-24T07:17:13.326694Z"}}
output_df["label_1"] = y_test_predict_after_pca

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:17:19.285283Z","iopub.execute_input":"2023-09-24T07:17:19.285689Z","iopub.status.idle":"2023-09-24T07:17:19.301756Z","shell.execute_reply.started":"2023-09-24T07:17:19.285657Z","shell.execute_reply":"2023-09-24T07:17:19.300579Z"}}
output_df

# %% [code]
output_df.to_csv('/kaggle/working/output7_1.csv',index=False)