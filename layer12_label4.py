# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:00.517035Z","iopub.execute_input":"2023-09-24T12:44:00.517451Z","iopub.status.idle":"2023-09-24T12:44:00.857992Z","shell.execute_reply.started":"2023-09-24T12:44:00.517419Z","shell.execute_reply":"2023-09-24T12:44:00.856482Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:00.860891Z","iopub.execute_input":"2023-09-24T12:44:00.861450Z","iopub.status.idle":"2023-09-24T12:44:02.090041Z","shell.execute_reply.started":"2023-09-24T12:44:00.861412Z","shell.execute_reply":"2023-09-24T12:44:02.088758Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:02.091336Z","iopub.execute_input":"2023-09-24T12:44:02.091632Z","iopub.status.idle":"2023-09-24T12:44:10.025442Z","shell.execute_reply.started":"2023-09-24T12:44:02.091606Z","shell.execute_reply":"2023-09-24T12:44:10.024218Z"}}
train_df = pd.read_csv('/kaggle/input/layer12/train.csv')
valid_df = pd.read_csv('/kaggle/input/layer12/valid.csv')
test_df = pd.read_csv('/kaggle/input/layer12/test.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:10.026637Z","iopub.execute_input":"2023-09-24T12:44:10.026955Z","iopub.status.idle":"2023-09-24T12:44:10.034474Z","shell.execute_reply.started":"2023-09-24T12:44:10.026928Z","shell.execute_reply":"2023-09-24T12:44:10.033338Z"}}
train_df.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:10.037616Z","iopub.execute_input":"2023-09-24T12:44:10.037919Z","iopub.status.idle":"2023-09-24T12:44:10.076677Z","shell.execute_reply.started":"2023-09-24T12:44:10.037895Z","shell.execute_reply":"2023-09-24T12:44:10.075629Z"}}
train_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:10.077765Z","iopub.execute_input":"2023-09-24T12:44:10.078002Z","iopub.status.idle":"2023-09-24T12:44:10.100526Z","shell.execute_reply.started":"2023-09-24T12:44:10.077982Z","shell.execute_reply":"2023-09-24T12:44:10.099623Z"}}
missing_columns = train_df.columns[train_df.isnull().any()]
missing_counts = train_df[missing_columns].isnull().sum()

print('Missing Columns and Counts')
for column in missing_columns:
    print( str(column) +' : '+ str(missing_counts[column]))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:10.102914Z","iopub.execute_input":"2023-09-24T12:44:10.103277Z","iopub.status.idle":"2023-09-24T12:44:10.132904Z","shell.execute_reply.started":"2023-09-24T12:44:10.103250Z","shell.execute_reply":"2023-09-24T12:44:10.131568Z"}}
train_data = train_df.copy()
valid_data = valid_df.copy()
test_data = test_df.copy()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:10.134477Z","iopub.execute_input":"2023-09-24T12:44:10.135086Z","iopub.status.idle":"2023-09-24T12:44:12.032587Z","shell.execute_reply.started":"2023-09-24T12:44:10.135034Z","shell.execute_reply":"2023-09-24T12:44:12.031587Z"}}
train_df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:12.033914Z","iopub.execute_input":"2023-09-24T12:44:12.034227Z","iopub.status.idle":"2023-09-24T12:44:16.004871Z","shell.execute_reply.started":"2023-09-24T12:44:12.034200Z","shell.execute_reply":"2023-09-24T12:44:16.003452Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:16.007041Z","iopub.execute_input":"2023-09-24T12:44:16.007488Z","iopub.status.idle":"2023-09-24T12:44:16.039489Z","shell.execute_reply.started":"2023-09-24T12:44:16.007452Z","shell.execute_reply":"2023-09-24T12:44:16.038553Z"}}
x_train_df = x_train['label_4'].copy()
y_train_df = y_train['label_4'].copy()

x_valid_df = x_valid['label_4'].copy()
y_valid_df = y_valid['label_4'].copy()

x_test_df = x_test['label_4'].copy()

# %% [markdown]
# # Cross Validation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:16.042733Z","iopub.execute_input":"2023-09-24T12:44:16.043360Z","iopub.status.idle":"2023-09-24T12:56:54.468922Z","shell.execute_reply.started":"2023-09-24T12:44:16.043331Z","shell.execute_reply":"2023-09-24T12:56:54.467857Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:56:54.470555Z","iopub.execute_input":"2023-09-24T12:56:54.470919Z","iopub.status.idle":"2023-09-24T12:56:57.004211Z","shell.execute_reply.started":"2023-09-24T12:56:54.470888Z","shell.execute_reply":"2023-09-24T12:56:57.003339Z"}}
from sklearn.decomposition import PCA

pca = PCA(n_components=0.975, svd_solver='full')
pca.fit(x_train_df)
x_train_df_pca = pd.DataFrame(pca.transform(x_train_df))
x_valid_df_pca = pd.DataFrame(pca.transform(x_valid_df))
x_test_df_pca = pd.DataFrame(pca.transform(x_test_df))
print('Shape after PCA: ',x_train_df_pca.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:56:57.007421Z","iopub.execute_input":"2023-09-24T12:56:57.007820Z","iopub.status.idle":"2023-09-24T12:56:57.014050Z","shell.execute_reply.started":"2023-09-24T12:56:57.007794Z","shell.execute_reply":"2023-09-24T12:56:57.013374Z"}}
from sklearn import metrics

# %% [markdown]
# # SVM

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:56:57.019305Z","iopub.execute_input":"2023-09-24T12:56:57.021214Z","iopub.status.idle":"2023-09-24T12:59:07.689175Z","shell.execute_reply.started":"2023-09-24T12:56:57.021185Z","shell.execute_reply":"2023-09-24T12:59:07.688277Z"}}
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classifier = svm.SVC(kernel='linear', C=1)

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

# %% [code] {"_kg_hide-output":true}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:11:22.050138Z","iopub.execute_input":"2023-09-24T13:11:22.050480Z","iopub.status.idle":"2023-09-24T13:12:21.050322Z","shell.execute_reply.started":"2023-09-24T13:11:22.050454Z","shell.execute_reply":"2023-09-24T13:12:21.048882Z"}}
from sklearn import svm

classifier = svm.SVC(kernel='rbf', C=100, gamma='scale', degree=4, class_weight='balanced')

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_predict_after_pca = classifier.predict(x_test_df_pca)



# %% [markdown]
# # RandomForest

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:15:48.162516Z","iopub.execute_input":"2023-09-24T13:15:48.163420Z","iopub.status.idle":"2023-09-24T13:18:14.539372Z","shell.execute_reply.started":"2023-09-24T13:15:48.163390Z","shell.execute_reply":"2023-09-24T13:18:14.538504Z"}}
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=42)

classifier.fit(x_train_df, y_train_df)

y_valid_pred = classifier.predict(x_valid_df)

print("accuracy_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_pred = classifier.predict(x_test_df)

# %% [markdown]
# # CSV Creation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:29:42.481463Z","iopub.execute_input":"2023-09-24T13:29:42.481912Z","iopub.status.idle":"2023-09-24T13:29:42.488639Z","shell.execute_reply.started":"2023-09-24T13:29:42.481874Z","shell.execute_reply":"2023-09-24T13:29:42.487890Z"}}
output_df=pd.DataFrame(columns=["ID","label_1","label_2","label_3","label_4"])

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:29:46.010754Z","iopub.execute_input":"2023-09-24T13:29:46.011102Z","iopub.status.idle":"2023-09-24T13:29:46.018783Z","shell.execute_reply.started":"2023-09-24T13:29:46.011049Z","shell.execute_reply":"2023-09-24T13:29:46.017140Z"}}
IDs = list(i for i in range(1, len(test_df)+1))
output_df["ID"] = IDs

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:29:58.796189Z","iopub.execute_input":"2023-09-24T13:29:58.796528Z","iopub.status.idle":"2023-09-24T13:29:58.801723Z","shell.execute_reply.started":"2023-09-24T13:29:58.796503Z","shell.execute_reply":"2023-09-24T13:29:58.800277Z"}}
output_df["label_4"] = y_test_predict_after_pca

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:30:03.044847Z","iopub.execute_input":"2023-09-24T13:30:03.045206Z","iopub.status.idle":"2023-09-24T13:30:03.063429Z","shell.execute_reply.started":"2023-09-24T13:30:03.045182Z","shell.execute_reply":"2023-09-24T13:30:03.062317Z"}}
output_df

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:30:16.262925Z","iopub.execute_input":"2023-09-24T13:30:16.263267Z","iopub.status.idle":"2023-09-24T13:30:16.272744Z","shell.execute_reply.started":"2023-09-24T13:30:16.263243Z","shell.execute_reply":"2023-09-24T13:30:16.271786Z"}}
output_df.to_csv('/kaggle/working/output12_4.csv',index=False)