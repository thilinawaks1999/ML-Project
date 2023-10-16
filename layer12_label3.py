# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:47:12.744873Z","iopub.execute_input":"2023-09-24T09:47:12.745211Z","iopub.status.idle":"2023-09-24T09:47:13.087676Z","shell.execute_reply.started":"2023-09-24T09:47:12.745185Z","shell.execute_reply":"2023-09-24T09:47:13.086036Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:47:13.090712Z","iopub.execute_input":"2023-09-24T09:47:13.093326Z","iopub.status.idle":"2023-09-24T09:47:15.520099Z","shell.execute_reply.started":"2023-09-24T09:47:13.093282Z","shell.execute_reply":"2023-09-24T09:47:15.518097Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:47:15.521643Z","iopub.execute_input":"2023-09-24T09:47:15.522062Z","iopub.status.idle":"2023-09-24T09:47:23.608638Z","shell.execute_reply.started":"2023-09-24T09:47:15.522027Z","shell.execute_reply":"2023-09-24T09:47:23.607914Z"}}
train_df = pd.read_csv('/kaggle/input/layer12/train.csv')
valid_df = pd.read_csv('/kaggle/input/layer12/valid.csv')
test_df = pd.read_csv('/kaggle/input/layer12/test.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:47:23.611090Z","iopub.execute_input":"2023-09-24T09:47:23.611455Z","iopub.status.idle":"2023-09-24T09:47:23.621021Z","shell.execute_reply.started":"2023-09-24T09:47:23.611427Z","shell.execute_reply":"2023-09-24T09:47:23.619619Z"}}
train_df.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:47:23.622352Z","iopub.execute_input":"2023-09-24T09:47:23.622666Z","iopub.status.idle":"2023-09-24T09:47:23.663966Z","shell.execute_reply.started":"2023-09-24T09:47:23.622642Z","shell.execute_reply":"2023-09-24T09:47:23.663258Z"}}
train_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:47:23.665036Z","iopub.execute_input":"2023-09-24T09:47:23.665413Z","iopub.status.idle":"2023-09-24T09:47:23.691899Z","shell.execute_reply.started":"2023-09-24T09:47:23.665390Z","shell.execute_reply":"2023-09-24T09:47:23.690411Z"}}
missing_columns = train_df.columns[train_df.isnull().any()]
missing_counts = train_df[missing_columns].isnull().sum()

print('Missing Columns and Counts')
for column in missing_columns:
    print( str(column) +' : '+ str(missing_counts[column]))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:47:23.693031Z","iopub.execute_input":"2023-09-24T09:47:23.693521Z","iopub.status.idle":"2023-09-24T09:47:23.722540Z","shell.execute_reply.started":"2023-09-24T09:47:23.693496Z","shell.execute_reply":"2023-09-24T09:47:23.721252Z"}}
train_data = train_df.copy()
valid_data = valid_df.copy()
test_data = test_df.copy()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:47:23.724306Z","iopub.execute_input":"2023-09-24T09:47:23.724737Z","iopub.status.idle":"2023-09-24T09:47:25.700904Z","shell.execute_reply.started":"2023-09-24T09:47:23.724710Z","shell.execute_reply":"2023-09-24T09:47:25.699794Z"}}
train_df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:47:25.701754Z","iopub.execute_input":"2023-09-24T09:47:25.702025Z","iopub.status.idle":"2023-09-24T09:47:29.665153Z","shell.execute_reply.started":"2023-09-24T09:47:25.702003Z","shell.execute_reply":"2023-09-24T09:47:29.663194Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:47:29.668523Z","iopub.execute_input":"2023-09-24T09:47:29.668896Z","iopub.status.idle":"2023-09-24T09:47:29.698519Z","shell.execute_reply.started":"2023-09-24T09:47:29.668868Z","shell.execute_reply":"2023-09-24T09:47:29.696959Z"}}
x_train_df = x_train['label_3'].copy()
y_train_df = y_train['label_3'].copy()

x_valid_df = x_valid['label_3'].copy()
y_valid_df = y_valid['label_3'].copy()

x_test_df = x_test['label_3'].copy()

# %% [markdown]
# # Cross Validation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:47:29.699833Z","iopub.execute_input":"2023-09-24T09:47:29.700208Z","iopub.status.idle":"2023-09-24T09:50:27.386015Z","shell.execute_reply.started":"2023-09-24T09:47:29.700179Z","shell.execute_reply":"2023-09-24T09:50:27.384911Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:50:27.387168Z","iopub.execute_input":"2023-09-24T09:50:27.389286Z","iopub.status.idle":"2023-09-24T09:50:30.570259Z","shell.execute_reply.started":"2023-09-24T09:50:27.389232Z","shell.execute_reply":"2023-09-24T09:50:30.568289Z"}}
from sklearn.decomposition import PCA

pca = PCA(n_components=0.975, svd_solver='full')
pca.fit(x_train_df)
x_train_df_pca = pd.DataFrame(pca.transform(x_train_df))
x_valid_df_pca = pd.DataFrame(pca.transform(x_valid_df))
x_test_df_pca = pd.DataFrame(pca.transform(x_test_df))
print('Shape after PCA: ',x_train_df_pca.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:50:30.571367Z","iopub.execute_input":"2023-09-24T09:50:30.571639Z","iopub.status.idle":"2023-09-24T09:50:30.575862Z","shell.execute_reply.started":"2023-09-24T09:50:30.571614Z","shell.execute_reply":"2023-09-24T09:50:30.575129Z"}}
from sklearn import metrics

# %% [markdown]
# # SVM

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:50:30.577032Z","iopub.execute_input":"2023-09-24T09:50:30.577487Z","iopub.status.idle":"2023-09-24T09:50:39.394392Z","shell.execute_reply.started":"2023-09-24T09:50:30.577461Z","shell.execute_reply":"2023-09-24T09:50:39.393108Z"}}
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classifier = svm.SVC(kernel='linear', C=1)

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:50:39.396051Z","iopub.execute_input":"2023-09-24T09:50:39.396424Z","iopub.status.idle":"2023-09-24T10:07:25.245007Z","shell.execute_reply.started":"2023-09-24T09:50:39.396398Z","shell.execute_reply":"2023-09-24T10:07:25.242790Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:07:25.247862Z","iopub.execute_input":"2023-09-24T10:07:25.248229Z","iopub.status.idle":"2023-09-24T10:07:40.853000Z","shell.execute_reply.started":"2023-09-24T10:07:25.248201Z","shell.execute_reply":"2023-09-24T10:07:40.851598Z"}}
from sklearn import svm

classifier = svm.SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'], degree=best_params['degree'], class_weight=best_params['class_weight'])

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_predict_after_pca = classifier.predict(x_test_df_pca)



# %% [markdown]
# # RandomForest

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:07:40.854495Z","iopub.execute_input":"2023-09-24T10:07:40.854816Z","iopub.status.idle":"2023-09-24T10:09:15.387295Z","shell.execute_reply.started":"2023-09-24T10:07:40.854789Z","shell.execute_reply":"2023-09-24T10:09:15.386209Z"}}
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=42)

classifier.fit(x_train_df, y_train_df)

y_valid_pred = classifier.predict(x_valid_df)

print("accuracy_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_pred = classifier.predict(x_test_df)

# %% [markdown]
# # CSV Creation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:09:15.388474Z","iopub.execute_input":"2023-09-24T10:09:15.388791Z","iopub.status.idle":"2023-09-24T10:09:15.396769Z","shell.execute_reply.started":"2023-09-24T10:09:15.388766Z","shell.execute_reply":"2023-09-24T10:09:15.395692Z"}}
output_df=pd.DataFrame(columns=["ID","label_1","label_2","label_3","label_4"])

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:09:15.398475Z","iopub.execute_input":"2023-09-24T10:09:15.398805Z","iopub.status.idle":"2023-09-24T10:09:15.416751Z","shell.execute_reply.started":"2023-09-24T10:09:15.398775Z","shell.execute_reply":"2023-09-24T10:09:15.415229Z"}}
IDs = list(i for i in range(1, len(test_df)+1))
output_df["ID"] = IDs

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:09:15.418479Z","iopub.execute_input":"2023-09-24T10:09:15.418764Z","iopub.status.idle":"2023-09-24T10:09:15.432027Z","shell.execute_reply.started":"2023-09-24T10:09:15.418744Z","shell.execute_reply":"2023-09-24T10:09:15.430116Z"}}
output_df["label_1"] = y_test_predict_after_pca

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:09:15.433876Z","iopub.execute_input":"2023-09-24T10:09:15.434259Z","iopub.status.idle":"2023-09-24T10:09:15.456704Z","shell.execute_reply.started":"2023-09-24T10:09:15.434229Z","shell.execute_reply":"2023-09-24T10:09:15.455472Z"}}
output_df

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:09:15.457638Z","iopub.execute_input":"2023-09-24T10:09:15.457990Z","iopub.status.idle":"2023-09-24T10:09:15.477584Z","shell.execute_reply.started":"2023-09-24T10:09:15.457961Z","shell.execute_reply":"2023-09-24T10:09:15.476275Z"}}
output_df.to_csv('/kaggle/working/output12_3.csv',index=False)