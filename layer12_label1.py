# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:39:05.774483Z","iopub.execute_input":"2023-09-24T09:39:05.775726Z","iopub.status.idle":"2023-09-24T09:39:06.237039Z","shell.execute_reply.started":"2023-09-24T09:39:05.775677Z","shell.execute_reply":"2023-09-24T09:39:06.235749Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:39:06.240416Z","iopub.execute_input":"2023-09-24T09:39:06.241570Z","iopub.status.idle":"2023-09-24T09:39:07.625821Z","shell.execute_reply.started":"2023-09-24T09:39:06.241524Z","shell.execute_reply":"2023-09-24T09:39:07.624669Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:39:07.627421Z","iopub.execute_input":"2023-09-24T09:39:07.628217Z","iopub.status.idle":"2023-09-24T09:39:18.677421Z","shell.execute_reply.started":"2023-09-24T09:39:07.628171Z","shell.execute_reply":"2023-09-24T09:39:18.676330Z"}}
train_df = pd.read_csv('/kaggle/input/layer12/train.csv')
valid_df = pd.read_csv('/kaggle/input/layer12/valid.csv')
test_df = pd.read_csv('/kaggle/input/layer12/test.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:39:18.678556Z","iopub.execute_input":"2023-09-24T09:39:18.678949Z","iopub.status.idle":"2023-09-24T09:39:18.686077Z","shell.execute_reply.started":"2023-09-24T09:39:18.678922Z","shell.execute_reply":"2023-09-24T09:39:18.685078Z"}}
train_df.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:39:18.688944Z","iopub.execute_input":"2023-09-24T09:39:18.689305Z","iopub.status.idle":"2023-09-24T09:39:18.729064Z","shell.execute_reply.started":"2023-09-24T09:39:18.689277Z","shell.execute_reply":"2023-09-24T09:39:18.728013Z"}}
train_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:39:18.730321Z","iopub.execute_input":"2023-09-24T09:39:18.730706Z","iopub.status.idle":"2023-09-24T09:39:18.763588Z","shell.execute_reply.started":"2023-09-24T09:39:18.730671Z","shell.execute_reply":"2023-09-24T09:39:18.762464Z"}}
missing_columns = train_df.columns[train_df.isnull().any()]
missing_counts = train_df[missing_columns].isnull().sum()

print('Missing Columns and Counts')
for column in missing_columns:
    print( str(column) +' : '+ str(missing_counts[column]))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:39:18.764727Z","iopub.execute_input":"2023-09-24T09:39:18.765012Z","iopub.status.idle":"2023-09-24T09:39:18.841711Z","shell.execute_reply.started":"2023-09-24T09:39:18.764987Z","shell.execute_reply":"2023-09-24T09:39:18.840643Z"}}
train_data = train_df.copy()
valid_data = valid_df.copy()
test_data = test_df.copy()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:39:18.842825Z","iopub.execute_input":"2023-09-24T09:39:18.843106Z","iopub.status.idle":"2023-09-24T09:39:21.413613Z","shell.execute_reply.started":"2023-09-24T09:39:18.843081Z","shell.execute_reply":"2023-09-24T09:39:21.412573Z"}}
train_df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:39:21.415074Z","iopub.execute_input":"2023-09-24T09:39:21.415566Z","iopub.status.idle":"2023-09-24T09:39:26.188637Z","shell.execute_reply.started":"2023-09-24T09:39:21.415538Z","shell.execute_reply":"2023-09-24T09:39:26.187797Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:39:26.189847Z","iopub.execute_input":"2023-09-24T09:39:26.190577Z","iopub.status.idle":"2023-09-24T09:39:26.267538Z","shell.execute_reply.started":"2023-09-24T09:39:26.190544Z","shell.execute_reply":"2023-09-24T09:39:26.266537Z"}}
x_train_df = x_train['label_1'].copy()
y_train_df = y_train['label_1'].copy()

x_valid_df = x_valid['label_1'].copy()
y_valid_df = y_valid['label_1'].copy()

x_test_df = x_test['label_1'].copy()

# %% [markdown]
# # Cross Validation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:39:26.268681Z","iopub.execute_input":"2023-09-24T09:39:26.268969Z","iopub.status.idle":"2023-09-24T09:58:36.084077Z","shell.execute_reply.started":"2023-09-24T09:39:26.268944Z","shell.execute_reply":"2023-09-24T09:58:36.082974Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:58:36.085503Z","iopub.execute_input":"2023-09-24T09:58:36.086452Z","iopub.status.idle":"2023-09-24T09:58:40.223853Z","shell.execute_reply.started":"2023-09-24T09:58:36.086411Z","shell.execute_reply":"2023-09-24T09:58:40.222460Z"}}
from sklearn.decomposition import PCA

pca = PCA(n_components=0.975, svd_solver='full')
pca.fit(x_train_df)
x_train_df_pca = pd.DataFrame(pca.transform(x_train_df))
x_valid_df_pca = pd.DataFrame(pca.transform(x_valid_df))
x_test_df_pca = pd.DataFrame(pca.transform(x_test_df))
print('Shape after PCA: ',x_train_df_pca.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:58:40.225842Z","iopub.execute_input":"2023-09-24T09:58:40.235045Z","iopub.status.idle":"2023-09-24T09:58:40.244635Z","shell.execute_reply.started":"2023-09-24T09:58:40.234985Z","shell.execute_reply":"2023-09-24T09:58:40.242861Z"}}
from sklearn import metrics

# %% [markdown]
# # SVM

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:58:40.251930Z","iopub.execute_input":"2023-09-24T09:58:40.252636Z","iopub.status.idle":"2023-09-24T09:59:11.308051Z","shell.execute_reply.started":"2023-09-24T09:58:40.252566Z","shell.execute_reply":"2023-09-24T09:59:11.307046Z"}}
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classifier = svm.SVC(kernel='linear', C=1)

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:59:11.309368Z","iopub.execute_input":"2023-09-24T09:59:11.309951Z","iopub.status.idle":"2023-09-24T10:26:11.555085Z","shell.execute_reply.started":"2023-09-24T09:59:11.309919Z","shell.execute_reply":"2023-09-24T10:26:11.553796Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:26:11.556849Z","iopub.execute_input":"2023-09-24T10:26:11.557563Z","iopub.status.idle":"2023-09-24T10:27:19.959459Z","shell.execute_reply.started":"2023-09-24T10:26:11.557519Z","shell.execute_reply":"2023-09-24T10:27:19.958291Z"}}
from sklearn import svm

classifier = svm.SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'], degree=best_params['degree'], class_weight=best_params['class_weight'])

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_predict_after_pca = classifier.predict(x_test_df_pca)



# %% [markdown]
# # RandomForest

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:27:19.960985Z","iopub.execute_input":"2023-09-24T10:27:19.961296Z","iopub.status.idle":"2023-09-24T10:30:58.363124Z","shell.execute_reply.started":"2023-09-24T10:27:19.961269Z","shell.execute_reply":"2023-09-24T10:30:58.361921Z"}}
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=42)

classifier.fit(x_train_df, y_train_df)

y_valid_pred = classifier.predict(x_valid_df)

print("accuracy_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_pred = classifier.predict(x_test_df)

# %% [markdown]
# # CSV Creation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:30:58.364649Z","iopub.execute_input":"2023-09-24T10:30:58.364971Z","iopub.status.idle":"2023-09-24T10:30:58.375407Z","shell.execute_reply.started":"2023-09-24T10:30:58.364942Z","shell.execute_reply":"2023-09-24T10:30:58.374301Z"}}
output_df=pd.DataFrame(columns=["ID","label_1","label_2","label_3","label_4"])

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:30:58.377599Z","iopub.execute_input":"2023-09-24T10:30:58.378911Z","iopub.status.idle":"2023-09-24T10:30:58.388248Z","shell.execute_reply.started":"2023-09-24T10:30:58.378878Z","shell.execute_reply":"2023-09-24T10:30:58.387360Z"}}
IDs = list(i for i in range(1, len(test_df)+1))
output_df["ID"] = IDs

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:30:58.389739Z","iopub.execute_input":"2023-09-24T10:30:58.390010Z","iopub.status.idle":"2023-09-24T10:30:58.396731Z","shell.execute_reply.started":"2023-09-24T10:30:58.389986Z","shell.execute_reply":"2023-09-24T10:30:58.395814Z"}}
output_df["label_1"] = y_test_predict_after_pca

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:30:58.397846Z","iopub.execute_input":"2023-09-24T10:30:58.398227Z","iopub.status.idle":"2023-09-24T10:30:58.418728Z","shell.execute_reply.started":"2023-09-24T10:30:58.398201Z","shell.execute_reply":"2023-09-24T10:30:58.417581Z"}}
output_df

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:30:58.420076Z","iopub.execute_input":"2023-09-24T10:30:58.420368Z","iopub.status.idle":"2023-09-24T10:30:58.433228Z","shell.execute_reply.started":"2023-09-24T10:30:58.420343Z","shell.execute_reply":"2023-09-24T10:30:58.432257Z"}}
output_df.to_csv('/kaggle/working/output12_1.csv',index=False)