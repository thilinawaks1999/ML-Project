# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:53:09.250436Z","iopub.execute_input":"2023-09-24T07:53:09.250884Z","iopub.status.idle":"2023-09-24T07:53:09.260344Z","shell.execute_reply.started":"2023-09-24T07:53:09.250850Z","shell.execute_reply":"2023-09-24T07:53:09.258964Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:53:09.262805Z","iopub.execute_input":"2023-09-24T07:53:09.263279Z","iopub.status.idle":"2023-09-24T07:53:09.273970Z","shell.execute_reply.started":"2023-09-24T07:53:09.263237Z","shell.execute_reply":"2023-09-24T07:53:09.272660Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:53:09.435334Z","iopub.execute_input":"2023-09-24T07:53:09.435779Z","iopub.status.idle":"2023-09-24T07:53:17.129623Z","shell.execute_reply.started":"2023-09-24T07:53:09.435713Z","shell.execute_reply":"2023-09-24T07:53:17.128414Z"}}
train_df = pd.read_csv('/kaggle/input/layer7/train.csv')
valid_df = pd.read_csv('/kaggle/input/layer7/valid.csv')
test_df = pd.read_csv('/kaggle/input/layer7/test.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:53:17.132067Z","iopub.execute_input":"2023-09-24T07:53:17.132420Z","iopub.status.idle":"2023-09-24T07:53:17.139803Z","shell.execute_reply.started":"2023-09-24T07:53:17.132391Z","shell.execute_reply":"2023-09-24T07:53:17.138441Z"}}
train_df.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:53:17.141630Z","iopub.execute_input":"2023-09-24T07:53:17.142179Z","iopub.status.idle":"2023-09-24T07:53:17.194349Z","shell.execute_reply.started":"2023-09-24T07:53:17.142132Z","shell.execute_reply":"2023-09-24T07:53:17.192925Z"}}
train_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:53:17.196170Z","iopub.execute_input":"2023-09-24T07:53:17.196623Z","iopub.status.idle":"2023-09-24T07:53:17.233958Z","shell.execute_reply.started":"2023-09-24T07:53:17.196589Z","shell.execute_reply":"2023-09-24T07:53:17.232669Z"}}
missing_columns = train_df.columns[train_df.isnull().any()]
missing_counts = train_df[missing_columns].isnull().sum()

print('Missing Columns and Counts')
for column in missing_columns:
    print( str(column) +' : '+ str(missing_counts[column]))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:53:17.237599Z","iopub.execute_input":"2023-09-24T07:53:17.238074Z","iopub.status.idle":"2023-09-24T07:53:17.321250Z","shell.execute_reply.started":"2023-09-24T07:53:17.238037Z","shell.execute_reply":"2023-09-24T07:53:17.319737Z"}}
train_data = train_df.copy()
valid_data = valid_df.copy()
test_data = test_df.copy()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:53:17.323480Z","iopub.execute_input":"2023-09-24T07:53:17.325521Z","iopub.status.idle":"2023-09-24T07:53:20.346390Z","shell.execute_reply.started":"2023-09-24T07:53:17.325477Z","shell.execute_reply":"2023-09-24T07:53:20.345142Z"}}
train_df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:53:20.348312Z","iopub.execute_input":"2023-09-24T07:53:20.349422Z","iopub.status.idle":"2023-09-24T07:53:25.406932Z","shell.execute_reply.started":"2023-09-24T07:53:20.349376Z","shell.execute_reply":"2023-09-24T07:53:25.405524Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:53:25.409161Z","iopub.execute_input":"2023-09-24T07:53:25.409617Z","iopub.status.idle":"2023-09-24T07:53:25.491969Z","shell.execute_reply.started":"2023-09-24T07:53:25.409575Z","shell.execute_reply":"2023-09-24T07:53:25.490378Z"}}
x_train_df = x_train['label_2'].copy()
y_train_df = y_train['label_2'].copy()

x_valid_df = x_valid['label_2'].copy()
y_valid_df = y_valid['label_2'].copy()

x_test_df = x_test['label_2'].copy()

# %% [markdown]
# # Cross Validation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T07:53:25.494125Z","iopub.execute_input":"2023-09-24T07:53:25.494603Z","iopub.status.idle":"2023-09-24T08:22:14.901632Z","shell.execute_reply.started":"2023-09-24T07:53:25.494560Z","shell.execute_reply":"2023-09-24T08:22:14.900393Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:22:14.903101Z","iopub.execute_input":"2023-09-24T08:22:14.903643Z","iopub.status.idle":"2023-09-24T08:22:19.070953Z","shell.execute_reply.started":"2023-09-24T08:22:14.903610Z","shell.execute_reply":"2023-09-24T08:22:19.066473Z"}}
from sklearn.decomposition import PCA

pca = PCA(n_components=0.975, svd_solver='full')
pca.fit(x_train_df)
x_train_df_pca = pd.DataFrame(pca.transform(x_train_df))
x_valid_df_pca = pd.DataFrame(pca.transform(x_valid_df))
x_test_df_pca = pd.DataFrame(pca.transform(x_test_df))
print('Shape after PCA: ',x_train_df_pca.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:22:19.078901Z","iopub.execute_input":"2023-09-24T08:22:19.080229Z","iopub.status.idle":"2023-09-24T08:22:19.092580Z","shell.execute_reply.started":"2023-09-24T08:22:19.080168Z","shell.execute_reply":"2023-09-24T08:22:19.091081Z"}}
from sklearn import metrics

# %% [markdown]
# # SVM

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:22:19.095246Z","iopub.execute_input":"2023-09-24T08:22:19.096350Z","iopub.status.idle":"2023-09-24T08:27:30.788205Z","shell.execute_reply.started":"2023-09-24T08:22:19.096289Z","shell.execute_reply":"2023-09-24T08:27:30.786923Z"}}
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classifier = svm.SVC(kernel='linear', C=1)

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:04:24.904175Z","iopub.status.idle":"2023-09-24T13:04:24.904865Z","shell.execute_reply.started":"2023-09-24T13:04:24.904517Z","shell.execute_reply":"2023-09-24T13:04:24.904546Z"}}
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
# Label 2 
# best parameters: {'kernel': 'rbf', 'gamma': 'scale', 'degree': 4, 'class_weight': 'balanced', 'C': 100}
# 
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:05:31.750855Z","iopub.execute_input":"2023-09-24T13:05:31.751286Z","iopub.status.idle":"2023-09-24T13:08:39.379170Z","shell.execute_reply.started":"2023-09-24T13:05:31.751248Z","shell.execute_reply":"2023-09-24T13:08:39.377924Z"}}
from sklearn import svm

classifier = svm.SVC(kernel='rbf', C=100, gamma='scale', degree=4, class_weight='balanced')

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_predict_after_pca = classifier.predict(x_test_df_pca)



# %% [markdown]
# # RandomForest

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:09:53.604110Z","iopub.execute_input":"2023-09-24T13:09:53.604538Z","iopub.status.idle":"2023-09-24T13:12:00.945883Z","shell.execute_reply.started":"2023-09-24T13:09:53.604509Z","shell.execute_reply":"2023-09-24T13:12:00.944385Z"}}
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=42)

classifier.fit(x_train_df, y_train_df)

y_valid_pred = classifier.predict(x_valid_df)

print("accuracy_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_pred = classifier.predict(x_test_df)

# %% [markdown]
# # CSV Creation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:12:00.961149Z","iopub.execute_input":"2023-09-24T13:12:00.962468Z","iopub.status.idle":"2023-09-24T13:12:00.976854Z","shell.execute_reply.started":"2023-09-24T13:12:00.962424Z","shell.execute_reply":"2023-09-24T13:12:00.975593Z"}}
output_df=pd.DataFrame(columns=["ID","label_1","label_2","label_3","label_4"])

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:13:07.011887Z","iopub.execute_input":"2023-09-24T13:13:07.013857Z","iopub.status.idle":"2023-09-24T13:13:07.021650Z","shell.execute_reply.started":"2023-09-24T13:13:07.013792Z","shell.execute_reply":"2023-09-24T13:13:07.020214Z"}}
IDs = list(i for i in range(1, len(test_df)+1))
output_df["ID"] = IDs

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:13:11.534829Z","iopub.execute_input":"2023-09-24T13:13:11.535258Z","iopub.status.idle":"2023-09-24T13:13:11.542384Z","shell.execute_reply.started":"2023-09-24T13:13:11.535213Z","shell.execute_reply":"2023-09-24T13:13:11.540920Z"}}
output_df["label_2"] = y_test_predict_after_pca

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:13:14.935957Z","iopub.execute_input":"2023-09-24T13:13:14.936395Z","iopub.status.idle":"2023-09-24T13:13:14.958674Z","shell.execute_reply.started":"2023-09-24T13:13:14.936360Z","shell.execute_reply":"2023-09-24T13:13:14.957296Z"}}
output_df

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:13:20.118129Z","iopub.execute_input":"2023-09-24T13:13:20.118526Z","iopub.status.idle":"2023-09-24T13:13:20.135612Z","shell.execute_reply.started":"2023-09-24T13:13:20.118495Z","shell.execute_reply":"2023-09-24T13:13:20.134551Z"}}
output_df.to_csv('/kaggle/working/output7_2.csv',index=False)