# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:20.544753Z","iopub.execute_input":"2023-09-24T12:44:20.545159Z","iopub.status.idle":"2023-09-24T12:44:20.992517Z","shell.execute_reply.started":"2023-09-24T12:44:20.545128Z","shell.execute_reply":"2023-09-24T12:44:20.991280Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:20.994675Z","iopub.execute_input":"2023-09-24T12:44:20.995171Z","iopub.status.idle":"2023-09-24T12:44:21.679297Z","shell.execute_reply.started":"2023-09-24T12:44:20.995140Z","shell.execute_reply":"2023-09-24T12:44:21.677909Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:21.681337Z","iopub.execute_input":"2023-09-24T12:44:21.681791Z","iopub.status.idle":"2023-09-24T12:44:33.422775Z","shell.execute_reply.started":"2023-09-24T12:44:21.681756Z","shell.execute_reply":"2023-09-24T12:44:33.421544Z"}}
train_df = pd.read_csv('/kaggle/input/layer12/train.csv')
valid_df = pd.read_csv('/kaggle/input/layer12/valid.csv')
test_df = pd.read_csv('/kaggle/input/layer12/test.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:33.425705Z","iopub.execute_input":"2023-09-24T12:44:33.426038Z","iopub.status.idle":"2023-09-24T12:44:33.434247Z","shell.execute_reply.started":"2023-09-24T12:44:33.426010Z","shell.execute_reply":"2023-09-24T12:44:33.433079Z"}}
train_df.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:33.436164Z","iopub.execute_input":"2023-09-24T12:44:33.436626Z","iopub.status.idle":"2023-09-24T12:44:33.488015Z","shell.execute_reply.started":"2023-09-24T12:44:33.436585Z","shell.execute_reply":"2023-09-24T12:44:33.487048Z"}}
train_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:33.489231Z","iopub.execute_input":"2023-09-24T12:44:33.490097Z","iopub.status.idle":"2023-09-24T12:44:33.526737Z","shell.execute_reply.started":"2023-09-24T12:44:33.490065Z","shell.execute_reply":"2023-09-24T12:44:33.525377Z"}}
missing_columns = train_df.columns[train_df.isnull().any()]
missing_counts = train_df[missing_columns].isnull().sum()

print('Missing Columns and Counts')
for column in missing_columns:
    print( str(column) +' : '+ str(missing_counts[column]))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:33.528236Z","iopub.execute_input":"2023-09-24T12:44:33.528625Z","iopub.status.idle":"2023-09-24T12:44:33.609025Z","shell.execute_reply.started":"2023-09-24T12:44:33.528585Z","shell.execute_reply":"2023-09-24T12:44:33.607778Z"}}
train_data = train_df.copy()
valid_data = valid_df.copy()
test_data = test_df.copy()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:33.610758Z","iopub.execute_input":"2023-09-24T12:44:33.611207Z","iopub.status.idle":"2023-09-24T12:44:36.378427Z","shell.execute_reply.started":"2023-09-24T12:44:33.611164Z","shell.execute_reply":"2023-09-24T12:44:36.377290Z"}}
train_df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:36.379541Z","iopub.execute_input":"2023-09-24T12:44:36.379859Z","iopub.status.idle":"2023-09-24T12:44:41.384402Z","shell.execute_reply.started":"2023-09-24T12:44:36.379832Z","shell.execute_reply":"2023-09-24T12:44:41.382858Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:41.389791Z","iopub.execute_input":"2023-09-24T12:44:41.390200Z","iopub.status.idle":"2023-09-24T12:44:41.471788Z","shell.execute_reply.started":"2023-09-24T12:44:41.390164Z","shell.execute_reply":"2023-09-24T12:44:41.470603Z"}}
x_train_df = x_train['label_2'].copy()
y_train_df = y_train['label_2'].copy()

x_valid_df = x_valid['label_2'].copy()
y_valid_df = y_valid['label_2'].copy()

x_test_df = x_test['label_2'].copy()

# %% [markdown]
# # Cross Validation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T12:44:41.473202Z","iopub.execute_input":"2023-09-24T12:44:41.473558Z","iopub.status.idle":"2023-09-24T13:10:04.366253Z","shell.execute_reply.started":"2023-09-24T12:44:41.473527Z","shell.execute_reply":"2023-09-24T13:10:04.364911Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:10:04.367777Z","iopub.execute_input":"2023-09-24T13:10:04.368204Z","iopub.status.idle":"2023-09-24T13:10:09.478679Z","shell.execute_reply.started":"2023-09-24T13:10:04.368173Z","shell.execute_reply":"2023-09-24T13:10:09.477457Z"}}
from sklearn.decomposition import PCA

pca = PCA(n_components=0.975, svd_solver='full')
pca.fit(x_train_df)
x_train_df_pca = pd.DataFrame(pca.transform(x_train_df))
x_valid_df_pca = pd.DataFrame(pca.transform(x_valid_df))
x_test_df_pca = pd.DataFrame(pca.transform(x_test_df))
print('Shape after PCA: ',x_train_df_pca.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:10:09.480380Z","iopub.execute_input":"2023-09-24T13:10:09.481087Z","iopub.status.idle":"2023-09-24T13:10:09.486757Z","shell.execute_reply.started":"2023-09-24T13:10:09.481035Z","shell.execute_reply":"2023-09-24T13:10:09.485557Z"}}
from sklearn import metrics

# %% [markdown]
# # SVM

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:17:38.076745Z","iopub.execute_input":"2023-09-24T13:17:38.077206Z","iopub.status.idle":"2023-09-24T13:24:11.700948Z","shell.execute_reply.started":"2023-09-24T13:17:38.077164Z","shell.execute_reply":"2023-09-24T13:24:11.699807Z"}}
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classifier = svm.SVC(kernel='linear', C=1)

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:16:43.481952Z","iopub.status.idle":"2023-09-24T13:16:43.482381Z","shell.execute_reply.started":"2023-09-24T13:16:43.482180Z","shell.execute_reply":"2023-09-24T13:16:43.482200Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:30:52.673420Z","iopub.execute_input":"2023-09-24T13:30:52.673876Z","iopub.status.idle":"2023-09-24T13:32:13.365042Z","shell.execute_reply.started":"2023-09-24T13:30:52.673842Z","shell.execute_reply":"2023-09-24T13:32:13.363830Z"}}
from sklearn import svm

classifier = svm.SVC(kernel='rbf', C=100, gamma='scale', degree=4, class_weight='balanced')

classifier.fit(x_train_df_pca, y_train_df)

y_valid_pred = classifier.predict(x_valid_df_pca)

print("acc_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_predict_after_pca = classifier.predict(x_test_df_pca)



# %% [markdown]
# # RandomForest

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:32:17.472513Z","iopub.execute_input":"2023-09-24T13:32:17.472952Z","iopub.status.idle":"2023-09-24T13:34:21.766009Z","shell.execute_reply.started":"2023-09-24T13:32:17.472913Z","shell.execute_reply":"2023-09-24T13:34:21.764859Z"}}
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=42)

classifier.fit(x_train_df, y_train_df)

y_valid_pred = classifier.predict(x_valid_df)

print("accuracy_score: ",metrics.accuracy_score(y_valid_df, y_valid_pred))

y_test_pred = classifier.predict(x_test_df)

# %% [markdown]
# # CSV Creation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:34:27.868975Z","iopub.execute_input":"2023-09-24T13:34:27.870035Z","iopub.status.idle":"2023-09-24T13:34:27.877383Z","shell.execute_reply.started":"2023-09-24T13:34:27.869988Z","shell.execute_reply":"2023-09-24T13:34:27.876516Z"}}
output_df=pd.DataFrame(columns=["ID","label_1","label_2","label_3","label_4"])

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:34:32.706685Z","iopub.execute_input":"2023-09-24T13:34:32.707123Z","iopub.status.idle":"2023-09-24T13:34:32.715674Z","shell.execute_reply.started":"2023-09-24T13:34:32.707088Z","shell.execute_reply":"2023-09-24T13:34:32.714488Z"}}
IDs = list(i for i in range(1, len(test_df)+1))
output_df["ID"] = IDs

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:34:34.947741Z","iopub.execute_input":"2023-09-24T13:34:34.948138Z","iopub.status.idle":"2023-09-24T13:34:34.953912Z","shell.execute_reply.started":"2023-09-24T13:34:34.948107Z","shell.execute_reply":"2023-09-24T13:34:34.952623Z"}}
output_df["label_2"] = y_test_predict_after_pca

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:34:37.908077Z","iopub.execute_input":"2023-09-24T13:34:37.909137Z","iopub.status.idle":"2023-09-24T13:34:37.928063Z","shell.execute_reply.started":"2023-09-24T13:34:37.909088Z","shell.execute_reply":"2023-09-24T13:34:37.926768Z"}}
output_df

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T13:34:41.277491Z","iopub.execute_input":"2023-09-24T13:34:41.277906Z","iopub.status.idle":"2023-09-24T13:34:41.290091Z","shell.execute_reply.started":"2023-09-24T13:34:41.277876Z","shell.execute_reply":"2023-09-24T13:34:41.288877Z"}}
output_df.to_csv('/kaggle/working/output12_2.csv',index=False)