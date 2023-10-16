
# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:28.576199Z","iopub.execute_input":"2023-09-24T09:00:28.576671Z","iopub.status.idle":"2023-09-24T09:00:28.935003Z","shell.execute_reply.started":"2023-09-24T09:00:28.576631Z","shell.execute_reply":"2023-09-24T09:00:28.934113Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
​
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
​
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
​
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
​
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
​
# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:28.936983Z","iopub.execute_input":"2023-09-24T09:00:28.937401Z","iopub.status.idle":"2023-09-24T09:00:31.173569Z","shell.execute_reply.started":"2023-09-24T09:00:28.937376Z","shell.execute_reply":"2023-09-24T09:00:31.171483Z"}}
import pandas as pd
import numpy as np
from pandas import Series
​
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import seaborn as sns
​
import matplotlib.pyplot as plt
%matplotlib inline
​
import warnings
warnings.filterwarnings("ignore")
​
# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:31.175455Z","iopub.execute_input":"2023-09-24T09:00:31.176128Z","iopub.status.idle":"2023-09-24T09:00:39.916129Z","shell.execute_reply.started":"2023-09-24T09:00:31.176088Z","shell.execute_reply":"2023-09-24T09:00:39.914999Z"}}
train_df = pd.read_csv('/kaggle/input/layer7/train.csv')
valid_df = pd.read_csv('/kaggle/input/layer7/valid.csv')
test_df = pd.read_csv('/kaggle/input/layer7/test.csv')
​
# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:39.917475Z","iopub.execute_input":"2023-09-24T09:00:39.918665Z","iopub.status.idle":"2023-09-24T09:00:39.927905Z","shell.execute_reply.started":"2023-09-24T09:00:39.918616Z","shell.execute_reply":"2023-09-24T09:00:39.925593Z"}}
train_df.shape
​
# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:39.931906Z","iopub.execute_input":"2023-09-24T09:00:39.932506Z","iopub.status.idle":"2023-09-24T09:00:39.996945Z","shell.execute_reply.started":"2023-09-24T09:00:39.932452Z","shell.execute_reply":"2023-09-24T09:00:39.995437Z"}}
train_df.head()
​
# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:39.998358Z","iopub.execute_input":"2023-09-24T09:00:39.998663Z","iopub.status.idle":"2023-09-24T09:00:40.021910Z","shell.execute_reply.started":"2023-09-24T09:00:39.998638Z","shell.execute_reply":"2023-09-24T09:00:40.020519Z"}}
missing_columns = train_df.columns[train_df.isnull().any()]
missing_counts = train_df[missing_columns].isnull().sum()
​
print('Missing Columns and Counts')
for column in missing_columns:
    print( str(column) +' : '+ str(missing_counts[column]))
​
# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:40.023317Z","iopub.execute_input":"2023-09-24T09:00:40.023706Z","iopub.status.idle":"2023-09-24T09:00:40.054574Z","shell.execute_reply.started":"2023-09-24T09:00:40.023673Z","shell.execute_reply":"2023-09-24T09:00:40.053163Z"}}
train_data = train_df.copy()
valid_data = valid_df.copy()
test_data = test_df.copy()
​
# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:40.056009Z","iopub.execute_input":"2023-09-24T09:00:40.056396Z","iopub.status.idle":"2023-09-24T09:00:42.083851Z","shell.execute_reply.started":"2023-09-24T09:00:40.056364Z","shell.execute_reply":"2023-09-24T09:00:42.082623Z"}}
train_df.describe()
​
# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:00:42.085076Z","iopub.execute_input":"2023-09-24T09:00:42.085363Z","iopub.status.idle":"2023-09-24T09:00:46.112662Z","shell.execute_reply.started":"2023-09-24T09:00:42.085339Z","shell.execute_reply":"2023-09-24T09:00:46.111136Z"}}
from sklearn.preprocessing import RobustScaler # eliminate outliers
​
x_train = {}
x_valid = {}
x_test = {}
​
y_train = {}
y_valid = {}
y_test = {}
​
