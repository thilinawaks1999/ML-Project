{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:39:05.774483Z\",\"iopub.execute_input\":\"2023-09-24T09:39:05.775726Z\",\"iopub.status.idle\":\"2023-09-24T09:39:06.237039Z\",\"shell.execute_reply.started\":\"2023-09-24T09:39:05.775677Z\",\"shell.execute_reply\":\"2023-09-24T09:39:06.235749Z\"}}\n# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here's several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the read-only \"../input/\" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk('/kaggle/input'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:39:06.240416Z\",\"iopub.execute_input\":\"2023-09-24T09:39:06.241570Z\",\"iopub.status.idle\":\"2023-09-24T09:39:07.625821Z\",\"shell.execute_reply.started\":\"2023-09-24T09:39:06.241524Z\",\"shell.execute_reply\":\"2023-09-24T09:39:07.624669Z\"}}\nimport pandas as pd\nimport numpy as np\nfrom pandas import Series\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.decomposition import PCA\nfrom sklearn.metrics import f1_score\nimport seaborn as sns\n\nimport matplotlib.pyplot as plt\n%matplotlib inline\n\nimport warnings\nwarnings.filterwarnings(\"ignore\")\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:39:07.627421Z\",\"iopub.execute_input\":\"2023-09-24T09:39:07.628217Z\",\"iopub.status.idle\":\"2023-09-24T09:39:18.677421Z\",\"shell.execute_reply.started\":\"2023-09-24T09:39:07.628171Z\",\"shell.execute_reply\":\"2023-09-24T09:39:18.676330Z\"}}\ntrain_df = pd.read_csv('/kaggle/input/layer12/train.csv')\nvalid_df = pd.read_csv('/kaggle/input/layer12/valid.csv')\ntest_df = pd.read_csv('/kaggle/input/layer12/test.csv')\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:39:18.678556Z\",\"iopub.execute_input\":\"2023-09-24T09:39:18.678949Z\",\"iopub.status.idle\":\"2023-09-24T09:39:18.686077Z\",\"shell.execute_reply.started\":\"2023-09-24T09:39:18.678922Z\",\"shell.execute_reply\":\"2023-09-24T09:39:18.685078Z\"}}\ntrain_df.shape\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:39:18.688944Z\",\"iopub.execute_input\":\"2023-09-24T09:39:18.689305Z\",\"iopub.status.idle\":\"2023-09-24T09:39:18.729064Z\",\"shell.execute_reply.started\":\"2023-09-24T09:39:18.689277Z\",\"shell.execute_reply\":\"2023-09-24T09:39:18.728013Z\"}}\ntrain_df.head()\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:39:18.730321Z\",\"iopub.execute_input\":\"2023-09-24T09:39:18.730706Z\",\"iopub.status.idle\":\"2023-09-24T09:39:18.763588Z\",\"shell.execute_reply.started\":\"2023-09-24T09:39:18.730671Z\",\"shell.execute_reply\":\"2023-09-24T09:39:18.762464Z\"}}\nmissing_columns = train_df.columns[train_df.isnull().any()]\nmissing_counts = train_df[missing_columns].isnull().sum()\n\nprint('Missing Columns and Counts')\nfor column in missing_columns:\n    print( str(column) +' : '+ str(missing_counts[column]))\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:39:18.764727Z\",\"iopub.execute_input\":\"2023-09-24T09:39:18.765012Z\",\"iopub.status.idle\":\"2023-09-24T09:39:18.841711Z\",\"shell.execute_reply.started\":\"2023-09-24T09:39:18.764987Z\",\"shell.execute_reply\":\"2023-09-24T09:39:18.840643Z\"}}\ntrain_data = train_df.copy()\nvalid_data = valid_df.copy()\ntest_data = test_df.copy()\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:39:18.842825Z\",\"iopub.execute_input\":\"2023-09-24T09:39:18.843106Z\",\"iopub.status.idle\":\"2023-09-24T09:39:21.413613Z\",\"shell.execute_reply.started\":\"2023-09-24T09:39:18.843081Z\",\"shell.execute_reply\":\"2023-09-24T09:39:21.412573Z\"}}\ntrain_df.describe()\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:39:21.415074Z\",\"iopub.execute_input\":\"2023-09-24T09:39:21.415566Z\",\"iopub.status.idle\":\"2023-09-24T09:39:26.188637Z\",\"shell.execute_reply.started\":\"2023-09-24T09:39:21.415538Z\",\"shell.execute_reply\":\"2023-09-24T09:39:26.187797Z\"}}\nfrom sklearn.preprocessing import RobustScaler # eliminate outliers\n\nx_train = {}\nx_valid = {}\nx_test = {}\n\ny_train = {}\ny_valid = {}\ny_test = {}\n\n#create dictionaries for each label\nfor target_label in ['label_1','label_2','label_3','label_4']:\n\n  if target_label == \"label_2\":\n    train = train_df[train_df['label_2'].notna()]\n    valid = valid_df[valid_df['label_2'].notna()]\n  else:\n    train = train_df\n    valid = valid_df\n\n  test = test_df\n\n  scaler = RobustScaler()\n\n  x_train[target_label] = pd.DataFrame(scaler.fit_transform(train.drop(['label_1','label_2','label_3','label_4'], axis=1)), columns=[f'feature_{i}' for i in range(1,769)])\n  y_train[target_label] = train[target_label]\n\n  x_valid[target_label] = pd.DataFrame(scaler.transform(valid.drop(['label_1','label_2','label_3','label_4'], axis=1)), columns=[f'feature_{i}' for i in range(1,769)])\n  y_valid  [target_label] = valid[target_label]\n\n  x_test[target_label] = pd.DataFrame(scaler.transform(test.drop([\"ID\"],axis=1)), columns=[f'feature_{i}' for i in range(1,769)])\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:39:26.189847Z\",\"iopub.execute_input\":\"2023-09-24T09:39:26.190577Z\",\"iopub.status.idle\":\"2023-09-24T09:39:26.267538Z\",\"shell.execute_reply.started\":\"2023-09-24T09:39:26.190544Z\",\"shell.execute_reply\":\"2023-09-24T09:39:26.266537Z\"}}\nx_train_df = x_train['label_1'].copy()\ny_train_df = y_train['label_1'].copy()\n\nx_valid_df = x_valid['label_1'].copy()\ny_valid_df = y_valid['label_1'].copy()\n\nx_test_df = x_test['label_1'].copy()\n\n# %% [markdown]\n# # Cross Validation\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:39:26.268681Z\",\"iopub.execute_input\":\"2023-09-24T09:39:26.268969Z\",\"iopub.status.idle\":\"2023-09-24T09:58:36.084077Z\",\"shell.execute_reply.started\":\"2023-09-24T09:39:26.268944Z\",\"shell.execute_reply\":\"2023-09-24T09:58:36.082974Z\"}}\nfrom sklearn.svm import SVC\nfrom sklearn.model_selection import cross_val_score, KFold\n\n# Perform cross-validation\nscores = cross_val_score(SVC(), x_train_df, y_train_df, cv=5, scoring='accuracy')\n\nmean_accuracy = scores.mean()\nstd_accuracy = scores.std()\n# Print the cross-validation scores\nprint('Support Vector Machines')\nprint('\\n')\nprint(\"Cross-validation scores:\", scores)\nprint(f\"Mean Accuracy: {mean_accuracy:.2f}\")\nprint(f\"Standard Deviation: {std_accuracy:.2f}\")\n\n# %% [markdown]\n# # Feature Engineering\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:58:36.085503Z\",\"iopub.execute_input\":\"2023-09-24T09:58:36.086452Z\",\"iopub.status.idle\":\"2023-09-24T09:58:40.223853Z\",\"shell.execute_reply.started\":\"2023-09-24T09:58:36.086411Z\",\"shell.execute_reply\":\"2023-09-24T09:58:40.222460Z\"}}\nfrom sklearn.decomposition import PCA\n\npca = PCA(n_components=0.975, svd_solver='full')\npca.fit(x_train_df)\nx_train_df_pca = pd.DataFrame(pca.transform(x_train_df))\nx_valid_df_pca = pd.DataFrame(pca.transform(x_valid_df))\nx_test_df_pca = pd.DataFrame(pca.transform(x_test_df))\nprint('Shape after PCA: ',x_train_df_pca.shape)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:58:40.225842Z\",\"iopub.execute_input\":\"2023-09-24T09:58:40.235045Z\",\"iopub.status.idle\":\"2023-09-24T09:58:40.244635Z\",\"shell.execute_reply.started\":\"2023-09-24T09:58:40.234985Z\",\"shell.execute_reply\":\"2023-09-24T09:58:40.242861Z\"}}\nfrom sklearn import metrics\n\n# %% [markdown]\n# # SVM\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:58:40.251930Z\",\"iopub.execute_input\":\"2023-09-24T09:58:40.252636Z\",\"iopub.status.idle\":\"2023-09-24T09:59:11.308051Z\",\"shell.execute_reply.started\":\"2023-09-24T09:58:40.252566Z\",\"shell.execute_reply\":\"2023-09-24T09:59:11.307046Z\"}}\nfrom sklearn import svm\nfrom sklearn.metrics import f1_score\nfrom sklearn.metrics import confusion_matrix\nfrom sklearn.metrics import classification_report\n\nclassifier = svm.SVC(kernel='linear', C=1)\n\nclassifier.fit(x_train_df_pca, y_train_df)\n\ny_valid_pred = classifier.predict(x_valid_df_pca)\n\nprint(\"acc_score: \",metrics.accuracy_score(y_valid_df, y_valid_pred))\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T09:59:11.309368Z\",\"iopub.execute_input\":\"2023-09-24T09:59:11.309951Z\",\"iopub.status.idle\":\"2023-09-24T10:26:11.555085Z\",\"shell.execute_reply.started\":\"2023-09-24T09:59:11.309919Z\",\"shell.execute_reply\":\"2023-09-24T10:26:11.553796Z\"}}\nfrom sklearn.svm import SVC\nfrom sklearn.model_selection import RandomizedSearchCV\nfrom scipy.stats import uniform, randint\nimport numpy as np\n\nparam_dist = {\n    'C': [100,10,1,0,0.1,0.01],\n    'kernel': ['rbf','linear','poly','sigmoid'],\n    'gamma': ['scale','auto'],\n    'degree': [1,2,3,4],\n    'class_weight' : ['none','balanced']\n}\n\nsvm = SVC()\n\nrandom_search = RandomizedSearchCV(\n    svm, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, random_state=42, scoring='accuracy'\n)\n\nrandom_search.fit(x_train_df_pca, y_train_df)\n\nbest_params = random_search.best_params_\nbest_model = random_search.best_estimator_\n\nprint(\"best parameters:\", best_params)\n\n# %% [markdown]\n# Label 1 \n# best parameters: {'kernel': 'rbf', 'gamma': 'scale', 'degree': 4, 'class_weight': 'balanced', 'C': 100}\n# \n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T10:26:11.556849Z\",\"iopub.execute_input\":\"2023-09-24T10:26:11.557563Z\",\"iopub.status.idle\":\"2023-09-24T10:27:19.959459Z\",\"shell.execute_reply.started\":\"2023-09-24T10:26:11.557519Z\",\"shell.execute_reply\":\"2023-09-24T10:27:19.958291Z\"}}\nfrom sklearn import svm\n\nclassifier = svm.SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'], degree=best_params['degree'], class_weight=best_params['class_weight'])\n\nclassifier.fit(x_train_df_pca, y_train_df)\n\ny_valid_pred = classifier.predict(x_valid_df_pca)\n\nprint(\"acc_score: \",metrics.accuracy_score(y_valid_df, y_valid_pred))\n\ny_test_predict_after_pca = classifier.predict(x_test_df_pca)\n\n\n\n# %% [markdown]\n# # RandomForest\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T10:27:19.960985Z\",\"iopub.execute_input\":\"2023-09-24T10:27:19.961296Z\",\"iopub.status.idle\":\"2023-09-24T10:30:58.363124Z\",\"shell.execute_reply.started\":\"2023-09-24T10:27:19.961269Z\",\"shell.execute_reply\":\"2023-09-24T10:30:58.361921Z\"}}\nfrom sklearn.ensemble import RandomForestClassifier\n\nclassifier = RandomForestClassifier(n_estimators=100, random_state=42)\n\nclassifier.fit(x_train_df, y_train_df)\n\ny_valid_pred = classifier.predict(x_valid_df)\n\nprint(\"accuracy_score: \",metrics.accuracy_score(y_valid_df, y_valid_pred))\n\ny_test_pred = classifier.predict(x_test_df)\n\n# %% [markdown]\n# # CSV Creation\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T10:30:58.364649Z\",\"iopub.execute_input\":\"2023-09-24T10:30:58.364971Z\",\"iopub.status.idle\":\"2023-09-24T10:30:58.375407Z\",\"shell.execute_reply.started\":\"2023-09-24T10:30:58.364942Z\",\"shell.execute_reply\":\"2023-09-24T10:30:58.374301Z\"}}\noutput_df=pd.DataFrame(columns=[\"ID\",\"label_1\",\"label_2\",\"label_3\",\"label_4\"])\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T10:30:58.377599Z\",\"iopub.execute_input\":\"2023-09-24T10:30:58.378911Z\",\"iopub.status.idle\":\"2023-09-24T10:30:58.388248Z\",\"shell.execute_reply.started\":\"2023-09-24T10:30:58.378878Z\",\"shell.execute_reply\":\"2023-09-24T10:30:58.387360Z\"}}\nIDs = list(i for i in range(1, len(test_df)+1))\noutput_df[\"ID\"] = IDs\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T10:30:58.389739Z\",\"iopub.execute_input\":\"2023-09-24T10:30:58.390010Z\",\"iopub.status.idle\":\"2023-09-24T10:30:58.396731Z\",\"shell.execute_reply.started\":\"2023-09-24T10:30:58.389986Z\",\"shell.execute_reply\":\"2023-09-24T10:30:58.395814Z\"}}\noutput_df[\"label_1\"] = y_test_predict_after_pca\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T10:30:58.397846Z\",\"iopub.execute_input\":\"2023-09-24T10:30:58.398227Z\",\"iopub.status.idle\":\"2023-09-24T10:30:58.418728Z\",\"shell.execute_reply.started\":\"2023-09-24T10:30:58.398201Z\",\"shell.execute_reply\":\"2023-09-24T10:30:58.417581Z\"}}\noutput_df\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-09-24T10:30:58.420076Z\",\"iopub.execute_input\":\"2023-09-24T10:30:58.420368Z\",\"iopub.status.idle\":\"2023-09-24T10:30:58.433228Z\",\"shell.execute_reply.started\":\"2023-09-24T10:30:58.420343Z\",\"shell.execute_reply\":\"2023-09-24T10:30:58.432257Z\"}}\noutput_df.to_csv('/kaggle/working/output12_1.csv',index=False)","metadata":{"_uuid":"a1cc3f6b-cfb0-4a19-8f86-06b3bec20d8f","_cell_guid":"2ea6a0ae-235e-452b-90af-3ee83d249e8a","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}