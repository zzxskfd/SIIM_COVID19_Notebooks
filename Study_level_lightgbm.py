# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from time import time
from sklearn.model_selection import train_test_split
# %%
# Paths
output_path = "output/"
# %%
# Load feature data
start_time = time()
X_data = np.load(output_path + "X_train_data.npy")
y_data = np.load(output_path + "y_train_data.npy")
print("X_data shape:", X_data.shape)
print("Time elapsed:", time() - start_time)
# %%
# Reshape data
start_time = time()
X_data = X_data.reshape(X_data.shape[0], -1)
X_data = X_data[:, 0: 1000]
print("X_data shape:", X_data.shape)
print("Time elapsed:", time() - start_time)
# # %%
# # Try to use pca to reduce dimension
# start_time = time()
# from sklearn.decomposition import PCA
# pca = PCA()
# pca.fit(X_data)
# print("Time elapsed:", time() - start_time)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)
# # %%
# # Save pca result
# X_feature_pca = np.array(pca.transform(X_data))
# print("X_feature_pca shape:", X_feature_pca.shape)
# np.save("X_feature_pca.npy", X_feature_pca)
# %%
# Split the train/test data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 114514, shuffle=True)
print("X_train len: " + str(len(X_train)))
print("y_train len: " + str(len(y_train)))
print("X_test len: " + str(len(X_test)))
print("y_test len: " + str(len(y_test)))
# %%
# Prepare Datasets for LightGBM
train_data = lgb.Dataset(data=X_train,label=y_train)
test_data = lgb.Dataset(data=X_test,label=y_test)
print("Done.")
# %%
# Set parameters
param = {'num_leaves':100, 
        #  'num_trees':100,
         'objective':'binary', 
         'metric':['auc', 'binary_logloss']}
# %%
# Train
start_time = time()
num_round = 100
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=8)
bst.save_model(output_path + "model_LightGBM.txt")
print("Time elapsed:", time() - start_time)
# %%
