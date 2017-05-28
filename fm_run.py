# -*- coding: utf-8 -*-
"""
baseline 2: ad.csv (creativeID/adID/camgaignID/advertiserID/appID/appPlatform) + lr
"""

import zipfile
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from feature import format_libsvm_feature
import os
import numpy as np
from fm import FactorizationMachine

# load data
data_root = "."
dfTrain = pd.read_csv("%s/train.csv"%data_root)
dfTest = pd.read_csv("%s/test.csv"%data_root)
dfAd = pd.read_csv("%s/ad.csv"%data_root)

# process data
dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
dfTest = pd.merge(dfTest, dfAd, on="creativeID")
y_train = dfTrain["label"].values

# feature engineering/encoding
enc = OneHotEncoder()
feats = ["creativeID", "adID", "camgaignID", "advertiserID", "appID", "appPlatform"]
for i,feat in enumerate(feats):
    x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))
    x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
    if i == 0:
        X_train, X_test = x_train, x_test
    else:
        X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

# # model training
# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# proba_test = lr.predict_proba(X_test)[:,1]

# format feature
''''
!!!!!!!!
This step is very very time consuming!!!
!!!!!!!!
'''
train_libsvm_fname = 'train_libsvm'
print 'train data shape:', X_train.shape
format_libsvm_feature(X_train.toarray(), y_train, train_libsvm_fname)
test_libsvm_fname = 'test_libsvm'
print 'test data shape:', X_test.shape
y_test = np.zeros(X_test.shape[0], dtype=int)
format_libsvm_feature(X_test, y_test,
                      test_libsvm_fname)
exit()

# train model
# fm = FactorizationMachine('/home/ffl/nus/MM/cur_trans/tool/libfm',
fm = FactorizationMachine('./libfm',
                              dim='1,1,8', init_stdev=0.005, iter=200,
                              method='mcmc', out='test_fm.out', task='c',
                              test=train_libsvm_fname, train=test_libsvm_fname)
fm.fit(None, None)
proba_test = fm.predict_proba(None)[:, 1]

# submission
df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission.zip", "w") as fout:
    fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)
