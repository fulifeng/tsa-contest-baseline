
# Read csv train&test data.


import pandas
import math
import collections
from time import time

def read_data(trainFile, testFile, categorical_fields, historical_fields):
    dfTrain, dfTest = pandas.read_csv(trainFile), pandas.read_csv(testFile)
    num_train, num_test = len(dfTrain), len(dfTest)
    map_feature_id = {} # map from feature_name to feature_id (starts from 0)
    trainData, testData, trainLabels, trainTime = [],[], dfTrain['label'].tolist(), dfTrain['clickTime'].tolist() # list of dicts for train and test data.
    dictTrain, dictTest = {}, {}
    for field in categorical_fields:
        dictTrain[field] = dfTrain[field].tolist()
        dictTest[field] = dfTest[field].tolist()
    for field in historical_fields:
        dictTrain[field] = dfTrain[field].tolist()
        dictTest[field] = dfTest[field].tolist()
    del dfTrain
    del dfTest
    #Read test.csv and generate testData
    t1 = time()
    for index in xrange(num_test):
        instance = {}
        # process categorical fields
        for field in categorical_fields:
            value = dictTest[field][index]
            if field == 'sitesetID' or value!=0:
                feature_name = '%s_%d' %(field, value)
                if map_feature_id.has_key(feature_name) == False:
                    map_feature_id[feature_name] = len(map_feature_id)
                feature_id = map_feature_id[feature_name]
                instance[feature_id] = 1.0
        # process historical fields
        for field in historical_fields:
            values = str(dictTest[field][index])
            values = eval(values.replace(';',','))
            for value in values:
                feature_name = '%s_%s' %(field, value)
                if map_feature_id.has_key(feature_name) == False:
                    map_feature_id[feature_name] = len(map_feature_id)
                feature_id = map_feature_id[feature_name]
                instance[feature_id] = 1.0 / math.sqrt(len(values))
        testData.append(instance)
    #print "Test data done [%.1f sec]." %(time()-t1)

    #Read train.csv and generate trainData and trainLabels
    t1 = time()
    for index in xrange(num_train):
        instance = {}
        # process categorical fields
        for field in categorical_fields:
            value = dictTrain[field][index]
            if field == 'sitesetID' or value!=0:
                feature_name = '%s_%d' %(field, value)
                if map_feature_id.has_key(feature_name) == False:
                    map_feature_id[feature_name] = len(map_feature_id)
                feature_id = map_feature_id[feature_name]
                instance[feature_id] = 1.0
        # process historical fields
        for field in historical_fields:
            values = str(dictTrain[field][index])
            values = eval(values.replace(';',','))
            for value in values:
                feature_name = '%s_%s' %(field, value)
                if map_feature_id.has_key(feature_name) == False:
                    map_feature_id[feature_name] = len(map_feature_id)
                feature_id = map_feature_id[feature_name]
                instance[feature_id] = 1.0 / math.sqrt(len(values))
        trainData.append(instance)
    #print "Train data   [%.1f sec]." %(time()-t1)
    return trainData, trainLabels, trainTime, testData

def out_data(trainData, trainLabels, trainTime, testData):
    # split data
    new_trainData, new_validationData, new_trainLables, new_validationLables = [], [], [], []
    for index in range( len(trainTime) ):
        if trainTime[index] < 280000:
            new_trainData.append( trainData[index] )
            new_trainLables.append( trainLabels[index] )
        elif trainTime[index] >= 280000 and trainTime[index] < 310000:
            new_validationData.append( trainData[index] )
            new_validationLables.append( trainLabels[index] )

    testLables = []
    for i in range( len(testData) ):
        testLables.append(0)

    # output data
    output(new_trainLables, new_trainData, 'train.libfm')
    output(new_validationLables, new_validationData, 'validation.libfm')
    output(testLables, testData, 'test.libfm')

def output(Lables, Data, file):
    f = open(file, 'w')
    for index in range( len(Lables) ):
        f.write( str(Lables[index]) )
        dic = Data[index]
        dic = {int(k): v for k, v in dic.items()}
        for key in sorted(dic):
            f.write(' ' + str(key) + ':' + str(dic[key]))
        f.write('\n')
    f.close()

if __name__ == '__main__':
    trainFile = 'merged-data/train_ad_app_position_user_actions.csv'
    testFile = 'merged-data/test_ad_app_position_user_actions.csv'
    categorical_fields=["creativeID", "adID", "camgaignID", "advertiserID", "appID", "appPlatform"]
    historical_fields=["appActions"]
    t1 = time()
    trainData, trainLabels, trainTime, testData = read_data(trainFile, testFile, categorical_fields, historical_fields)
    print ("Load data done. [%.1f sec]" %(time()-t1))

    out_data(trainData, trainLabels, trainTime, testData)
