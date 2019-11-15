#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019-3-24 21:05
#@Author: Seasons
#@File  : Random forest.py

from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from pandas import DataFrame
import os
import csv

def random_forest_fs(featurenames, x, y):

    #随机森林训练
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1234)

    forest = RandomForestClassifier(n_estimators=100000, n_jobs=-1, class_weight={0:1, 1:1})
    forest.fit(x_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    return x_train, importances, indices

def svmrfecv_fs(featurenames, x, y):

    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear", cache_size=8192, class_weight={0:1, 1: 1})
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(4), n_jobs=-1)
    rfecv.fit(x, y)

    print("Optimal number of features with svmrfecv: %d" % rfecv.n_features_)

    result = sorted(zip(map(lambda x: round(x, 4), rfecv.grid_scores_), featurenames), reverse=True)
    resultframe = DataFrame(result)
    with open('../Feature_select_output/svmrfecv.txt', 'w', newline='') as out:
        resultframe.iloc[:, 1].to_csv(out, index=False)

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.savefig('svmrefcv.png')
    #plt.show()

    return result


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def writecsv(str, method, Dataframe, number):
    mkdir('../Feature_select_output')
    with open('../Feature_select_output/%s.csv' % str, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(Dataframe.columns.values.tolist())
        for f in range(len(method)):
            if f < int(number):
                print(method[f])
                line = list(Dataframe.loc[Dataframe['Unnamed: 0'] == method[f][1]].values)
                writer.writerow(line[0])

if __name__ == '__main__':
    #countpath = input('Please enter the path of the counting matrix after rough screening:\n')
    countpath = '../modified_data/colon_filtered.csv'
    #tranpath = input('Please enter the path of the Tumor metastasis matrix:\n')
    tranpath = '../modified_data/colon_tran.csv'
    #number = input('Please enter the maximum gene numbers after the feature selection:\n')
    number = 20

    try:
        dataframe = pd.read_csv("%s" % countpath, index_col=0)  # 读入表达矩阵
        label = pd.read_csv("%s" % tranpath, index_col=0)  # 读入转移情况
        dataframe = dataframe.T
        featurenames = dataframe.columns
        # 矩阵预处�?
        x, y = dataframe.iloc[:, 0:].values, label.iloc[:, 0].values

        x_selected, importances, indices = random_forest_fs(featurenames, x, y)
        rfecv = list(svmrfecv_fs(featurenames, x, y))

        # 标准�?
        random = []
        rfe = []
        sum1 = 0
        sum2 = 0
        for i in range(0, x_selected.shape[1]):
            sum1 = sum1 + importances[indices[i]]
        for i in range(0, x_selected.shape[1]):
            random.append([importances[indices[i]] / sum1, featurenames[indices[i]]])

        threshold = 1.0 / len(featurenames)
        rfecv = rfecv[:]
        for i in range(0, len(rfecv)):
            sum2 = sum2 + rfecv[i][0]
        for i in range(0, len(rfecv)):
            rfe.append((rfecv[i][0] / sum2, rfecv[i][1]))

        union = []
        for i in random:
            for j in rfe:
                if i[1] == j[1]:
                    union.append((i[0]*1.2 + j[0], i[1]))
        union = sorted(union, key=lambda x: x[0], reverse=True)

        Dataframe = pd.read_csv("%s" % countpath)

        writecsv('union_filtered{1:1}', union[0:20], Dataframe, number)
        writecsv('svmrfecv_filtered{1:1}', rfe[0:20], Dataframe, number)
        writecsv('random_forest_filtered{1:1}', random[0:20], Dataframe, number)
        with open('Complete_ranking.txt', 'a') as rank:
            rank.writelines(random)
            rank.writelines(rfe)
            rank.writelines(union)
    except:
        print('errer')



