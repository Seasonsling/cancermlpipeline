#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019-3-31 10:50
#@Author: Seasons
#@File  : Prediction.py

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB

# initialize
def initialize(dimension):
    weight = np.full((dimension, 1), 0.01)
    bias = 0.01
    return weight, bias


def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head


def forwardBackward(weight, bias, x_train, y_train):
    # Forward
    y_head = sigmoid(np.dot(weight.T, x_train) + bias)
    loss = -(y_train * np.log(y_head) + (1 - y_train) * np.log(1 - y_head))
    cost = np.sum(loss) / x_train.shape[1]

    # Backward
    derivative_weight = np.dot(x_train, ((y_head - y_train).T)) / x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
    gradients = {"Derivative Weight": derivative_weight, "Derivative Bias": derivative_bias}
    return cost, gradients


def update(weight, bias, x_train, y_train, learningRate, iteration):
    costList = []
    index = []
    # for each iteration, update weight and bias values
    for i in range(iteration):
        cost, gradients = forwardBackward(weight, bias, x_train, y_train)
        weight = weight - learningRate * gradients["Derivative Weight"]
        bias = bias - learningRate * gradients["Derivative Bias"]

        costList.append(cost)
        index.append(i)

    parameters = {"weight": weight, "bias": bias}

    print("iteration:", iteration)
    print("cost:", cost)

    plt.plot(index, costList)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.savefig("../Prediction_output/loggestic_iteration.png")
    plt.show()

    return parameters, gradients


def predict(weight, bias, x_test):
    z = np.dot(weight.T, x_test) + bias
    y_head = sigmoid(z)

    y_prediction = np.zeros((1, x_test.shape[1]))

    for i in range(y_head.shape[1]):
        if y_head[0, i] <= 0.5:
            y_prediction[0, i] = 0
        else:
            y_prediction[0, i] = 1
    return y_prediction


def logistic_regression(x_train, y_train, x_test, y_test, learningRate, iteration):
    dimension = x_train.shape[0]
    weight, bias = initialize(dimension)
    parameters, gradients = update(weight, bias, x_train, y_train, learningRate, iteration)
    y_prediction = predict(parameters["weight"], parameters["bias"], x_test)

    print("Manuel Test Accuracy of logistic_regression: {:.2f}%".format((100 - np.mean(np.abs(y_prediction - y_test)) * 100) / 100 * 100))


# KNN Model
def KNN(x_train, y_train, x_test, y_test):
    # try ro find best k value
    scoreList = []
    best = KNeighborsClassifier(n_neighbors=1)
    best.fit(x_train.T, y_train.T)
    scoreList.append(best.score(x_test.T, y_test.T))
    for i in range(2, 20):
        knn2 = KNeighborsClassifier(n_neighbors=i)  # n_neighbors means k
        knn2.fit(x_train.T, y_train.T)
        if knn2.score(x_test.T, y_test.T) > best.score(x_test.T, y_test.T):
            best = knn2
        scoreList.append(knn2.score(x_test.T, y_test.T))

    plt.plot(range(1, 20), scoreList)
    plt.xticks(np.arange(1, 20, 1))
    plt.xlabel("K value")
    plt.ylabel("Score")
    plt.savefig("../Prediction_output/knn_iteration.png")
    plt.show()
    print("Maximum KNN Score is {:.2f}%".format((max(scoreList)) * 100))

    return best

def SVM(x_train, y_train,x_test, y_test):
    # try to find the best kernel
    scoreList = []
    best = SVC(cache_size=4096, kernel='linear', probability=True, gamma='scale')
    best.fit(x_train.T, y_train.T)
    scoreList.append(('linear', best.score(x_test.T, y_test.T)))
    for i in ('sigmoid', 'poly', 'rbf'):
        svm = SVC(cache_size=4096, kernel='%s' % i, probability=True, gamma='scale')
        svm.fit(x_train.T, y_train.T)
        if svm.score(x_test.T, y_test.T) > best.score(x_test.T, y_test.T):
            best = svm
        scoreList.append((i, svm.score(x_test.T, y_test.T)))


    plt.title('Various kernels of SVM algorithms')
    plt.plot([x[0] for x in scoreList], [x[1] for x in scoreList])
    plt.xticks([x[0] for x in scoreList])
    plt.xlabel("Kernel")
    plt.ylabel("Score")
    plt.savefig("../Prediction_output/svm_kernel.png")
    plt.show()
    print("Maximum kernel Score is {:.2f}%".format((max([x[1] for x in scoreList])) * 100))

    return best


def naive_bayes(x_train, y_train, x_test, y_test):
    nb = GaussianNB(var_smoothing=1e-11)
    nb.fit(x_train.T, y_train.T)
    print("Accuracy of Naive Bayes: {:.2f}%".format(nb.score(x_test.T,y_test.T)*100))
    return nb


def confuse_matrix(knn, svm, nb, boost, random, x_train, y_train, x_test, y_test, count, number):

    knn.fit(x_train.T, y_train.T)
    svm.fit(x_train.T, y_train.T)
    nb.fit(x_train.T, y_train.T)
    boost.fit(x_train.T, y_train.T)
    random.fit(x_train.T, y_train.T)

    y_head_knn = knn.predict(x_test.T)
    y_head_svm = svm.predict(x_test.T)
    y_head_nb = nb.predict(x_test.T)
    y_head_boost = boost.predict(x_test.T)
    y_head_random = random.predict(x_test.T)

    #confuse matrix
    from sklearn.metrics import confusion_matrix

    cm_knn = confusion_matrix(y_test.T,y_head_knn)
    cm_svm = confusion_matrix(y_test.T,y_head_svm)
    cm_nb = confusion_matrix(y_test.T,y_head_nb)
    cm_boost = confusion_matrix(y_test.T, y_head_boost)
    cm_random = confusion_matrix(y_test.T, y_head_random)

    plt.figure(figsize=(24, 12))

    plt.suptitle("Confusion Matrixes of %s(class_weight = %s)" % (count, number),fontsize=24)
    plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

    plt.subplot(2,3,1)
    plt.title("K Nearest Neighbors Confusion Matrix")
    sns.heatmap(cm_knn,annot=True,cmap="Blues", fmt="d",cbar=False, linewidths=.1)

    plt.subplot(2,3,2)
    plt.title("Support Vector Machine Confusion Matrix")
    sns.heatmap(cm_svm,annot=True,cmap="Blues", fmt="d",cbar=False, linewidths=.1)

    plt.subplot(2,3,3)
    plt.title("Naive Bayes Confusion Matrix")
    sns.heatmap(cm_nb,annot=True,cmap="Blues", fmt="d",cbar=False, linewidths=.1)

    plt.subplot(2,3,4)
    plt.title("Gradient-boosting Confusion Matrixes")
    sns.heatmap(cm_boost,annot=True, cmap="Blues", fmt="d", cbar=False, linewidths=.1)

    plt.subplot(2,3,5)
    plt.title("Random_forest Confusion Matrixes")
    sns.heatmap(cm_random,annot=True, cmap="Blues", fmt="d", cbar=False, linewidths=.1)
    plt.savefig("../Prediction_output/confuse_matrix_%s_%s.png" % (count, number))

    plt.show()



def gradient_boosting(x_train, y_train, x_test, y_test):

    x_train = x_train.T
    x_test = x_test.T
    y_train = y_train.T
    y_test = y_test.T

    # 数据缩放
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # 梯度提升回归树  GBRT
    gbrt = GradientBoostingClassifier(random_state=10, warm_start=True)

    # 网格搜索算法查询最优参数（n_estimators 树个数；learning_rate 学习率； max_depth 树深度 ）
    params_gbrt = [{
        'n_estimators':[500,2000,5000,30000],
        'learning_rate':[0.01,0.1,0.05],
        'max_depth':[11,9,7,5,3]
    }]

    gbrt_grid = GridSearchCV(gbrt, params_gbrt, cv=4, n_jobs=-1)
    gbrt_grid.fit(x_train, y_train)
    #gbrt.fit(x_train, y_train)

    print('Report of gradient_boosting:\n')
    print("Best cross-validation score: {:.2f}".format(gbrt_grid.best_score_))
    print("Best parameters: ", gbrt_grid.best_params_)
    print("Accuracy on training set:{:.3f}".format(gbrt_grid.score(x_train, y_train)))
    print("Accuracy on test set:{:.3f}".format(gbrt_grid.score(x_test, y_test)))

    ## 使用gbrt的最优参数构建模型 并评估模型性能
    best_params_gbrt = gbrt_grid.best_params_

    gbrt_model = GradientBoostingClassifier(**best_params_gbrt)
    gbrt_model.fit(x_train, y_train)
    print("Accuracy on test set:{:.3f}".format(gbrt_model.score(x_test, y_test)))

    gbrt_pred = gbrt_model.predict(x_test)
    # f1 score
    print("f1 score of GBRT: {:.2f}".format(f1_score(y_test,gbrt_pred)))
    # 模型评估报告
    print("Classification report of GBRT: \n{}".format(classification_report(y_test, gbrt_pred,
                                            target_names=["no metastasis", "metastasis"])))
    gbrt_proba = gbrt_model.predict_proba(x_test)

    # 计算GBRT的准确度和召回率
    precision_gbrt, recall_gbrt,thresholds_gbrt = precision_recall_curve(y_test, gbrt_proba[:,1])
    plt.plot(precision_gbrt, recall_gbrt, label="GBRT")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend(loc=1)
    plt.savefig("../Prediction_output/gradient_boosting_iteration.png")
    plt.show()

    ## 计算混淆矩阵
    confusion_gbrt = confusion_matrix(y_test, gbrt_pred)
    print("Confusion matrix of GBRT:\n {}".format(confusion_gbrt))

    return gbrt_model


def plot_feature_importances(data, model):
    data.drop(columns=['target'], inplace=True)
    n_features = data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.savefig("../Prediction_output/plot_feature_importances.png")
    plt.show()


def random_forest(x_train, y_train, x_test, y_test, data):
    x_train = x_train.T
    x_test = x_test.T
    y_train = y_train.T
    y_test = y_test.T

    rf = RandomForestClassifier(n_jobs=-1)

    rf.fit(x_train, y_train)
    print("Accuracy of random forest model is {:.2f}".format(rf.score(x_test, y_test)))
    proba = rf.predict_proba(x_test)
    # log_proba = rf.predict_log_proba(x_test)
    print(rf.predict(x_test))
    pred_rf = rf.predict(x_test)
    print((pred_rf == y_test).sum() / y_test.size)

    # 计算随机森林的准确度和召回率
    precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, proba[:, 1])
    plt.plot(precision_rf, recall_rf, label="rf")
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.legend(loc=1)
    plt.show()

    # f1 score
    print("f1 score : {:.2f}".format(f1_score(y_test, pred_rf)))
    ## 模型评估报告
    print("classification report: \n{}".format(classification_report(y_test, pred_rf,
                                                                     target_names=["no metastasis", "metastasis"])))

    #使用网格搜索 查找最优参数
    param_grid = [{'n_estimators': [30000, 5000, 2000, 500],
                   'max_features': [3, 5, 10, 15, 20],
                   'max_depth': [3, 5, 7, 9, 11]}]
    kfold = KFold(n_splits=4, shuffle=True)

    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=4, n_jobs=-1)  ## cv=kfold
    grid_search.fit(x_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    print("Best parameters: ", grid_search.best_params_)

    print("Test score: {:.2f}".format(grid_search.score(x_test, y_test)))

    proba_grid = grid_search.predict_proba(x_test)
    # 计算随机森林的准确度和召回率
    precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, proba_grid[:, 1])
    plt.plot(precision_rf, recall_rf, label="rf")
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.legend(loc=1)
    pred_grid = grid_search.predict(x_test)
    # f1 score
    print("f1 score : {:.2f}".format(f1_score(y_test, pred_grid)))
    # 模型评估报告
    print("classification report: \n{}".format(classification_report(y_test, pred_grid,
                                                                     target_names=["no metastasis", "metastasis"])))
    params = grid_search.best_params_  ##使用网格搜索的最优参数
    model = RandomForestClassifier().set_params(**params)
    model.fit(x_train, y_train)

    ## 计算特征重要性
    plot_feature_importances(data, model)
    return model


def pltROC(knn, svm, nb, boost, random, x_train, y_trian, x_test, y_test, count, number):
    x_test = x_test.T
    y_test = y_test.T

    knn.fit(x_train.T, y_train.T)
    svm.fit(x_train.T, y_train.T)
    nb.fit(x_train.T, y_train.T)
    boost.fit(x_train.T, y_train.T)
    random.fit(x_train.T, y_train.T)

    # 绘制ROC曲线 计算AUC
    d = {knn: 'KNN', svm: 'SVM', nb: 'Naive_Bayes', boost: 'Gradient-boosting', random: 'Random forest'}
    c = {'random_forest': 'Random forest', 'svmrfecv' : 'SVMRFE', 'union': 'Union'}
    # model.predict_proba(x_test)[:,1]
    # y_test.values
    plt.figure(figsize=(24, 16))
    fig, ax = plt.subplots()
    # ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".4")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 10
    plt.title('ROC curve for cancer classifier with the feature selecting \nmethod of %s' % c[count])
    # plt.subplots_adjust(wspace=2, hspace=1, top = 1)
    plt.xlabel("FPR")
    plt.ylabel("TPR(recall)")
    plt.grid(True)
    for model in d:
        proba_rf = model.predict_proba(x_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, proba_rf)
        close_default_rf = np.argmin(np.abs(thresholds - 0.5))
        plt.plot(fpr, tpr, label='ROC Curve of %s' % d[model] + '(AUC = {:.2f})'.format(roc_auc_score(y_test, proba_rf)))
        plt.plot(fpr[close_default_rf], tpr[close_default_rf], 'o', markersize=10, fillstyle="none", c='k', mew=2)
    plt.plot(fpr[close_default_rf], tpr[close_default_rf], 'o', label = 'Thresholds 0.5 RF', markersize=10, fillstyle="none", c='k', mew=2)
    fig.tight_layout()
    plt.legend(loc=4)
    plt.savefig("../Prediction_output/AUC_%s_%s_.png" % (count, number), dpi = 300)
    plt.show()
    # rf_auc = roc_auc_score(y_test, proba_rf)
    # print("AUC of %s" % d[model]+": {:.3f}".format(rf_auc))
    # print(auc(fpr, tpr))  # 使用auc函数亦可得到roc曲线下面积
    pass


def preprocess(countpath, tranpath):
    dataframe = pd.read_csv("%s" % countpath, index_col=0)
    featurenames = np.array(dataframe.index)
    label = pd.read_csv("%s" % tranpath, index_col=0)
    dataframe = dataframe.T
    x, y = dataframe, label
    y.index = x.index
    df = x.join(y.iloc[:, 0])
    df.rename(columns={'x': 'target'}, inplace=True)

    countNoMetastasis = len(df[df.target == 0])
    countHaveMetastasis = len(df[df.target == 1])
    print("Proportion of patients without metastasis {:.2f}%".format((countNoMetastasis / (len(df.target)) * 100)))
    print("Proportion of patients with metastasis: {:.2f}%".format((countHaveMetastasis / (len(df.target)) * 100)))

    # Creating Model for Logistic Regression
    y = df.target.values
    x_data = df.drop(['target'], axis=1)
    # Normalize
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
    x = x.dropna(axis=1, how='any')  # drop all rows that have any NaN value
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # transpose matrices

    over_samples_train = SMOTE(random_state=10, n_jobs=-1, sampling_strategy=1)
    x_train, y_train = over_samples_train.fit_sample(x_train, y_train)
    over_samples_test = SMOTE(random_state=10, n_jobs=-1, sampling_strategy=1)
    x_test, y_test = over_samples_test.fit_sample(x_test, y_test)

    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

    return x_train, y_train, x_test, y_test, featurenames, df


if __name__ == '__main__':
    #countpath = input('Please enter the path of the counting matrix after rough screening:\n')
    #tranpath = input('Please enter the path of the Tumor metastasis matrix:\n')
    # try:
    for number in {'{1:1}'}:
        for count in {'random_forest', 'union', 'svmrfecv'}:
            print('---------------------------------------------------------------------')
            print('%s_%s' %(count, number))
            countpath = '../Feature_select_output/%s_filtered%s.csv' % (count, number)
            tranpath = '../modified_data/colon_tran.csv'
            x_train, y_train, x_test, y_test, featurenames, dataframe = preprocess(countpath, tranpath)
            logistic_regression(x_train, y_train, x_test, y_test, 0.2, 50000)
            knn = KNN(x_train, y_train, x_test, y_test)
            svm = SVM(x_train, y_train, x_test, y_test)
            nb = naive_bayes(x_train, y_train, x_test, y_test)
            boost = gradient_boosting(x_train, y_train, x_test, y_test)
            random = random_forest(x_train, y_train, x_test, y_test, dataframe)

            confuse_matrix(knn, svm, nb, boost, random, x_train, y_train, x_test, y_test, count, number)
            pltROC(knn, svm, nb, boost, random, x_train, y_train, x_test, y_test, count, number)

    # except:
    #     pass





