#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019-3-30 23:36
#@Author: Seasons
#@File  : SHAP.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
import shap #for SHAP values
from IPython.display import display, HTML
from pdpbox import pdp, info_plots #for partial plots
from imblearn.over_sampling import SMOTE

dataframe = pd.read_csv("../Feature_select_output/random_forest_filtered{1_6}.csv", index_col=0)
featurenames = np.array(dataframe.index)
label = pd.read_csv("../modified_data/colon_tran.csv")
dataframe = dataframe.T
X, y = dataframe, label
y.index = X.index
dt = X.join(y['x'])
dt.rename(columns={'x':'target'}, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(dt.drop('target', 1), dt['target'], test_size = 0.3, random_state=0) #split the data
feature_names = [i for i in X_train.columns]

over_samples = SMOTE(random_state=10, n_jobs=-1, sampling_strategy=1)
X_train,y_train = over_samples.fit_sample(X_train, y_train)
X_test,y_test = over_samples.fit_sample(X_test, y_test)

model = RandomForestClassifier(n_jobs=-1, max_depth=11, max_features=3, n_estimators=500)
model.fit(X_train, y_train)

estimator = model.estimators_[1]

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'

export_graphviz(estimator, out_file='tree.dot',
                feature_names = feature_names,
                class_names = y_train_str,
                rounded = True, proportion = True,
                label='root',
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

from IPython.display import Image
Image(filename = 'tree.png')

y_predict = model.predict(X_test)
y_pred_quant = model.predict_proba(X_test)[:, 1]
y_pred_bin = model.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred_bin)
print(confusion_matrix)

total=sum(sum(confusion_matrix))

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity : ', specificity)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for feature classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
print(auc(fpr, tpr))


perm = PermutationImportance(model).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = featurenames)

base_features = dt.columns.values.tolist()
base_features.remove('target')

X_test = pd.DataFrame(X_test)
X_test.columns = base_features

for feat_name in base_features:
    pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=base_features, feature=feat_name)
    pdp.pdp_plot(pdp_dist, feat_name)
    plt.savefig('../Prediction_output/%s.png' % feat_name)
    plt.show()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, plot_type="bar")

#
shap.summary_plot(shap_values[1], X_test)

#
def disease_risk_factors(model, patient):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient)

data_for_prediction = X_test.iloc[1,:].astype(float)
disease_risk_factors(model, data_for_prediction)


