
README
===========================

General description for cancermlpipeline
We provide info on:
-How to
-How to
-How to 

****

|Author|ChengChen|
|---|---|
|E-mail|3150105343@zju.edu.cn|

|Author|Ling Tianyi|
|---|---|
|E-mail|3160102182@zju.edu.cn|


## Table of Contents

* [Abstract](#abstract)
* [Algorithm](#algorithm)
* [Useful Scripts](#useful-script)
   * [Script dependencies](#script-dependencies)
   * [Run_Prediction.py](#Run_Prediction.py)
   * [Prediction.R](#prediction)
* [Data](#data)
  
   * [RNA-seq data](#RNAseq)
   * [Clinic Data](#clinicdata)
   * [Demo](#result-files)
* [Running Analyses from the Paper](#running-analyses-from-the-paper)
* [Dependencies](#dependencies)
      * [Python](#python)
      * [R](#r)
      * [Compute Environment](#compute-environment)
      * [Data Downloads](#data-downloads)
  
   * [Running all analyses](#running-all-analyses)
* [Reference](#reference)



## Abstract

A plenty of previous work have been published related to gene signature for cancer diagnose and prognostic factor analysis and  formulating a standard pipeline to screen the gene signature for distinguishing patients with high risks from those with low-risks for cancer metastasis and predicting their prognosis have becoming one of the heating topic The goal of this study was to further screen the gene signature for classification of high and low risk of cancer metastasis using a gene expression profile dataset. And to evaluating the performance difference while applying various machine learning algorithms to identify signature genes and calculating risk index. We hope to establish a standardized pipeline for modeling on different cancers and the final result could be competitive among all published methods.



## Pipeline

### **Dataset**

We download TCGA dataset as demo data to perform our research. Compared to other dataset, TCGA data has a integrated clinic record include survival data, metastasis status and recurrence information, which are crucial to classifier in supervised learning. However, some aspects in TCGA  clinic data are quite imbalance in scale such as metastasis rates in breast cancer(the metastasis probability is less than 1/10 in total due to its high benign rate. We strongly suggest to balance the data (use such as SMOTE or modify cost matrix ) before apply modeling methods in order to achieve both satisfied AOC and precision

### **Normalization** 

Before the modeling procedure, normalization and gene prefiltering should be taken as roughly dimension reduction method. We only apply genes are variable and meet the Standard Variance cutoff and have a normal expression level among all examples (to filter out mt genes ) We do not use a Differential expressed genes list given by edgeR or Deseq because these packages calculate DE genes across different labeled batches and only keep genes with a significant logfoldchange difference. We regard this kind of anticipated filter will leave out some important genes and have too many manually control factor, thus we use Standard Variance cutoff to keep top 2000 genes and use these genes to construct machine learning models.

### **Feature select** 

To build up our classifier, we need to select the most meaningful features (gene) from 2000 filtered gene profiling. By using machine learning methods such as random forest and  SVMRFECV, we concluded a 20 gene list that may explain the RNA pattern difference between metastasis patient and those who don’t. And use this 20 gene list as the feature vector used in following classifier.

### Modeling & Evaluating 

Our classifier is to identify easy-to- [metastasis](javascript:;) cancer in [primary](javascript:;) cancer’ RNAseq samples and to apply this model to predict the risk of  cancer metastasis in. Models are evaluated by AUC and precision and different algorithms are compared to achieve best performance in dataset

### Risk analysis 

Another approach that could measure the performance of our model is to associate our gene list with the clinic survival data. Cancer metastasis strongly effect patients’ life expectation and prognostic condition. We calculate risk index and use survival test to analysis the correlation between computational result and biological meaning.



## Algorithm

### Imbalance data Preprocessing

Data Imbalance can be of the following types:

1. *Under-representation of a class in one or more important predictor variables.* 
2. *Under-representation of one class in the outcome (dependent) variable.* 

In out demo, we want to predict, from a large clinical phenotype dataset, which patients are labeled with the stage of their cancer's metastasis conditions. However, that only 16% of patients labeled with M1 or other detected metastasis stages, and it's quite normal in other TCGA cancer metastasis datasets. Therefore our prediction task is a complex mission with $typeII$ imbalance data.

**The solutions can mainly divided into two aspects.** The first dimension is mainly from the layer of raw-data. And the main method is sampling. Since our demo metastasis data are unbalanced, we can sample them by some strategies, so that our data can be relatively balanced. The second schemes optimize the algorithm from the perspective of the algorithm, considering the difference in the cost of different misclassifications, so that our algorithm can also have better results under unbalanced data.

In our scripts, we take both of them into consideration.

#### SMOTE

**Oversampling** and **undersampling** in data analysis are techniques used to adjust the class distribution of a dataset (i.e. the ratio between the different classes/categories represented). 

>From Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall and W. Philip Kegelmeyer’s **“SMOTE: Synthetic Minority Over-sampling Technique”** (Journal of Artificial Intelligence Research, 2002, Vol. 16, pp. 321–357): “This paper shows that a combination of our method of over-sampling the minority (abnormal) class and under-sampling the majority (normal) class can achieve better classifier performance (in ROC space) than only under-sampling the majority class. This paper also shows that a combination of our method of over-sampling the minority class and under-sampling the majority class can achieve better classifier performance (in ROC space) than varying the loss ratios in Ripper or class priors in Naive-Bayes. Our method of over-sampling the minority class involves creating synthetic minority class examples.”

**SMOTE** is an improved scheme based on random oversampling algorithm. For random oversampling adopts the strategy of simply copying samples to increase the minority samples, it is prone to problem of overfitting, which makes the information learned by the model too special (Specific ) and not generalized, the basic idea of the SMOTE algorithm is to analyze a small number of samples and add new samples to the data set based on a small number of samples. 

**The algorithm flow** is as follows:

1. For each sample $x$ in a few classes, calculate the distance from all samples in a few sample sets by Euclidean distance, and get its *k-nearest neighbors*​.

2. Set a sampling ratio according to the sample imbalance ratio to determine the sampling magnification *N*. For each minority sample <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;$\emph{x}$" title="$\emph{x}$" />, randomly select several samples from its *k-nearest neighbors*, assuming that the selected neighbor is <img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\widehat{x}" title="\small \widehat{x}" /> .

3. For each randomly selected neighbor <img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\widehat{x}" title="\small \widehat{x}" />, construct a new sample with the original sample according to the following formula
  
   <img src="https://latex.codecogs.com/gif.latex?x_{new}&space;=&space;x&space;&plus;&space;rand(0,&space;1)&space;×&space;(\hat{x}-x)" title="x_{new} = x + rand(0, 1) × (\hat{x}-x)" />
   
<div align = center>
     <img src="README.assets/311426-26e19e9f443dbba3.png">
    <div><B><i>Figure1</i> sketch map of SMOTE</B></div>
   </div>

**The main pseudo-code**:

***Algorithm SMOTE(T, N, k)***
**Input:**  *Number of minority class samples T; Amount of SMOTE N%; Number of nearest*
*neighbors k*
**Output:** *(N/100) \* T synthetic minority class samples*

<div>
  <img src="README.assets/1559034482209.png" width = "90%", height = "90%"/>
  <div align = "center"><B>pseudo-code of SMOTE</B></div>
</div>


The SMOTE algorithm discards the practice of random oversampling and replicating samples, which can prevent the problem of random oversampling and over-fitting. It has been proved that this method can improve the performance of the classifier.



#### Cost Matrix and class_weight

Sampling algorithm solves the problem of imbalance data ML from the data level. And the other method of solving unbalanced disaster in ML is mainly based on Cost-Sensitive Learning. The core element of cost-sensitive learning method is the cost matrix. It is noted that the cost of different types of misclassifications in actual applications is different. For example, in our cancer metastatic prediction model, the cost of “diagnosing healthy people as tumor metastatic ” and “diagnosing recurrence patients as normal” are different, obviously the latter requires higher penalty weight.

Since we mainly use the python package of `sklearn` in feature selecting and modeling, another parameter  named "class_weight" also worth mentioning.

Here is how `class_weight` works: It penalizes mistakes in samples of `class[i]` with `class_weight[i]`instead of 1. So higher class-weight means you want to put more emphasis on a class. From what you say it seems class 0 is 19 times more frequent than class 1. So you should increase the `class_weight` of class 1 relative to class 0, say {0:.1, 1:.9}. If the `class_weight` doesn't sum to 1, it will basically change the regularization parameter.



### Feature Selection

In machine learning and statistics, feature selection, also known as variable selection, attribute selection or variable subset selection, is the process of selecting a subset of relevant features (variables, predictors) for use in model construction. 

The benefits of performing feature selection before modeling our data can be illustrate as follows:

1. **Reduces Overfitting**: Less redundant data means less opportunity to make decisions based on noise.

2. **Improves Accuracy**: Less misleading data means modeling accuracy improves.
3. **Reduces Training Time**: fewer data points reduce algorithm complexity and algorithms train faster.

And to build up our classifier, we select the most meaningful 20 features (gene) from 2000 filtered gene profiling, and the evaluation results proves the effectiveness of feature selection.

<a href="random"> </a>

#### Random forest

Random forests are among the most popular machine learning methods thanks to their relatively good accuracy, robustness and ease of use. They also provide two straightforward methods for feature selection: mean decrease impurity and mean decrease accuracy. And  we apply the mean decrease impurity method in our program. 

Random forest consists of a number of decision trees. Every node in the decision trees is a condition on a single feature, designed to split the dataset into two so that similar response values end up in the same set. The measure based on which the (locally) optimal condition is chosen is called impurity. For classification, it is typically either [Gini impurity](http://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) or [information gain/entropy](http://en.wikipedia.org/wiki/Information_gain_in_decision_trees) and for regression trees it is [variance](http://en.wikipedia.org/wiki/Variance). Thus when training a tree, it can be computed how much each feature decreases the weighted impurity in a tree. For a forest, the impurity decrease from each feature can be averaged and the features are ranked according to this measure.

but there are a few things to keep in mind when using the impurity based ranking. 

> Firstly, feature selection based on impurity reduction **is biased towards preferring variables with more categories**. Secondly, when the **dataset has two (or more) correlated features, then from the point of view of the model, any of these correlated features can be used as the predictor**, with no concrete preference of one over the others. But once one of them is used, the importance of others is significantly reduced since effectively the impurity they can remove is already removed by the first feature. As a consequence, **they will have a lower reported importance**. Therefore when we interpret the data, it can lead to the incorrect conclusion that one of the variables is a strong predictor while the others in the same group are unimportant, while actually they are very close in terms of their relationship with the response variable.

Due to random forest algorithm meets difficulties when dealing with those features with high correlation, We tried another feature selection method which has been proved to handle this problem effectively: SVMRFECV.

<div>
  <div align = "center">
    <span style="position: absolute; top: 0px; left: 10px;"><B>A</B></span> 
  	<img src="README.assets/1559116484461.png" width = "50%"/> 
    <span style="position: absolute; top: 0px; left: 490px;"><B>B</B></span> 
  	<img src="README.assets/1559118453159.png" width = "40%" align = "right"/>
  </div>	
    <div align = "left"><br><B><i>Figure2 A) Sturcture of randomforest.</i></B> It randomize the variables (columns) and data (rows), generates thousands of classification trees, and then summarize the results of the classification tree. <B><i>B) Importance of top 20 features(genes)</i></B></div>
</div>

#### SVMRFECV

SVM-RFE was introduced by Guyon et al. for selecting genes from microarray data analysis for cancer classification. It includes four steps: 

​	1) Train an SVM on the training set; 

​	2) calculate ranking criteria based on the SVM weights; 

​	3) Eliminate features with the smallest ranking criterion; and 

​	4) Repeat the process. 

The feature elimination method **is sensitive to small perturbations of the training set**. The features it extracts from training set might not have good prediction performance in an independent testing set. And Fan Zhang .etc(2013, [BMC Med Genomics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3552693/#)) adopted leave-one-out cross validation method to improve the stability and robustness of SVM-RFE. In addition, they chose |*W*| as ranking criteria instead of $W^2 $in the SVM-RFE-CV algorithm. The SVM-RFE chose $C_i = W^2_i$ as ranking criteria and eliminates the feature with smallest ranking criterion. The original optimization equation in SVM actually depends on the absolute value of weight $|W|$. Substituting $½W^2$ for $|W|$ can change the non-convex optimization to a quadratic programming optimization which is more easy to solve mathematically. But when we loop the feature elimination based on leave-one-out cross-validation, $½W^2$ loses its advantages over $|W|$ on convexity of optimization. And $|W|$ has bigger ranking criteria than$½W^2$, which makes optimization selection more accurate. Therefore, we chose $|W|$ as ranking criteria in the SVM-RFE-CV algorithm.

The SVM Recursive Feature Elimination method based on Cross-Validation (SVM-RFE-CV) is described as follows:

```pascal
k = **K**; *#*Select All features

for (i in 1:n) #n is the sample size
{
  Build a SVM using the ith sample as testing set and others as training set;
  Calculate the feature weight Wi and the error rate Ei;
}

Sum up weights: W = sum(abs(Wi));
Sum up error rates: E = sum(Ei);

E0 = E;
while (E < = E0)
{
  E0 = E;
  rkw = rank(W); #rank the feature score
  k = k[which(rkw > 1)]; # remove features with lowest feature score
  
  for (i in 1:n)
  {
     * Build a SVM using the ith sample as testing set and others as training set;
     * Calculate the feature weight Wi and the error rate Ei;
  }

  Sum up weights: W = sum(abs(Wi));
  Sum up error rates: E = sum(Ei);
}
```

The error rate is calculated by 1 minus accuracy. All error rates for the n cross validations are summed up as determination of loop iterations.



### Classification & Prediction

Classification is the process of predicting the class of given data points. Classes are sometimes called as targets/ labels or categories. Classification predictive modeling is the task of approximating a mapping function (f) from input variables (X) to discrete output variables (y).

Classification belongs to the category of supervised learning where the targets also provided with the input data. There are many applications in classification in many domains such as in credit approval, medical diagnosis, target marketing etc.

There are two types of learners in classification as lazy learners and eager learners.

1. **Lazy learners**: Lazy learners simply store the training data and wait until a testing data appear. When it does, classification is conducted based on the most related data in the stored training data. Compared to eager learners, lazy learners have less training time but more time in predicting. 

    *Ex. k-nearest neighbor, Case-based reasoning*

2. **Eager learners**: Eager learners construct a classification model based on the given training data before receiving data for classification. It must be able to commit to a single hypothesis that covers the entire instance space. Due to the model construction, eager learners take a long time for train and less time to predict.

   *Ex. Decision Tree, Naive-Bayes, Artificial Neural Networks*

#### Logistic Regression

Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. Unlike linear regression which outputs continuous number values, logistic regression transforms its output using the logistic sigmoid function to return a probability value which can then be mapped to two or more discrete classes.

#### SVM

**Support vector machines (SVMs)** are a set of supervised learning methods used for classification, regression and outliers detection. Intuitively, a good separation is achieved by the hyper-plane that has the largest distance to the nearest training data points of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.

<div align = center>
    <img src="README.assets/1559147208231.png" width = "80%">
    <div><B><i>Figure3</i></B> SVM schematic</div>
</div>

The *kernel function* can be any of the following:

- linear: <img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\langle&space;x,&space;x'\rangle" title="\large \langle x, x'\rangle" />.
- polynomial:<img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;(\gamma&space;\langle&space;x,&space;x'\rangle&space;&plus;&space;r)^d" title="\large (\gamma \langle x, x'\rangle + r)^d" />. dd is specified by keyword `degree`, rr by `coef0`.
- rbf: <img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\exp(-\gamma&space;\|x-x'\|^2)" title="\large \exp(-\gamma \|x-x'\|^2)" />. <img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\gamma" title="\large \gamma" /> is specified by keyword `gamma`, must be greater than 0.
- sigmoid<img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\tanh(\gamma&space;\langle&space;x,x'\rangle&space;&plus;&space;r)" title="\large \tanh(\gamma \langle x,x'\rangle + r)" />, where <img src="https://latex.codecogs.com/gif.latex?r" title="r" /> is specified by `coef0`.

The advantages of support vector machines are:

> - Effective in high dimensional spaces.
> - Still effective in cases where number of dimensions is greater than the number of samples.
> - Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
> - Versatile: different [Kernel functions](https://scikit-learn.org/stable/modules/svm.html#svm-kernels) can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

> - If the number of features is much greater than the number of samples, avoid over-fitting in choosing [Kernel functions](https://scikit-learn.org/stable/modules/svm.html#svm-kernels) and regularization term is crucial.
> - SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see [Scores and probabilities](https://scikit-learn.org/stable/modules/svm.html#scores-probabilities), below).

#### Naive-bayes

Naive Bayes is a probabilistic classifier inspired by the Bayes theorem under a simple assumption which is the attributes are conditionally independent.

<img src="README.assets/1_IgqsLmWDtDHjCnrpahWuag.png" align = "center" width = "50%" >

The classification is conducted by deriving the maximum posterior which is the maximal $P(C_i|X)$ with the above assumption applying to Bayes theorem. This assumption greatly reduces the computational cost by only counting the class distribution. Even though the assumption is not valid in most cases since the attributes are dependent, surprisingly Naive Bayes has able to perform impressively.

Naive Bayes is a very simple algorithm to implement and good results have obtained in most cases. It can be easily scalable to larger datasets since it takes linear time, rather than by expensive iterative approximation as used for many other types of classifiers.

Naive Bayes can suffer from a problem called the zero probability problem. When the conditional probability is zero for a particular attribute, it fails to give a valid prediction. This needs to be fixed explicitly using a Laplacian estimator.

#### KNN

*k*-Nearest Neighbor is a lazy learning algorithm which stores all instances correspond to training data points in n-dimensional space. When an unknown discrete data is received, it analyzes the closest k number of instances saved (nearest neighbors)and returns the most common class as the prediction and for real-valued data it returns the mean of k nearest neighbors.

<div align = center>
    <img src="README.assets/1_nryjAUA_yMXVZ4BtZUUx-w.png" width = "60%">
    <div><B><i>Figure3</i> kNN schematic</B></div>
</div>



In the distance-weighted nearest neighbor algorithm, it weights the contribution of each of the k neighbors according to their distance using the following query giving greater weight to the closest neighbors.

<div align = center>
    <img src="README.assets/1_Ct-bj81E0PTAV39yAaXlxA.png" width="13%">
    <br>
    <span style="font-size:12px">Distance calculating query</span>
</div>

#### Gradient-boosting

> **Gradient boosting** is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an **ensemble of weak prediction models**, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

the intuition behind `gradient boosting` algorithm is to repetitively leverage the patterns in residuals and strengthen a model with weak predictions and make it better. Once we reach a stage that residuals do not have any pattern that could be modeled, we can stop modeling residuals (otherwise it might lead to overfitting). Algorithmically, we are minimizing our loss function, such that test loss reach its minima.

In pseudocode, the generic gradient boosting method is:

<div>
    <img src="README.assets/1559386800740.png" width = "100%" align=left>
    <div align = center><i>gradient boosting algorithm</i></div>
</div>



#### Random forest

[see here]()

### Evaluation index

#### SHAP value

**SHAP (SHapley Additive exPlanations)** is a unified approach to explain the output of any machine learning model. SHAP connects game theory with local explanations, uniting several previous methods and representing the only possible consistent and locally accurate additive feature attribution method based on expectations.

SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value, in other word, represent a feature's responsibility for a change in the model output

#### Confusion matrix, Accuracy and F1 score

 A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. All the measures except AUC can be calculated by using left most four parameters:

* **True Positives (TP)** - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes. 
* **True Negatives (TN)** - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no. 
* **False Positives (FP)** – When actual class is no and predicted class is yes. 
* **False Negatives (FN)** – When actual class is yes but predicted class in no.

<div align = center>
    <img src="README.assets/1_Z54JgbS4DUwWSknhDCvNTQ.png" width = "50%">
    <div align = center><i>confusion matrix</i></div>
</div>



**Accuracy** - Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations.  

<img src="https://latex.codecogs.com/gif.latex?\textrm{Accuracy}&space;=&space;\frac{\textrm{nr.&space;correct&space;predictions}}{\textrm{nr.&space;total&space;predictions}}&space;=&space;\frac{\textrm{TP&plus;TN}}{\textrm{TP&plus;TN&plus;FP&plus;FN}}" title="\textrm{Accuracy} = \frac{\textrm{nr. correct predictions}}{\textrm{nr. total predictions}} = \frac{\textrm{TP+TN}}{\textrm{TP+TN+FP+FN}}" />

**Precision** - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. High precision relates to the low false positive rate. We have got 0.788 precision which is pretty good.

<img src="https://latex.codecogs.com/gif.latex?\textrm{Precision}&space;=&space;\frac{\textrm{TP}}{\textrm{TP&plus;FP}}" title="\textrm{Precision} = \frac{\textrm{TP}}{\textrm{TP+FP}}" />

**Recall** (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class.

<img src="https://latex.codecogs.com/gif.latex?\textrm{Recall}&space;=&space;\frac{\textrm{TP}}{\textrm{TP&plus;FN}}" title="\textrm{Recall} = \frac{\textrm{TP}}{\textrm{TP+FN}}" />

**F1 score** - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if we have an uneven class distribution. 

<img src="https://latex.codecogs.com/gif.latex?F_{1}&space;=&space;2&space;\times\frac{\textrm{Precision}&space;\times&space;\textrm{Recall}}{\textrm{Precision}&space;&plus;&space;\textrm{Recall}}" title="F_{1} = 2 \times\frac{\textrm{Precision} \times \textrm{Recall}}{\textrm{Precision} + \textrm{Recall}}" />

#### ROC and PRC curve

Accuracy is a great measure but only when we have symmetric datasets where values of false positive and false negatives are almost same.



## Useful Scripts

### Script dependencies

#### Python script dependencies:

seaborn\==0.9.0

pandas\==0.24.2

PDPbox\==0.2.0

shap\==0.29.1

numpy\==1.16.2

imbalanced_learn\==0.4.3

matplotlib\==3.0.3

eli5\==0.8.2

ipython\==7.5.0

imblearn\==0.0

scikit_learn\==0.21.2

tensorflow==1.13.1

the method of 

```python
pip install -r requirements.txt
```



### Run_Prediction.py

balabala

### prediction.R

## Data

### RNAseq data

### Clinic data

### Demo



##  Running Analyses from the Paper

## Dependencies

### R

### Python

### Data Downloads

## Reference
