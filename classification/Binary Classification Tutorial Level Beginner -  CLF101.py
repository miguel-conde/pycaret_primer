# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: env_pycaret
#     language: python
#     name: env_pycaret
# ---

# + {"colab_type": "text", "id": "Y57RMM1LEQmR", "cell_type": "markdown"}
# #  <span style="color:orange">Binary Classification Tutorial (CLF101) - Level Beginner</span>

# + {"colab_type": "text", "id": "GM-nQ7LqEQma", "cell_type": "markdown"}
# **Created using: PyCaret 2.2** <br />
# **Date Updated: November 11, 2020**
#
# # 1.0 Tutorial Objective
# Welcome to the Binary Classification Tutorial (CLF101) - Level Beginner. This tutorial assumes that you are new to PyCaret and looking to get started with Binary Classification using the `pycaret.classification` Module.
#
# In this tutorial we will learn:
#
#
# * **Getting Data:**  How to import data from PyCaret repository
# * **Setting up Environment:**  How to setup an experiment in PyCaret and get started with building classification models
# * **Create Model:**  How to create a model, perform stratified cross validation and evaluate classification metrics
# * **Tune Model:**  How to automatically tune the hyper-parameters of a classification model
# * **Plot Model:**  How to analyze model performance using various plots
# * **Finalize Model:** How to finalize the best model at the end of the experiment
# * **Predict Model:**  How to make predictions on new / unseen data
# * **Save / Load Model:**  How to save / load a model for future use
#
# Read Time : Approx. 30 Minutes
#
#
# ## 1.1 Installing PyCaret
# The first step to get started with PyCaret is to install pycaret. Installation is easy and will only take a few minutes. Follow the instructions below:
#
# #### Installing PyCaret in Local Jupyter Notebook
# `pip install pycaret`  <br />
#
# #### Installing PyCaret on Google Colab or Azure Notebooks
# `!pip install pycaret`
#
#
# ## 1.2 Pre-Requisites
# - Python 3.6 or greater
# - PyCaret 2.0 or greater
# - Internet connection to load data from pycaret's repository
# - Basic Knowledge of Binary Classification
#
# ## 1.3 For Google colab users:
# If you are running this notebook on Google colab, run the following code at top of your notebook to display interactive visuals.<br/>
# <br/>
# `from pycaret.utils import enable_colab` <br/>
# `enable_colab()`
#
#
# ## 1.4 See also:
# - __[Binary Classification Tutorial (CLF102) - Intermediate Level](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Intermediate%20-%20CLF102.ipynb)__
# - __[Binary Classification Tutorial (CLF103) - Expert Level](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Expert%20-%20CLF103.ipynb)__

# + {"colab_type": "text", "id": "2DJaOwC_EQme", "cell_type": "markdown"}
# # 2.0 What is Binary Classification?
# Binary classification is a supervised machine learning technique where the goal is to predict categorical class labels which are discrete and unoredered such as Pass/Fail, Positive/Negative, Default/Not-Default etc. A few real world use cases for classification are listed below:
#
# - Medical testing to determine if a patient has a certain disease or not - the classification property is the presence of the disease.
# - A "pass or fail" test method or quality control in factories, i.e. deciding if a specification has or has not been met – a go/no-go classification.
# - Information retrieval, namely deciding whether a page or an article should be in the result set of a search or not – the classification property is the relevance of the article, or the usefulness to the user.
#
# __[Learn More about Binary Classification](https://medium.com/@categitau/in-one-of-my-previous-posts-i-introduced-machine-learning-and-talked-about-the-two-most-common-c1ac6e18df16)__

# + {"colab_type": "text", "id": "XC3kSuueEQmh", "cell_type": "markdown"}
# # 3.0 Overview of the Classification Module in PyCaret
# PyCaret's classification module (`pycaret.classification`) is a supervised machine learning module which is used for classifying the elements into a binary group based on various techniques and algorithms. Some common use cases of classification problems include predicting customer default (yes or no), customer churn (customer will leave or stay), disease found (positive or negative).
#
# The PyCaret classification module can be used for Binary or Multi-class classification problems. It has over 18 algorithms and 14 plots to analyze the performance of models. Be it hyper-parameter tuning, ensembling or advanced techniques like stacking, PyCaret's classification module has it all.

# + {"colab_type": "text", "id": "aAKRo-EbEQml", "cell_type": "markdown"}
# # 4.0 Dataset for the Tutorial

# + {"colab_type": "text", "id": "VLKxlFjrEQmq", "cell_type": "markdown"}
# For this tutorial we will use a dataset from UCI called **Default of Credit Card Clients Dataset**. This dataset contains information on default payments, demographic factors, credit data, payment history, and billing statements of credit card clients in Taiwan from April 2005 to September 2005. There are 24,000 samples and 25 features. Short descriptions of each column are as follows:
#
# - **ID:** ID of each client
# - **LIMIT_BAL:** Amount of given credit in NT dollars (includes individual and family/supplementary credit)
# - **SEX:** Gender (1=male, 2=female)
# - **EDUCATION:** (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# - **MARRIAGE:** Marital status (1=married, 2=single, 3=others)
# - **AGE:** Age in years
# - **PAY_0 to PAY_6:** Repayment status by n months ago (PAY_0 = last month ... PAY_6 = 6 months ago) (Labels: -1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
# - **BILL_AMT1 to BILL_AMT6:** Amount of bill statement by n months ago ( BILL_AMT1 = last_month .. BILL_AMT6 = 6 months ago)
# - **PAY_AMT1 to PAY_AMT6:** Amount of payment by n months ago ( BILL_AMT1 = last_month .. BILL_AMT6 = 6 months ago)
# - **default:** Default payment (1=yes, 0=no) `Target Column`
#
# #### Dataset Acknowledgement:
# Lichman, M. (2013). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
#
# The original dataset and data dictionary can be __[found here.](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)__ 

# + {"colab_type": "text", "id": "Ui_rALqYEQmv", "cell_type": "markdown"}
# # 5.0 Getting the Data

# + {"colab_type": "text", "id": "BfqIMeJNEQmz", "cell_type": "markdown"}
# You can download the data from the original source __[found here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)__ and load it using pandas __[(Learn How)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)__ or you can use PyCaret's data respository to load the data using the `get_data()` function (This will require an internet connection).

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 211}, "colab_type": "code", "id": "lUvE187JEQm3", "outputId": "8741262c-0e33-4ec0-b54d-3c8fb41e52c0"}
from pycaret.datasets import get_data
dataset = get_data('credit')

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 33}, "colab_type": "code", "id": "kMqDGBkJEQnN", "outputId": "b2015b7a-4c1a-4377-d9cf-3e9ac5ce3ea2"}
#check the shape of data
dataset.shape

# + {"colab_type": "text", "id": "LyGFryEhEQne", "cell_type": "markdown"}
# In order to demonstrate the `predict_model()` function on unseen data, a sample of 1200 records has been withheld from the original dataset to be used for predictions. This should not be confused with a train/test split as this particular split is performed to simulate a real life scenario. Another way to think about this is that these 1200 records are not available at the time when the machine learning experiment was performed.

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 50}, "colab_type": "code", "id": "hXmaL1xFEQnj", "outputId": "f1f62a7d-5d3d-4832-ee00-a4d20ee39c41"}
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# + {"colab_type": "text", "id": "y9s9wNcjEQn0", "cell_type": "markdown"}
# # 6.0 Setting up Environment in PyCaret

# + {"colab_type": "text", "id": "ZlA01j6NEQn7", "cell_type": "markdown"}
# The `setup()` function initializes the environment in pycaret and creates the transformation pipeline to prepare the data for modeling and deployment. `setup()` must be called before executing any other function in pycaret. It takes two mandatory parameters: a pandas dataframe and the name of the target column. All other parameters are optional and are used to customize the pre-processing pipeline (we will see them in later tutorials).
#
# When `setup()` is executed, PyCaret's inference algorithm will automatically infer the data types for all features based on certain properties. The data type should be inferred correctly but this is not always the case. To account for this, PyCaret displays a table containing the features and their inferred data types after `setup()` is executed. If all of the data types are correctly identified `enter` can be pressed to continue or `quit` can be typed to end the expriment. Ensuring that the data types are correct is of fundamental importance in PyCaret as it automatically performs a few pre-processing tasks which are imperative to any machine learning experiment. These tasks are performed differently for each data type which means it is very important for them to be correctly configured.
#
# In later tutorials we will learn how to overwrite PyCaret's infered data type using the `numeric_features` and `categorical_features` parameters in `setup()`.

# + {"colab": {}, "colab_type": "code", "id": "BOmRR0deEQoA"}
from pycaret.classification import *

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 803}, "colab_type": "code", "id": "k2IuvfDHEQoO", "outputId": "c7754ae9-b060-4218-b6f0-de65a815aa3a"}
exp_clf101 = setup(data = data, target = 'default', session_id=123) 

# + {"colab_type": "text", "id": "JJSOhIOxEQoY", "cell_type": "markdown"}
# Once the setup has been succesfully executed it prints the information grid which contains several important pieces of information. Most of the information is related to the pre-processing pipeline which is constructed when `setup()` is executed. The majority of these features are out of scope for the purposes of this tutorial however a few important things to note at this stage include:
#
# - **session_id :**  A pseudo-random number distributed as a seed in all functions for later reproducibility. If no `session_id` is passed, a random number is automatically generated that is distributed to all functions. In this experiment, the `session_id` is set as `123` for later reproducibility.<br/>
# <br/>
# - **Target Type :**  Binary or Multiclass. The Target type is automatically detected and shown. There is no difference in how the experiment is performed for Binary or Multiclass problems. All functionalities are identical.<br/>
# <br/>
# - **Label Encoded :**  When the Target variable is of type string (i.e. 'Yes' or 'No') instead of 1 or 0, it automatically encodes the label into 1 and 0 and displays the mapping (0 : No, 1 : Yes) for reference. In this experiment no label encoding is required since the target variable is of type numeric. <br/>
# <br/>
# - **Original Data :**  Displays the original shape of the dataset. In this experiment (22800, 24) means 22,800 samples and 24 features including the target column. <br/>
# <br/>
# - **Missing Values :**  When there are missing values in the original data this will show as True. For this experiment there are no missing values in the dataset. 
# <br/>
# <br/>
# - **Numeric Features :**  The number of features inferred as numeric. In this dataset, 14 out of 24 features are inferred as numeric. <br/>
# <br/>
# - **Categorical Features :**  The number of features inferred as categorical. In this dataset, 9 out of 24 features are inferred as categorical. <br/>
# <br/>
# - **Transformed Train Set :**  Displays the shape of the transformed training set. Notice that the original shape of (22800, 24) is transformed into (15959, 91) for the transformed train set and the number of features have increased to 91 from 24 due to categorical encoding <br/>
# <br/>
# - **Transformed Test Set :**  Displays the shape of the transformed test/hold-out set. There are 6841 samples in test/hold-out set. This split is based on the default value of 70/30 that can be changed using the `train_size` parameter in setup. <br/>
#
# Notice how a few tasks that are imperative to perform modeling are automatically handled such as missing value imputation (in this case there are no missing values in the training data, but we still need imputers for unseen data), categorical encoding etc. Most of the parameters in `setup()` are optional and used for customizing the pre-processing pipeline. These parameters are out of scope for this tutorial but as you progress to the intermediate and expert levels, we will cover them in much greater detail.

# + {"colab_type": "text", "id": "it_nJo1IEQob", "cell_type": "markdown"}
# # 7.0 Comparing All Models

# + {"colab_type": "text", "id": "apb_B9bBEQof", "cell_type": "markdown"}
# Comparing all models to evaluate performance is the recommended starting point for modeling once the setup is completed (unless you exactly know what kind of model you need, which is often not the case). This function trains all models in the model library and scores them using stratified cross validation for metric evaluation. The output prints a score grid that shows average Accuracy, AUC, Recall, Precision, F1, Kappa, and MCC accross the folds (10 by default) along with training times.

# + {"colab": {}, "colab_type": "code", "id": "AsG0b1NIEQoj", "outputId": "a6e3a510-45a1-4782-8ffe-0ec138a64eed"}
best_model = compare_models()

# + {"colab_type": "text", "id": "nZAUhQGLEQoz", "cell_type": "markdown"}
# Two simple words of code ***(not even a line)*** have trained and evaluated over 15 models using cross validation. The score grid printed above highlights the highest performing metric for comparison purposes only. The grid by default is sorted using 'Accuracy' (highest to lowest) which can be changed by passing the `sort` parameter. For example `compare_models(sort = 'Recall')` will sort the grid by Recall instead of Accuracy. If you want to change the fold parameter from the default value of `10` to a different value then you can use the `fold` parameter. For example `compare_models(fold = 5)` will compare all models on 5 fold cross validation. Reducing the number of folds will improve the training time. By default, `compare_models` return the best performing model based on default sort order but can be used to return a list of top N models by using `n_select` parameter.
# -

print(best_model)

# + {"colab_type": "text", "id": "P5m2pciOEQo4", "cell_type": "markdown"}
# # 8.0 Create a Model

# + {"colab_type": "text", "id": "u_6cIilfEQo7", "cell_type": "markdown"}
# `create_model` is the most granular function in PyCaret and is often the foundation behind most of the PyCaret functionalities. As the name suggests this function trains and evaluates a model using cross validation that can be set with `fold` parameter. The output prints a score grid that shows Accuracy, AUC, Recall, Precision, F1, Kappa and MCC by fold. 
#
# For the remaining part of this tutorial, we will work with the below models as our candidate models. The selections are for illustration purposes only and do not necessarily mean they are the top performing or ideal for this type of data.
#
# - Decision Tree Classifier ('dt')
# - K Neighbors Classifier ('knn')
# - Random Forest Classifier ('rf')
#
# There are 18 classifiers available in the model library of PyCaret. To see list of all classifiers either check the `docstring` or use `models` function to see the library.
# -

models()

# + {"colab_type": "text", "id": "UWMSeyNhEQo-", "cell_type": "markdown"}
# ### 8.1 Decision Tree Classifier

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 392}, "colab_type": "code", "id": "LP896uSIEQpD", "outputId": "d6d31562-feb5-4052-ee23-0a444fecaacf"}
dt = create_model('dt')

# + {"colab": {}, "colab_type": "code", "id": "FRat05yGEQpQ", "outputId": "c8e6a190-8bec-4646-d2c8-8a92b129c484"}
#trained model object is stored in the variable 'dt'. 
print(dt)

# + {"colab_type": "text", "id": "rWUojqBCEQpb", "cell_type": "markdown"}
# ### 8.2 K Neighbors Classifier

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 392}, "colab_type": "code", "id": "2uonD20gEQpe", "outputId": "560e3cb6-41d5-4293-b1c5-2bd1cf3bc63b"}
knn = create_model('knn')

# + {"colab_type": "text", "id": "nSg3OUjuEQpu", "cell_type": "markdown"}
# ### 8.3 Random Forest Classifier

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 392}, "colab_type": "code", "id": "FGCoUiQpEQpz", "outputId": "212cb736-6dcb-4b77-e45b-14ad895bff43"}
rf = create_model('rf')

# + {"colab_type": "text", "id": "z6F3Fk7TEQp8", "cell_type": "markdown"}
# Notice that the mean score of all models matches with the score printed in `compare_models()`. This is because the metrics printed in the `compare_models()` score grid are the average scores across all CV folds. Similar to `compare_models()`, if you want to change the fold parameter from the default value of 10 to a different value then you can use the `fold` parameter. For Example: `create_model('dt', fold = 5)` will create a Decision Tree Classifier using 5 fold stratified CV.

# + {"colab_type": "text", "id": "XvpjzbGQEQqB", "cell_type": "markdown"}
# # 9.0 Tune a Model

# + {"colab_type": "text", "id": "nc_GgksHEQqE", "cell_type": "markdown"}
# When a model is created using the `create_model()` function it uses the default hyperparameters to train the model. In order to tune hyperparameters, the `tune_model()` function is used. This function automatically tunes the hyperparameters of a model using `Random Grid Search` on a pre-defined search space. The output prints a score grid that shows Accuracy, AUC, Recall, Precision, F1, Kappa, and MCC by fold for the best model. To use the custom search grid, you can pass `custom_grid` parameter in the `tune_model` function (see 9.2 KNN tuning below). <br/>
# <br/>

# + {"colab_type": "text", "id": "BQlMCxrUEQqG", "cell_type": "markdown"}
# ### 9.1 Decision Tree Classifier

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 392}, "colab_type": "code", "id": "of46aj6vEQqJ", "outputId": "26f7f708-739a-489b-bb76-b33e0a800362"}
tuned_dt = tune_model(dt)

# + {"colab": {}, "colab_type": "code", "id": "__anDkttEQqV", "outputId": "7cf46ace-012a-4131-b8b8-370f9d4a63cb"}
#tuned model object is stored in the variable 'tuned_dt'. 
print(tuned_dt)

# + {"colab_type": "text", "id": "CD-f0delEQqq", "cell_type": "markdown"}
# ### 9.2 K Neighbors Classifier

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 392}, "colab_type": "code", "id": "xN1nYwFXEQqv", "outputId": "e4ab669d-bee0-4a9d-f5c7-2ed07ec613b9"}
import numpy as np
tuned_knn = tune_model(knn, custom_grid = {'n_neighbors' : np.arange(0,50,1)})
# -

print(tuned_knn)

# + {"colab_type": "text", "id": "KO3zIfs-EQrA", "cell_type": "markdown"}
# ### 9.3 Random Forest Classifier

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 392}, "colab_type": "code", "id": "gmaIfnBMEQrE", "outputId": "a59cebfa-f81e-477c-f83c-e9443fd80b0f"}
tuned_rf = tune_model(rf)

# + {"colab_type": "text", "id": "IqxEZRi1EQrO", "cell_type": "markdown"}
# By default, `tune_model` optimizes `Accuracy` but this can be changed using `optimize` parameter. For example: `tune_model(dt, optimize = 'AUC')` will search for the hyperparameters of a Decision Tree Classifier that results in the highest `AUC` instead of `Accuracy`. For the purposes of this example, we have used the default metric `Accuracy` only for the sake of simplicity. Generally, when the dataset is imbalanced (such as the credit dataset we are working with) `Accuracy` is not a good metric for consideration. The methodology behind selecting the right metric to evaluate a classifier is beyond the scope of this tutorial but if you would like to learn more about it, you can __[click here](https://medium.com/@MohammedS/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b)__ to read an article on how to choose the right evaluation metric.
#
# Metrics alone are not the only criteria you should consider when finalizing the best model for production. Other factors to consider include training time, standard deviation of kfolds etc. As you progress through the tutorial series we will discuss those factors in detail at the intermediate and expert levels. For now, let's move forward considering the Tuned Random Forest Classifier `tuned_rf`, as our best model for the remainder of this tutorial.

# + {"colab_type": "text", "id": "w_P46O0jEQrT", "cell_type": "markdown"}
# # 10.0 Plot a Model

# + {"colab_type": "text", "id": "FGM9GOtjEQrV", "cell_type": "markdown"}
# Before model finalization, the `plot_model()` function can be used to analyze the performance across different aspects such as AUC, confusion_matrix, decision boundary etc. This function takes a trained model object and returns a plot based on the test / hold-out set. 
#
# There are 15 different plots available, please see the `plot_model()` docstring for the list of available plots.

# + {"colab_type": "text", "id": "euqkQYJaEQrY", "cell_type": "markdown"}
# ### 10.1 AUC Plot

# + {"colab": {}, "colab_type": "code", "id": "RLbLqvkHEQra", "outputId": "fe40b5e3-6375-43e8-e97d-1d487e02eb2d"}
plot_model(tuned_rf, plot = 'auc')

# + {"colab_type": "text", "id": "bwyoTUDQEQrm", "cell_type": "markdown"}
# ### 10.2 Precision-Recall Curve

# + {"colab": {}, "colab_type": "code", "id": "4IvchQoiEQrr", "outputId": "fdff2076-86fc-42f5-beee-f0051ea30dd4"}
plot_model(tuned_rf, plot = 'pr')

# + {"colab_type": "text", "id": "_r9rwEw7EQrz", "cell_type": "markdown"}
# ### 10.3 Feature Importance Plot

# + {"colab": {}, "colab_type": "code", "id": "nVScSxJ-EQr2", "outputId": "f44f4b08-b749-4d0e-dcc9-d7e3dc6240c8"}
plot_model(tuned_rf, plot='feature')

# + {"colab_type": "text", "id": "FfWC3NEhEQr9", "cell_type": "markdown"}
# ### 10.4 Confusion Matrix

# + {"colab": {}, "colab_type": "code", "id": "OAB5mes-EQsA", "outputId": "bd82130d-2cc3-4b63-df5d-03b7aa54bf52"}
plot_model(tuned_rf, plot = 'confusion_matrix')

# + {"colab_type": "text", "id": "deClKJrbEQsJ", "cell_type": "markdown"}
# *Another* way to analyze the performance of models is to use the `evaluate_model()` function which displays a user interface for all of the available plots for a given model. It internally uses the `plot_model()` function. 

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 436, "referenced_widgets": ["42d5400d235d40b78190016ef0dabe11", "41031579127f4a53b58957e601465083", "12bf8b3c6ae8444a900474912589fdf1", "9bb3600d38c04691b444ff375ad5e3f5", "8886001bc7c1463ba58a8453f5c55073", "0a06fb091bd94ce6b6ab892e2c6faadf", "3cc1e83b91f34b289c7d52003f20a97a", "8d709ec9ec484944b1f9773748857f84", "8399e21b17634116861a5abaa9c0ccf7", "d5b6fce1763b4b54898ff3397b0f5bb0", "57b94ac505d142769b79de2f1e5c1166", "2a81017413ca4fe789c2272a5831a069", "02771b4dc3284414ab05df1906f4556b", "9e338844e75b4e17be8483529f5f38fd", "22588a12c0db4067982e62ebbe7e6930"]}, "colab_type": "code", "id": "OcLV1Ln6EQsN", "outputId": "7b5b8b4e-8d4a-4371-9a4f-cabb0a96265a"}
evaluate_model(tuned_rf)

# + {"colab_type": "text", "id": "RX5pYUJJEQsV", "cell_type": "markdown"}
# # 11.0 Predict on test / hold-out Sample

# + {"colab_type": "text", "id": "mFSvRYiaEQsd", "cell_type": "markdown"}
# Before finalizing the model, it is advisable to perform one final check by predicting the test/hold-out set and reviewing the evaluation metrics. If you look at the information grid in Section 6 above, you will see that 30% (6,841 samples) of the data has been separated out as test/hold-out sample. All of the evaluation metrics we have seen above are cross validated results based on the training set (70%) only. Now, using our final trained model stored in the `tuned_rf` variable we will predict against the hold-out sample and evaluate the metrics to see if they are materially different than the CV results.

# + {"colab": {}, "colab_type": "code", "id": "nwaZk6oTEQsi", "outputId": "d30c8533-d347-4fa6-f18e-5b2abc937bec"}
predict_model(tuned_rf);

# + {"colab_type": "text", "id": "E-fHsX2AEQsx", "cell_type": "markdown"}
# The accuracy on test/hold-out set is **`0.8116`** compared to **`0.8203`** achieved on the `tuned_rf` CV results (in section 9.3 above). This is not a significant difference. If there is a large variation between the test/hold-out and CV results, then this would normally indicate over-fitting but could also be due to several other factors and would require further investigation. In this case, we will move forward with finalizing the model and predicting on unseen data (the 5% that we had separated in the beginning and never exposed to PyCaret).
#
# (TIP : It's always good to look at the standard deviation of CV results when using `create_model()`.)

# + {"colab_type": "text", "id": "r79BGjIfEQs1", "cell_type": "markdown"}
# # 12.0 Finalize Model for Deployment

# + {"colab_type": "text", "id": "B-6xJ9kQEQs7", "cell_type": "markdown"}
# Model finalization is the last step in the experiment. A normal machine learning workflow in PyCaret starts with `setup()`, followed by comparing all models using `compare_models()` and shortlisting a few candidate models (based on the metric of interest) to perform several modeling techniques such as hyperparameter tuning, ensembling, stacking etc. This workflow will eventually lead you to the best model for use in making predictions on new and unseen data. The `finalize_model()` function fits the model onto the complete dataset including the test/hold-out sample (30% in this case). The purpose of this function is to train the model on the complete dataset before it is deployed in production.

# + {"colab": {}, "colab_type": "code", "id": "_--tO4KGEQs-"}
final_rf = finalize_model(tuned_rf)

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 147}, "colab_type": "code", "id": "U9W6kXsSEQtQ", "outputId": "794b24a4-9c95-4730-eddd-f82e4925b866"}
#Final Random Forest model parameters for deployment
print(final_rf)

# + {"colab_type": "text", "id": "kgdOjxypEQtd", "cell_type": "markdown"}
# **Caution:** One final word of caution. Once the model is finalized using `finalize_model()`, the entire dataset including the test/hold-out set is used for training. As such, if the model is used for predictions on the hold-out set after `finalize_model()` is used, the information grid printed will be misleading as you are trying to predict on the same data that was used for modeling. In order to demonstrate this point only, we will use `final_rf` under `predict_model()` to compare the information grid with the one above in section 11. 

# + {"colab": {}, "colab_type": "code", "id": "NJDk3I-EEQtg", "outputId": "4d75663a-e86f-4826-c8e4-c9aa722648df"}
predict_model(final_rf);

# + {"colab_type": "text", "id": "V77JC5JVEQtp", "cell_type": "markdown"}
# Notice how the AUC in `final_rf` has increased to **`0.7526`** from **`0.7407`**, even though the model is the same. This is because the `final_rf` variable has been trained on the complete dataset including the test/hold-out set.

# + {"colab_type": "text", "id": "hUzc6tXNEQtr", "cell_type": "markdown"}
# # 13.0 Predict on unseen data

# + {"colab_type": "text", "id": "dx5vXjChEQtt", "cell_type": "markdown"}
# The `predict_model()` function is also used to predict on the unseen dataset. The only difference from section 11 above is that this time we will pass the `data_unseen` parameter. `data_unseen` is the variable created at the beginning of the tutorial and contains 5% (1200 samples) of the original dataset which was never exposed to PyCaret. (see section 5 for explanation)

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 211}, "colab_type": "code", "id": "0y5KWLC6EQtx", "outputId": "30771f87-7847-43ce-e984-9963cff7d043"}
unseen_predictions = predict_model(final_rf, data=data_unseen)
unseen_predictions.head()

# + {"colab_type": "text", "id": "oPYmVpugEQt5", "cell_type": "markdown"}
# The `Label` and `Score` columns are added onto the `data_unseen` set. Label is the prediction and score is the probability of the prediction. Notice that predicted results are concatenated to the original dataset while all the transformations are automatically performed in the background. You can also check the metrics on this since you have actual target column `default` available. To do that we will use `pycaret.utils` module. See example below:
# -

from pycaret.utils import check_metric
check_metric(unseen_predictions['default'], unseen_predictions['Label'], metric = 'Accuracy')

# + {"colab_type": "text", "id": "L__po3sUEQt7", "cell_type": "markdown"}
# # 14.0 Saving the model

# + {"colab_type": "text", "id": "1sQPT7jrEQt-", "cell_type": "markdown"}
# We have now finished the experiment by finalizing the `tuned_rf` model which is now stored in `final_rf` variable. We have also used the model stored in `final_rf` to predict `data_unseen`. This brings us to the end of our experiment, but one question is still to be asked: What happens when you have more new data to predict? Do you have to go through the entire experiment again? The answer is no, PyCaret's inbuilt function `save_model()` allows you to save the model along with entire transformation pipeline for later use.

# + {"colab": {}, "colab_type": "code", "id": "ln1YWIXTEQuA", "outputId": "d3cb0652-f72e-44e8-9455-824b12740bff"}
save_model(final_rf,'Final RF Model 11Nov2020')

# + {"colab_type": "text", "id": "WE6f48AYEQuR", "cell_type": "markdown"}
# (TIP : It's always good to use date in the filename when saving models, it's good for version control.)

# + {"colab_type": "text", "id": "Z8OBesfkEQuU", "cell_type": "markdown"}
# # 15.0 Loading the saved model

# + {"colab_type": "text", "id": "V2K_WLaaEQuW", "cell_type": "markdown"}
# To load a saved model at a future date in the same or an alternative environment, we would use PyCaret's `load_model()` function and then easily apply the saved model on new unseen data for prediction.

# + {"colab": {}, "colab_type": "code", "id": "Siw_2EIUEQub", "outputId": "5da8b7c9-01f7-469c-f0c9-b19c8ce11bcc"}
saved_final_rf = load_model('Final RF Model 11Nov2020')

# + {"colab_type": "text", "id": "1zyi6-Q-EQuq", "cell_type": "markdown"}
# Once the model is loaded in the environment, you can simply use it to predict on any new data using the same `predict_model()` function. Below we have applied the loaded model to predict the same `data_unseen` that we used in section 13 above.

# + {"colab": {}, "colab_type": "code", "id": "HMPO1ka9EQut"}
new_prediction = predict_model(saved_final_rf, data=data_unseen)

# + {"colab": {}, "colab_type": "code", "id": "7wyDQQSzEQu8", "outputId": "23065436-42e3-4441-ed58-a8863f8971f9"}
new_prediction.head()

# + {"colab_type": "text", "id": "bf8I1uqcEQvD", "cell_type": "markdown"}
# Notice that the results of `unseen_predictions` and `new_prediction` are identical.
# -

from pycaret.utils import check_metric
check_metric(new_prediction['default'], new_prediction['Label'], metric = 'Accuracy')

# + {"colab_type": "text", "id": "_HeOs8BhEQvF", "cell_type": "markdown"}
# # 16.0 Wrap-up / Next Steps?

# + {"colab_type": "text", "id": "VqG1NnwXEQvK", "cell_type": "markdown"}
# This tutorial has covered the entire machine learning pipeline from data ingestion, pre-processing, training the model, hyperparameter tuning, prediction and saving the model for later use. We have completed all of these steps in less than 10 commands which are naturally constructed and very intuitive to remember such as `create_model()`, `tune_model()`, `compare_models()`. Re-creating the entire experiment without PyCaret would have taken well over 100 lines of code in most libraries.
#
# We have only covered the basics of `pycaret.classification`. In following tutorials we will go deeper into advanced pre-processing, ensembling, generalized stacking and other techniques that allow you to fully customize your machine learning pipeline and are must know for any data scientist.
#
# See you at the next tutorial. Follow the link to __[Binary Classification Tutorial (CLF102) - Intermediate Level](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Intermediate%20-%20CLF102.ipynb)__
