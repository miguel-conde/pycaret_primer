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

# + {"colab_type": "text", "id": "mCOJwSTxfuMP", "cell_type": "markdown"}
# #  <span style="color:orange">Multiclass Classification Tutorial (MCLF101) - Level Beginner</span>

# + {"colab_type": "text", "id": "h0Bb-8rXfuMS", "cell_type": "markdown"}
# **Created using: PyCaret 2.0** <br />
# **Date Updated: August 24, 2020**
#
# # 1.0 Tutorial Objective
# Welcome to the Multiclass Classification Tutorial **(MCLF101)** - Level Beginner. This tutorial assumes that you are new to PyCaret and looking to get started with Multiclass Classification using the `pycaret.classification` Module.
#
# In this tutorial we will learn:
#
#
# * **Getting Data:**  How to import data from PyCaret repository
# * **Setting up Environment:**  How to setup an experiment in PyCaret and get started with building multiclass models
# * **Create Model:**  How to create a model, perform stratified cross validation and evaluate classification metrics
# * **Tune Model:**  How to automatically tune the hyper-parameters of a multiclass model
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
# - Basic Knowledge of Multiclass Classification
#
# ## 1.3 For Google colab users:
# If you are running this notebook on Google colab, run the following code at top of your notebook to display interactive visuals.<br/>
# <br/>
# `from pycaret.utils import enable_colab` <br/>
# `enable_colab()`
#
# ## 1.4 See also:
# - __[Binary Classification Tutorial (CLF101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Beginner%20-%20%20CLF101.ipynb)__
# - __[Binary Classification Tutorial (CLF102) - Level Intermediate](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Intermediate%20-%20CLF102.ipynb)__

# + {"colab_type": "text", "id": "VDM6TNTrfuMV", "cell_type": "markdown"}
# # 2.0 What is Multiclass Classification?
# Multiclass classification is a supervised machine learning technique where the goal is to classify instances into one of three or more classes. (Classifying instances into one of two classes is called Binary Classification). Multiclass classification should not be confused with multi-label classification, where multiple labels are to be predicted for each instance.
#
# __[Learn More about Multiclass Classification](https://towardsdatascience.com/machine-learning-multiclass-classification-with-imbalanced-data-set-29f6a177c1a)__

# + {"colab_type": "text", "id": "FEybtNqwfuMY", "cell_type": "markdown"}
# # 3.0 Overview of the Classification Module in PyCaret
# PyCaret's classification module (`pycaret.classification`) is a supervised machine learning module which is used for classifying the elements into binary or multinomial groups based on various techniques and algorithms. 
#
# The PyCaret classification module can be used for Binary or Multi-class classification problems. It has over 18 algorithms and 14 plots to analyze the performance of models. Be it hyper-parameter tuning, ensembling or advanced techniques like stacking, PyCaret's classification module has it all.

# + {"colab_type": "text", "id": "x2KiOHinfuMa", "cell_type": "markdown"}
# # 4.0 Dataset for the Tutorial

# + {"colab_type": "text", "id": "B0re-c4CfuMe", "cell_type": "markdown"}
# For this tutorial we will use the **Iris Dataset** from UCI. This is perhaps the best known database to be found in the pattern recognition literature. The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. Short descriptions of each column are as follows:
#
# - **sepal_length:** Length of Sepal
# - **sepal_width:** Width of Sepal
# - **petal_length:** Length of Petal
# - **petal_width:** Width of Petal
# - **species:** One of three class (Setosa, Versicolour, Virginica) `Target Column`
#
# #### Dataset Acknowledgement:
# Creator: R.A. Fisher, Donor : Michael Marshall (MARSHALL%PLU '@' io.arc.nasa.gov)
#
# The original dataset and data dictionary can be __[found here.](https://archive.ics.uci.edu/ml/datasets/Iris)__ 

# + {"colab_type": "text", "id": "e-NQVSpIfuMi", "cell_type": "markdown"}
# # 5.0 Getting the Data

# + {"colab_type": "text", "id": "LGV6PMCtfuMl", "cell_type": "markdown"}
# You can download the data from the original source __[found here](https://archive.ics.uci.edu/ml/datasets/Iris)__ and load it using pandas __[(Learn How)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)__ or you can use PyCaret's data respository to load the data using the `get_data()` function (This will require an internet connection).

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 204}, "colab_type": "code", "id": "wEki9ZbCfuMm", "outputId": "edcba15b-233e-476c-d119-f9a07863f0df"}
from pycaret.datasets import get_data
dataset = get_data('iris')

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 34}, "colab_type": "code", "id": "mNFEXhMsfuMs", "outputId": "f5705470-be5f-429d-faea-6caa1f23ee1e"}
#check the shape of data
dataset.shape

# + {"colab_type": "text", "id": "o4C_ep4RfuMx", "cell_type": "markdown"}
# In order to demonstrate the `predict_model()` function on unseen data, a sample of 15 records has been withheld from the original dataset to be used for predictions. This should not be confused with a train/test split as this particular split is performed to simulate a real life scenario. Another way to think about this is that these 15 records were not available at the time when the machine learning experiment was performed.

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 51}, "colab_type": "code", "id": "Ttyujuo7fuMy", "outputId": "61f75ccf-541b-4efa-d00a-e0a95bf4017e"}
data = dataset.sample(frac=0.9, random_state=786)
data_unseen = dataset.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# + {"colab_type": "text", "id": "QXG8qOz5fuM3", "cell_type": "markdown"}
# # 6.0 Setting up the Environment in PyCaret

# + {"colab_type": "text", "id": "nhFahBmtfuM5", "cell_type": "markdown"}
# The `setup()` function initializes the environment in pycaret and creates the transformation pipeline to prepare the data for modeling and deployment. `setup()` must be called before executing any other function in pycaret. It takes two mandatory parameters: a pandas dataframe and the name of the target column. All other parameters are optional and are used to customize the pre-processing pipeline (we will see them in later tutorials).
#
# When `setup()` is executed, PyCaret's inference algorithm will automatically infer the data types for all features based on certain properties. The data type should be inferred correctly but this is not always the case. To account for this, PyCaret displays a table containing the features and their inferred data types after `setup()` is executed. If all of the data types are correctly identified `enter` can be pressed to continue or `quit` can be typed to end the expriment. Ensuring that the data types are correct is of fundamental importance in PyCaret as it automatically performs a few pre-processing tasks which are imperative to any machine learning experiment. These tasks are performed differently for each data type which means it is very important for them to be correctly configured.
#
# In later tutorials we will learn how to overwrite PyCaret's infered data type using the `numeric_features` and `categorical_features` parameters in `setup()`.

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 956}, "colab_type": "code", "id": "9PcuxLQSfuNC", "outputId": "68abc9ac-a0d2-4b33-f524-9cabd0256fcc"}
from pycaret.classification import *
exp_mclf101 = setup(data = data, target = 'species', session_id=123) 

# + {"colab_type": "text", "id": "8Mz7JeKqfuNJ", "cell_type": "markdown"}
# Once the setup has been succesfully executed it prints the information grid which contains several important pieces of information. Most of the information is related to the pre-processing pipeline which is constructed when `setup()` is executed. The majority of these features are out of scope for the purposes of this tutorial however a few important things to note at this stage include:
#
# - **session_id :**  A pseduo-random number distributed as a seed in all functions for later reproducibility. If no `session_id` is passed, a random number is automatically generated that is distributed to all functions. In this experiment, the `session_id` is set as `123` for later reproducibility.<br/>
# <br/>
# - **Target Type :**  Binary or Multiclass. The Target type is automatically detected and shown. There is no difference in how the experiment is performed for Binary or Multiclass problems. All functionalities are identical.<br/>
# <br/>
# - **Label Encoded :**  When the Target variable is of type string (i.e. 'Yes' or 'No') instead of 1 or 0, it automatically encodes the label into 1 and 0 and displays the mapping (0 : No, 1 : Yes) for reference. In this experiment label encoding is applied as follows: Iris-setosa: 0, Iris-versicolor: 1, Iris-virginica: 2. <br/>
# <br/>
# - **Original Data :**  Displays the original shape of the dataset. In this experiment (135, 5) means 135 samples and 5 features including the target column. <br/>
# <br/>
# - **Missing Values :**  When there are missing values in the original data this will show as True. For this experiment there are no missing values in the dataset.<br/>
# <br/>
# - **Numeric Features :**  The number of features inferred as numeric. In this dataset, 4 out of 5 features are inferred as numeric. <br/>
# <br/>
# - **Categorical Features :**  The number of features inferred as categorical. In this dataset, there are no categorical features. <br/>
# <br/>
# - **Transformed Train Set :**  Displays the shape of the transformed training set. Notice that the original shape of (135, 5) is transformed into (94, 4) for the transformed train set. <br/>
# <br/>
# - **Transformed Test Set :**  Displays the shape of the transformed test/hold-out set. There are 41 samples in test/hold-out set. This split is based on the default value of 70/30 that can be changed using the `train_size` parameter in setup. <br/>
#
# Notice how a few tasks that are imperative to perform modeling are automatically handled such as missing value imputation, categorical encoding etc. Most of the parameters in `setup()` are optional and used for customizing the pre-processing pipeline. These parameters are out of scope for this tutorial but as you progress to the intermediate and expert levels, we will cover them in much greater detail.

# + {"colab_type": "text", "id": "i9ZvVSDYfuNL", "cell_type": "markdown"}
# # 7.0 Comparing All Models

# + {"colab_type": "text", "id": "Kr65WJW2fuNM", "cell_type": "markdown"}
# Comparing all models to evaluate performance is the recommended starting point for modeling once the setup is completed (unless you exactly know what kind of model you need, which is often not the case). This function trains all models in the model library and scores them using stratified cross validation for metric evaluation. The output prints a score grid that shows average Accuracy, Recall, Precision, F1, Kappa, and MCC accross the folds (10 by default) along with training times.

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 355}, "colab_type": "code", "id": "1wcMisYyfuNN", "outputId": "d3e16fbb-c33b-4bda-df3f-de2507f821c8"}
best = compare_models()

# + {"colab_type": "text", "id": "JxcCbKtffuNV", "cell_type": "markdown"}
# Two simple words of code ***(not even a line)*** have trained and evaluated over 15 models using cross validation. The score grid printed above highlights the highest performing metric for comparison purposes only. The grid by default is sorted using 'Accuracy' (highest to lowest) which can be changed by passing the `sort` parameter. For example `compare_models(sort = 'Recall')` will sort the grid by Recall instead of `Accuracy`. If you want to change the fold parameter from the default value of `10` to a different value then you can use the `fold` parameter. For example `compare_models(fold = 5)` will compare all models on 5 fold cross validation. Reducing the number of folds will improve the training time. By default, `compare_models` return the best performing model based on default sort order but can be used to return a list of top N models by using `n_select` parameter.
#
# **Note:** The `AUC` metric is not available for Multiclass classification however the column will still be shown with `zero` values to maintain consistency between the Binary Classification and Multiclass Classification display grids. 

# + {"colab_type": "text", "id": "qH7AeJqjfuNX", "cell_type": "markdown"}
# # 8.0 Create a Model

# + {"colab_type": "text", "id": "UM0qn2f_fuNY", "cell_type": "markdown"}
# `create_model` is the most granular function in PyCaret and is often the foundation behind most of the PyCaret functionalities. As the name suggests this function trains and evaluates a model using cross validation that can be set with `fold` parameter. The output prints a score grid that shows Accuracy, Recall, Precision, F1, Kappa and MCC by fold. 
#
# For the remaining part of this tutorial, we will work with the below models as our candidate models. The selections are for illustration purposes only and do not necessarily mean they are the top performing or ideal for this type of data.
#
# - Decision Tree Classifier ('dt')
# - K Neighbors Classifier ('knn')
# - Logistic Regression ('lr')
#
# There are 18 classifiers available in the model library of PyCaret. Please view the `create_model()` docstring for the list of all available models.

# + {"colab_type": "text", "id": "yyApyoXbfuNZ", "cell_type": "markdown"}
# ### 8.1 Decision Tree Classifier

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 421}, "colab_type": "code", "id": "U6v4R3MJfuNb", "outputId": "da32db85-d927-48b1-b6d8-1744331ec306"}
dt = create_model('dt')

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 272}, "colab_type": "code", "id": "7jwPc7ZBfuNg", "outputId": "89c7ee4c-55ed-4764-fd03-0974c22f6b1f"}
#trained model object is stored in the variable 'dt'. 
print(dt)

# + {"colab_type": "text", "id": "PhHDTJYwfuNl", "cell_type": "markdown"}
# ### 8.2 K Neighbors Classifier

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 421}, "colab_type": "code", "id": "WYHQ1DxkfuNm", "outputId": "26077e3b-a77b-4682-a518-a125e1f6c85a"}
knn = create_model('knn')

# + {"colab_type": "text", "id": "8IA_BtchfuNt", "cell_type": "markdown"}
# ### 8.3 Logistic Regression

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 421}, "colab_type": "code", "id": "u3nAFk1MfuNv", "outputId": "a863ebc8-ead4-4edc-ce9d-d91653f4fbd6"}
lr = create_model('lr')

# + {"colab_type": "text", "id": "HRtpGzhvfuN1", "cell_type": "markdown"}
# Notice that the Mean score of all models matches with the score printed in `compare_models()`. This is because the metrics printed in the `compare_models()` score grid are the average scores across all CV folds. Similar to `compare_models()`, if you want to change the fold parameter from the default value of 10 to a different value then you can use the `fold` parameter. For Example: `create_model('dt', fold = 5)` will create a Decision Tree Classifier using 5 fold stratified CV.

# + {"colab_type": "text", "id": "jzY5pn-OfuN4", "cell_type": "markdown"}
# # 9.0 Tune a Model

# + {"colab_type": "text", "id": "FShSLa10fuN5", "cell_type": "markdown"}
# When a model is created using the `create_model()` function it uses the default hyperparameters to train the model. In order to tune hyperparameters, the `tune_model()` function is used. This function automatically tunes the hyperparameters of a model using `Random Grid Search` on a pre-defined search space. The output prints a score grid that shows Accuracy, AUC, Recall, Precision, F1, Kappa, and MCC by fold for the best model. To use the custom search grid, you can pass `custom_grid` parameter in the `tune_model` function (see 9.2 KNN tuning below). <br/>
# <br/>

# + {"colab_type": "text", "id": "WVgKXyjdfuN7", "cell_type": "markdown"}
# ### 9.1 Decision Tree Classifier

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 421}, "colab_type": "code", "id": "sQqeuiqLfuN9", "outputId": "2ae99738-689c-4506-da89-2a5710b2b40b"}
tuned_dt = tune_model(dt)

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 272}, "colab_type": "code", "id": "Wgxxg07CfuOD", "outputId": "78e965df-7c6a-4548-b2ec-57a051df89bd"}
#tuned model object is stored in the variable 'tuned_dt'. 
print(tuned_dt)

# + {"colab_type": "text", "id": "fe9QKCZ8fuOU", "cell_type": "markdown"}
# ### 9.2 K Neighbors Classifier
# -

import numpy as np
tuned_knn = tune_model(knn, custom_grid = {'n_neighbors' : np.arange(0,50,1)})

# + {"colab_type": "text", "id": "oF2hDwoifuOd", "cell_type": "markdown"}
# ### 9.3 Logistic Regression

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 421}, "colab_type": "code", "id": "3wyDGn_NfuOf", "outputId": "de306431-f9f6-4a25-cefb-03201aa77b96"}
tuned_lr = tune_model(lr)

# + {"colab_type": "text", "id": "BVdIeXY5fuOi", "cell_type": "markdown"}
# The `tune_model()` function is a random grid search of hyperparameters over a pre-defined search space. By default, it is set to optimize `Accuracy` but this can be changed using the `optimize` parameter. For example: `tune_model(dt, optimize = 'Recall')` will search for the hyperparameters of a Decision Tree Classifier that result in the highest `Recall`. For the purposes of this example, we have used the default metric `Accuracy` for the sake of simplicity only. The methodology behind selecting the right metric to evaluate a classifier is beyond the scope of this tutorial but if you would like to learn more about it, you can __[click here](https://medium.com/@MohammedS/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b)__ to read an article on how to choose the right evaluation metric.
#
# Metrics alone are not the only criteria you should consider when finalizing the best model for production. Other factors to consider include training time, standard deviation of kfolds etc. As you progress through the tutorial series we will discuss those factors in detail at the intermediate and expert levels. For now, let's move forward considering the Tuned K Neighbors Classifier as our best model for the remainder of this tutorial.

# + {"colab_type": "text", "id": "3zct2zSYfuOl", "cell_type": "markdown"}
# # 10.0 Plot a Model

# + {"colab_type": "text", "id": "a3_m4yEdfuOm", "cell_type": "markdown"}
# Before model finalization, the `plot_model()` function can be used to analyze the performance across different aspects such as AUC, confusion_matrix, decision boundary etc. This function takes a trained model object and returns a plot based on the test / hold-out set. 
#
# There are 15 different plots available, please see the `plot_model()` docstring for the list of available plots.

# + {"colab_type": "text", "id": "AegNi_BXfuOn", "cell_type": "markdown"}
# ### 10.1 Confusion Matrix

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 374}, "colab_type": "code", "id": "kI0tERkNfuOo", "outputId": "d13e1e50-aa8c-420e-ba85-3affca7b9500"}
plot_model(tuned_knn, plot = 'confusion_matrix')

# + {"colab_type": "text", "id": "WdQqEtO4fuOt", "cell_type": "markdown"}
# ### 10.2 Classification Report

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 401}, "colab_type": "code", "id": "wHKBMDaZfuOu", "outputId": "2a627204-ff7b-4a2b-f307-1d5707c159f5"}
plot_model(tuned_knn, plot = 'class_report')

# + {"colab_type": "text", "id": "iidIhDUufuOx", "cell_type": "markdown"}
# ### 10.3 Decision Boundary Plot

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 361}, "colab_type": "code", "id": "9_2R_acYfuOz", "outputId": "aedd0637-b29e-49e5-999a-a255be6bd313"}
plot_model(tuned_knn, plot='boundary')

# + {"colab_type": "text", "id": "JY8z7B8MfuO2", "cell_type": "markdown"}
# ### 10.4 Prediction Error Plot

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 401}, "colab_type": "code", "id": "AtSP2JAXfuO3", "outputId": "e3230ae1-7d7e-4270-9935-f2dd54295dd1"}
plot_model(tuned_knn, plot = 'error')

# + {"colab_type": "text", "id": "hxGSKMKZfuO7", "cell_type": "markdown"}
# *Another* way to analyze the performance of models is to use the `evaluate_model()` function which displays a user interface for all of the available plots for a given model. It internally uses the `plot_model()` function. 

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 421, "referenced_widgets": ["9cc0efb4189e48c6beb4fe1c0c4a2fdc", "a63ed06ee88c4ca3883ac38e665bddf8", "b046e2e525ca425a80aac86d7b665854", "36a827017abf468d939c97492b5c73ab", "17f607299eaf4b05a1542960ba1f8953", "e4c7bdc3be744945b405b9905048f64f", "bdb566b147ef4202b351fd1014883ce6", "0a505068ebbf4d068aa7d75f702ac715", "094d443554a84f2788d4802512cda617", "4c1ee7022483425d9541b0e49f1efcd2", "17810a03d1424e729b70ec2044e9e7f6", "9d7425bc499c4f57a680a7166389e5dc", "08ddc7a42f8d492faa945a5d7eac4a86", "0a1412c6e3f6498ca9d221f259716943", "6e43059d835f4d62aa1e5c5bf7d676e1", "c20470f03116445ab707f5857592d046", "52e6b5978ec94fdabdc9adf9286c3078", "d0366972e1664000b72674d0a285a785", "66b421a890bc462b97e5d9d23fe5809a", "904ec053638f498aa1d0e4944145f736", "d8e230198678421ea3afef9ca721090d", "a1a8e8de38404b83bec2300554e42aa8", "7448bcccbfaa432c8f1f9936a51ccf3c", "eae1b12ae33041e9a14d7372fd0d6be3", "2d502879922046aaba1d84769e93ff1b", "0dc3288edbff469189f52ebf88863853", "d360dbb5ee734c839b8a30238d824633", "f0c4a07ca3cf4460bed061f0609b9b2b", "57e3592dda974775b2fc52ee73ecb50c", "cc64cb2f6f8148d7ad9ce9d9cd16dd5f", "3dd8046812e942bf9e2ccc62ba4bf2a5", "a4291bb80392498ba074a033b6c60d86", "2c22eeecfa054ab59f5a2bc6ea4e7244", "46eb9473b4c64c59956eae387e60555a", "eecbd2ce9de64166bcfddadaef395d65", "4e0a64915ccd490aae253c73dbf91a99"]}, "colab_type": "code", "id": "K5pXw5FFfuO8", "outputId": "3efd9151-1313-4d83-ccbb-877e088c44bc"}
evaluate_model(tuned_knn)

# + {"colab_type": "text", "id": "Wo-Ob7AffuO_", "cell_type": "markdown"}
# # 11.0 Predict on test / hold-out Sample

# + {"colab_type": "text", "id": "Bqznw_2efuPB", "cell_type": "markdown"}
# Before finalizing the model, it is advisable to perform one final check by predicting the test/hold-out set and reviewing the evaluation metrics. If you look at the information grid in Section 6 above, you will see that 30% (41 samples) of the data has been separated out as a test/hold-out sample. All of the evaluation metrics we have seen above are cross validated results based on the training set (70%) only. Now, using our final trained model stored in the `tuned_knn` variable we will predict against the hold-out sample and evaluate the metrics to see if they are materially different than the CV results.

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 80}, "colab_type": "code", "id": "8handW2hfuPD", "outputId": "b654968b-faec-4ce4-ff3b-338e263c2ceb"}
predict_model(tuned_knn);

# + {"colab_type": "text", "id": "XIM4IVWGfuPL", "cell_type": "markdown"}
# The accuracy on the test/hold-out set is **`0.9512`** compared to **`0.9356`** achieved on the `tuned_knn` CV results (in section 9.2 above). This is not a significant difference. If there is a large variation between the test/hold-out and CV results, then this would normally indicate over-fitting but could also be due to several other factors and would require further investigation. In this case, we will move forward with finalizing the model and predicting on unseen data (the 10% that we had separated in the beginning and never exposed to PyCaret).
#
# (TIP : It's always good to look at the standard deviation of CV results when using `create_model()`.)

# + {"colab_type": "text", "id": "HaVz-bOIfuPN", "cell_type": "markdown"}
# # 12.0 Finalize Model for Deployment

# + {"colab_type": "text", "id": "7pRFqxXVfuPO", "cell_type": "markdown"}
# Model finalization is the last step in the experiment. A normal machine learning workflow in PyCaret starts with `setup()`, followed by comparing all models using `compare_models()` and shortlisting a few candidate models (based on the metric of interest) to perform several modeling techniques such as hyperparameter tuning, ensembling, stacking etc. This workflow will eventually lead you to the best model for use in making predictions on new and unseen data. The `finalize_model()` function fits the model onto the complete dataset including the test/hold-out sample (30% in this case). The purpose of this function is to train the model on the complete dataset before it is deployed in production.

# + {"colab": {}, "colab_type": "code", "id": "oIOJaUkEfuPP"}
final_knn = finalize_model(tuned_knn)

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 136}, "colab_type": "code", "id": "tyo7534XfuPV", "outputId": "c4562eb1-07ac-4c3b-b56d-4f7ae25d929a"}
#Final K Nearest Neighbour parameters for deployment
print(final_knn)

# + {"colab_type": "text", "id": "da2mWvRefuPk", "cell_type": "markdown"}
# # 13.0 Predict on unseen data

# + {"colab_type": "text", "id": "BAAHbcnVfuPl", "cell_type": "markdown"}
# The `predict_model()` function is also used to predict on the unseen dataset. The only difference from section 11 above is that this time we will pass the `data_unseen` parameter. `data_unseen` is the variable created at the beginning of the tutorial and contains 10% (15 samples) of the original dataset which was never exposed to PyCaret. (see section 5 for explanation)

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 204}, "colab_type": "code", "id": "rvL8lBlAfuPn", "outputId": "a6d8a475-6c99-4baf-a900-ac8d7c6468b0"}
unseen_predictions = predict_model(final_knn, data=data_unseen)
unseen_predictions.head()

# + {"colab_type": "text", "id": "LKi7aVcJfuPr", "cell_type": "markdown"}
# The `Label` and `Score` columns are added onto the `data_unseen` set. Label is the prediction and score is the probability of the prediction. Notice that predicted results are concatenated to the original dataset while all the transformations are automatically performed in the background.

# + {"colab_type": "text", "id": "uAhEcJnqfuPs", "cell_type": "markdown"}
# # 14.0 Saving the model

# + {"colab_type": "text", "id": "vENiYvosfuPt", "cell_type": "markdown"}
# We have now finished the experiment by finalizing the `tuned_knn` model which is now stored in the `final_knn` variable. We have also used the model stored in `final_knn` to predict `data_unseen`. This brings us to the end of our experiment, but one question is still to be asked: What happens when you have more new data to predict? Do you have to go through the entire experiment again? The answer is no, PyCaret's inbuilt function `save_model()` allows you to save the model along with entire transformation pipeline for later use.

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 34}, "colab_type": "code", "id": "2UL0UC7UfuPu", "outputId": "003611da-e34b-4de1-d251-94e633323ee7"}
save_model(final_knn,'Final KNN Model 08Feb2020')

# + {"colab_type": "text", "id": "Gt94CUaXfuPz", "cell_type": "markdown"}
# (TIP : It's always good to use date in the filename when saving models, it's good for version control.)

# + {"colab_type": "text", "id": "uVFI4If3fuP1", "cell_type": "markdown"}
# # 15.0 Loading the saved model

# + {"colab_type": "text", "id": "7eHcdfrRfuP3", "cell_type": "markdown"}
# To load a saved model at a future date in the same or an alternative environment, we would use PyCaret's `load_model()` function and then easily apply the saved model on new unseen data for prediction.

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 34}, "colab_type": "code", "id": "8ThDTvUifuP6", "outputId": "c599c153-1079-477c-a8d7-66cc846fc0c2"}
saved_final_knn = load_model('Final KNN Model 08Feb2020')

# + {"colab_type": "text", "id": "7bo2xBltfuQC", "cell_type": "markdown"}
# Once the model is loaded in the environment, you can simply use it to predict on any new data using the same `predict_model()` function. Below we have applied the loaded model to predict the same `data_unseen` that we used in section 13 above.

# + {"colab": {}, "colab_type": "code", "id": "bcPp5zcjfuQD"}
new_prediction = predict_model(saved_final_knn, data=data_unseen)

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 204}, "colab_type": "code", "id": "NYhNg8RyfuQM", "outputId": "05ff52a7-a98a-4d45-f2b7-75d3186785ce"}
new_prediction.head()

# + {"colab_type": "text", "id": "Kp_XadZRfuQQ", "cell_type": "markdown"}
# Notice that the results of `unseen_predictions` and `new_prediction` are identical.

# + {"colab_type": "text", "id": "9_gIC-mSfuQR", "cell_type": "markdown"}
# # 16.0 Wrap-up / Next Steps?

# + {"colab_type": "text", "id": "Rn_0E7H0fuQU", "cell_type": "markdown"}
# This tutorial has covered the entire machine learning pipeline from data ingestion, pre-processing, training the model, hyperparameter tuning, prediction and saving the model for later use. We have completed all of these steps in less than 10 commands which are naturally constructed and very intuitive to remember such as `create_model()`, `tune_model()`, `compare_models()`. Re-creating the entire experiment without PyCaret would have taken well over 100 lines of code in most libraries.
#
# We have only covered the basics of `pycaret.classification`. In following tutorials we will go deeper into advanced pre-processing, ensembling, generalized stacking and other techniques that allow you to fully customize your machine learning pipeline and are must know for any data scientist. 
