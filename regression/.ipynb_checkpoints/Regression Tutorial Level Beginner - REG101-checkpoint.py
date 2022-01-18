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

# + {"colab_type": "text", "id": "SAd865lNzZpT", "cell_type": "markdown"}
# #  <span style="color:orange">Regression Tutorial (REG101) - Level Beginner</span>

# + {"colab_type": "text", "id": "yXN8UznszZpc", "cell_type": "markdown"}
# **Created using: PyCaret 2.2** <br />
# **Date Updated: November 25, 2020**
#
# # 1.0  Tutorial Objective
# Welcome to Regression Tutorial (REG101) - Level Beginner. This tutorial assumes that you are new to PyCaret and looking to get started with Regression using the `pycaret.regression` Module.
#
# In this tutorial we will learn:
#
#
# * **Getting Data:**  How to import data from PyCaret repository
# * **Setting up Environment:**  How to setup an experiment in PyCaret and get started with building regression models
# * **Create Model:**  How to create a model, perform cross validation and evaluate regression metrics
# * **Tune Model:**  How to automatically tune the hyperparameters of a regression model
# * **Plot Model:**  How to analyze model performance using various plots
# * **Finalize Model:** How to finalize the best model at the end of the experiment
# * **Predict Model:**  How to make prediction on new / unseen data
# * **Save / Load Model:**  How to save / load a model for future use
#
# Read Time : Approx. 30 Minutes
#
#
# ## 1.1 Installing PyCaret
# The first step to get started with PyCaret is to install PyCaret. Installation is easy and will only take a few minutes. Follow the instructions below:
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
# - Basic Knowledge of Regression
#
# ## 1.3 For Google Colab Users:
# If you are running this notebook on Google colab, run the following code at top of your notebook to display interactive visuals.<br/>
# <br/>
# `from pycaret.utils import enable_colab` <br/>
# `enable_colab()`
#
# ## 1.4 See also:
# - __[Regression Tutorial (REG102) - Level Intermediate](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Intermediate%20-%20REG102.ipynb)__
# - __[Regression Tutorial (REG103) - Level Expert](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Expert%20-%20REG103.ipynb)__

# + {"colab_type": "text", "id": "HuEUiXXhzZpi", "cell_type": "markdown"}
# # 2.0 What is Regression?
#
# Regression analysis is a set of statistical processes for estimating the relationships between a dependent variable (often called the 'outcome variable', or 'target') and one or more independent variables (often called 'features', 'predictors', or 'covariates'). The objective of regression in machine learning is to predict continuous values such as sales amount, quantity, temperature etc.
#
# __[Learn More about Regression](https://hbr.org/2015/11/a-refresher-on-regression-analysis)__

# + {"colab_type": "text", "id": "xnEk7n5ZzZpm", "cell_type": "markdown"}
# # 3.0 Overview of the Regression Module in PyCaret
# PyCaret's Regression module (`pycaret.regression`) is a supervised machine learning module which is used for predicting continuous values / outcomes using various techniques and algorithms. Regression can be used for predicting values / outcomes such as sales, units sold, temperature or any number which is continuous.
#
# PyCaret's regression module has over 25 algorithms and 10 plots to analyze the performance of models. Be it hyper-parameter tuning, ensembling or advanced techniques like stacking, PyCaret's regression module has it all.

# + {"colab_type": "text", "id": "uN95Uqo6zZpq", "cell_type": "markdown"}
# # 4.0 Dataset for the Tutorial

# + {"colab_type": "text", "id": "Guj8GFIJzZpu", "cell_type": "markdown"}
# For this tutorial we will use a dataset based on a case study called **"Sarah Gets a Diamond"**. This case was presented in the first year decision analysis course at Darden School of Business (University of Virginia). The basis for the data is a case regarding a hopeless romantic MBA student choosing the right diamond for his bride-to-be, Sarah. The data contains 6000 records for training. Short descriptions of each column are as follows:
#
# - **ID:** Uniquely identifies each observation (diamond)
# - **Carat Weight:** The weight of the diamond in metric carats. One carat is equal to 0.2 grams, roughly the same weight as a paperclip
# - **Cut:** One of five values indicating the cut of the diamond in the following order of desirability (Signature-Ideal, Ideal, Very Good, Good, Fair)
# - **Color:** One of six values indicating the diamond's color in the following order of desirability (D, E, F - Colorless, G, H, I - Near colorless)
# - **Clarity:** One of seven values indicating the diamond's clarity in the following order of desirability (F - Flawless, IF - Internally Flawless, VVS1 or VVS2 - Very, Very Slightly Included, or VS1 or VS2 - Very Slightly Included, SI1 - Slightly Included)
# - **Polish:** One of four values indicating the diamond's polish (ID - Ideal, EX - Excellent, VG - Very Good, G - Good)
# - **Symmetry:** One of four values indicating the diamond's symmetry (ID - Ideal, EX - Excellent, VG - Very Good, G - Good)
# - **Report:** One of of two values "AGSL" or "GIA" indicating which grading agency reported the qualities of the diamond qualities
# - **Price:** The amount in USD that the diamond is valued `Target Column`
#
#
# #### Dataset Acknowledgement:
# This case was prepared by Greg Mills (MBA ’07) under the supervision of Phillip E. Pfeifer, Alumni Research Professor of Business Administration. Copyright (c) 2007 by the University of Virginia Darden School Foundation, Charlottesville, VA. All rights reserved.
#
# The original dataset and description can be __[found here.](https://github.com/DardenDSC/sarah-gets-a-diamond)__ 

# + {"colab_type": "text", "id": "wwUzzm1YzZpz", "cell_type": "markdown"}
# # 5.0 Getting the Data

# + {"colab_type": "text", "id": "PFCSZ_NKzZp3", "cell_type": "markdown"}
# You can download the data from the original source __[found here](https://github.com/DardenDSC/sarah-gets-a-diamond)__ and load it using pandas __[(Learn How)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)__ or you can use PyCaret's data respository to load the data using the `get_data()` function (This will require internet connection).

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 191}, "colab_type": "code", "id": "H6qS5U--zZp7", "outputId": "2a11a81c-7e67-425a-a3ef-091d2c9fbd30"}
from pycaret.datasets import get_data
dataset = get_data('diamond')

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 33}, "colab_type": "code", "id": "D5PerU66zZqK", "outputId": "2fdd6ab8-7d68-4cc4-81a7-0cb82ed70799"}
#check the shape of data
dataset.shape

# + {"colab_type": "text", "id": "7eWmeLvYzZqY", "cell_type": "markdown"}
# In order to demonstrate the `predict_model()` function on unseen data, a sample of 600 records has been withheld from the original dataset to be used for predictions. This should not be confused with a train/test split as this particular split is performed to simulate a real life scenario. Another way to think about this is that these 600 records are not available at the time when the machine learning experiment was performed.

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 50}, "colab_type": "code", "id": "R4K9F7BXzZqc", "outputId": "22b1c4e7-a1e1-48d2-8ddc-907e716d5b53"}
data = dataset.sample(frac=0.9, random_state=786)
data_unseen = dataset.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# + {"colab_type": "text", "id": "DxnJV14BzZqq", "cell_type": "markdown"}
# # 6.0 Setting up Environment in PyCaret

# + {"colab_type": "text", "id": "15-blMPOzZqw", "cell_type": "markdown"}
# The `setup()` function initializes the environment in pycaret and creates the transformation pipeline to prepare the data for modeling and deployment. `setup()` must be called before executing any other function in pycaret. It takes two mandatory parameters: a pandas dataframe and the name of the target column. All other parameters are optional and are used to customize the pre-processing pipeline (we will see them in later tutorials).
#
# When `setup()` is executed, PyCaret's inference algorithm will automatically infer the data types for all features based on certain properties. The data type should be inferred correctly but this is not always the case. To account for this, PyCaret displays a table containing the features and their inferred data types after `setup()` is executed. If all of the data types are correctly identified `enter` can be pressed to continue or `quit` can be typed to end the expriment. Ensuring that the data types are correct is of fundamental importance in PyCaret as it automatically performs a few pre-processing tasks which are imperative to any machine learning experiment. These tasks are performed differently for each data type which means it is very important for them to be correctly configured.
#
# In later tutorials we will learn how to overwrite PyCaret's inferred data type using the `numeric_features` and `categorical_features` parameters in `setup()`.

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 803}, "colab_type": "code", "id": "7V2FN4KQzZrA", "outputId": "43d8d23d-ef08-438a-8cc3-ba78e9773aca"}
from pycaret.regression import *
exp_reg101 = setup(data = data, target = 'Price', session_id=123) 

# + {"colab_type": "text", "id": "nWBypX32zZrP", "cell_type": "markdown"}
# Once the setup has been succesfully executed it prints the information grid which contains several important pieces of information. Most of the information is related to the pre-processing pipeline which is constructed when `setup()` is executed. The majority of these features are out of scope for the purposes of this tutorial. However, a few important things to note at this stage include:
#
# - **session_id :**  A pseudo-random number distributed as a seed in all functions for later reproducibility. If no `session_id` is passed, a random number is automatically generated that is distributed to all functions. In this experiment, the `session_id` is set as `123` for later reproducibility.<br/>
# <br/>
# - **Original Data :**  Displays the original shape of dataset. In this experiment (5400, 8) means 5400 samples and 8 features including the target column. <br/>
# <br/>
# - **Missing Values :**  When there are missing values in the original data, this will show as True. For this experiment there are no missing values in the dataset.<br/>
# <br/>
# - **Numeric Features :**  Number of features inferred as numeric. In this dataset, 1 out of 8 features are inferred as numeric. <br/>
# <br/>
# - **Categorical Features :**  Number of features inferred as categorical. In this dataset, 6 out of 8 features are inferred as categorical. <br/>
# <br/>
# - **Transformed Train Set :** Displays the shape of the transformed training set. Notice that the original shape of (5400, 8) is transformed into (3779, 28) for the transformed train set. The number of features has increased from 8 from 28 due to categorical encoding <br/>
# <br/>
# - **Transformed Test Set :** Displays the shape of transformed test/hold-out set. There are 1621 samples in test/hold-out set. This split is based on the default value of 70/30 that can be changed using `train_size` parameter in setup. <br/>
#
# Notice how a few tasks that are imperative to perform modeling are automatically handled, such as missing value imputation (in this case there are no missing values in training data, but we still need imputers for unseen data), categorical encoding etc. Most of the parameters in `setup()` are optional and used for customizing the pre-processing pipeline. These parameters are out of scope for this tutorial but as you progress to the intermediate and expert levels, we will cover them in much greater detail.
#
#

# + {"colab_type": "text", "id": "xBqHzabEzZrT", "cell_type": "markdown"}
# # 7.0 Comparing All Models

# + {"colab_type": "text", "id": "QHiNl6UmzZrW", "cell_type": "markdown"}
# Comparing all models to evaluate performance is the recommended starting point for modeling once the setup is completed (unless you exactly know what kind of model you need, which is often not the case). This function trains all models in the model library and scores them using k-fold cross validation for metric evaluation. The output prints a score grid that shows average MAE, MSE, RMSE, R2, RMSLE and MAPE accross the folds (10 by default) along with training time.

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 421}, "colab_type": "code", "id": "atJfMGD6zZrb", "outputId": "af936da0-cc93-4429-ef61-20d940e0aa4e"}
best = compare_models(exclude = ['ransac'])

# + {"colab_type": "text", "id": "epD0BEVyzZrr", "cell_type": "markdown"}
# Two simple words of code ***(not even a line)*** have trained and evaluated over 20 models using cross validation. The score grid printed above highlights the highest performing metric for comparison purposes only. The grid by default is sorted using `R2` (highest to lowest) which can be changed by passing `sort` parameter. For example `compare_models(sort = 'RMSLE')` will sort the grid by RMSLE (lower to higher since lower is better). If you want to change the fold parameter from the default value of `10` to a different value then you can use the `fold` parameter. For example `compare_models(fold = 5)` will compare all models on 5 fold cross validation. Reducing the number of folds will improve the training time. By default, compare_models return the best performing model based on default sort order but can be used to return a list of top N models by using `n_select` parameter. </br>
#
# Notice that how `exclude` parameter is used to block certain models (in this case `RANSAC`).

# + {"colab_type": "text", "id": "ZzpBazV1zZrx", "cell_type": "markdown"}
# # 8.0 Create a Model

# + {"colab_type": "text", "id": "IPqPRp5OzZr1", "cell_type": "markdown"}
# `create_model` is the most granular function in PyCaret and is often the foundation behind most of the PyCaret functionalities. As the name suggests this function trains and evaluates a model using cross validation that can be set with fold parameter. The output prints a score grid that shows MAE, MSE, RMSE, R2, RMSLE and MAPE by fold. 
#
# For the remaining part of this tutorial, we will work with the below models as our candidate models. The selections are for illustration purposes only and do not necessarily mean they are the top performing or ideal for this type of data.
#
# - AdaBoost Regressor ('ada')
# - Light Gradient Boosting Machine ('lightgbm') 
# - Decision Tree	 ('dt')
#
# There are 25 regressors available in the model library of PyCaret. To see list of all regressors either check the docstring or use `models` function to see the library.
# -

models()

# + {"colab_type": "text", "id": "wxKHHQcbzZr5", "cell_type": "markdown"}
# ### 8.1 AdaBoost Regressor

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 392}, "colab_type": "code", "id": "-NVGDCR3zZr8", "outputId": "06f5fc68-d2a5-4b59-fea2-3661ea1cb29d"}
ada = create_model('ada')

# + {"colab": {}, "colab_type": "code", "id": "NHL2zciizZsI", "outputId": "d606ad03-ecd5-487b-b205-bfc7a841f3a5"}
print(ada)

# + {"colab_type": "text", "id": "T-dvDHxCzZsU", "cell_type": "markdown"}
# ### 8.2 Light Gradient Boosting Machine 

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 392}, "colab_type": "code", "id": "NC7OVDVrzZsX", "outputId": "a5abc702-d270-4134-892a-e9ecf82bdebb"}
lightgbm = create_model('lightgbm')

# + {"colab_type": "text", "id": "j8DvIuOrzZsm", "cell_type": "markdown"}
# ### 8.3 Decision Tree

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 392}, "colab_type": "code", "id": "1Y_Czm6xzZsr", "outputId": "df27ebb4-257b-440e-cd9b-de05ba0c0f35"}
dt = create_model('dt')

# + {"colab_type": "text", "id": "NsOBIl8szZs1", "cell_type": "markdown"}
# Notice that the Mean score of all models matches with the score printed in `compare_models()`. This is because the metrics printed in the `compare_models()` score grid are the average scores across all CV folds. Similar to `compare_models()`, if you want to change the fold parameter from the default value of 10 to a different value then you can use the `fold` parameter. For Example: `create_model('dt', fold = 5)` to create Decision Tree using 5 fold cross validation.

# + {"colab_type": "text", "id": "8RZB8YllzZs7", "cell_type": "markdown"}
# # 9.0 Tune a Model

# + {"colab_type": "text", "id": "AYYWC1X5zZs-", "cell_type": "markdown"}
# When a model is created using the `create_model` function it uses the default hyperparameters to train the model. In order to tune hyperparameters, the `tune_model` function is used. This function automatically tunes the hyperparameters of a model using `Random Grid Search` on a pre-defined search space. The output prints a score grid that shows MAE, MSE, RMSE, R2, RMSLE and MAPE by fold. To use the custom search grid, you can pass `custom_grid` parameter in the `tune_model` function (see 9.2 LightGBM tuning below).

# + {"colab_type": "text", "id": "5uUSmZLGzZtB", "cell_type": "markdown"}
# ### 9.1 AdaBoost Regressor

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 392}, "colab_type": "code", "id": "XM7qgcGIzZtE", "outputId": "e87c7fac-4dfe-4733-ddc5-d44359b0ea0d"}
tuned_ada = tune_model(ada)

# + {"colab": {}, "colab_type": "code", "id": "Ul0HJFoRzZtU", "outputId": "5ac4ab0b-c746-4a9f-a286-3e5cd2bbe536"}
print(tuned_ada)

# + {"colab_type": "text", "id": "3kvdvfdUzZtj", "cell_type": "markdown"}
# ### 9.2 Light Gradient Boosting Machine
# -

import numpy as np
lgbm_params = {'num_leaves': np.arange(10,200,10),
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'learning_rate': np.arange(0.1,1,0.1)
                        }

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 392}, "colab_type": "code", "id": "s1agvmDFzZtm", "outputId": "7cfab1a7-e7c2-40df-ad1e-a40067bcc4e0"}
tuned_lightgbm = tune_model(lightgbm, custom_grid = lgbm_params)
# -

print(tuned_lightgbm)

# + {"colab_type": "text", "id": "Ovz73MkgzZtx", "cell_type": "markdown"}
# ### 9.3 Decision Tree

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 392}, "colab_type": "code", "id": "kFImcOpXzZt0", "outputId": "36165e01-0b2e-4fa4-efcc-5efa70a3236f"}
tuned_dt = tune_model(dt)

# + {"colab_type": "text", "id": "TcSjA7SUzZt-", "cell_type": "markdown"}
# By default, `tune_model` optimizes `R2` but this can be changed using optimize parameter. For example: tune_model(dt, optimize = 'MAE') will search for the hyperparameters of a Decision Tree Regressor that results in the lowest `MAE` instead of highest `R2`. For the purposes of this example, we have used the default metric `R2` for the sake of simplicity only. The methodology behind selecting the right metric to evaluate a regressor is beyond the scope of this tutorial but if you would like to learn more about it, you can __[click here](https://www.dataquest.io/blog/understanding-regression-error-metrics/)__ to develop an understanding on regression error metrics.
#
# Metrics alone are not the only criteria you should consider when finalizing the best model for production. Other factors to consider include training time, standard deviation of k-folds etc. As you progress through the tutorial series we will discuss those factors in detail at the intermediate and expert levels. For now, let's move forward considering the Tuned Light Gradient Boosting Machine stored in the `tuned_lightgbm` variable as our best model for the remainder of this tutorial.

# + {"colab_type": "text", "id": "HR-mHgtCzZuE", "cell_type": "markdown"}
# # 10.0 Plot a Model

# + {"colab_type": "text", "id": "N6i7Ggg_zZuH", "cell_type": "markdown"}
# Before model finalization, the `plot_model()` function can be used to analyze the performance across different aspects such as Residuals Plot, Prediction Error, Feature Importance etc. This function takes a trained model object and returns a plot based on the test / hold-out set. 
#
# There are over 10 plots available, please see the `plot_model()` docstring for the list of available plots.

# + {"colab_type": "text", "id": "HJCYRQj9zZuU", "cell_type": "markdown"}
# ### 10.1 Residual Plot

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 376}, "colab_type": "code", "id": "ml-qe8dTzZuX", "outputId": "4c69b0b8-8e82-4003-f0e2-158b19a79f34"}
plot_model(tuned_lightgbm)

# + {"colab_type": "text", "id": "rM9dWgfVzZuh", "cell_type": "markdown"}
# ### 10.2 Prediction Error Plot

# + {"colab": {}, "colab_type": "code", "id": "GPwWRYehzZuk", "outputId": "e3dfe255-fa08-42f7-e556-74c7909f6e6e"}
plot_model(tuned_lightgbm, plot = 'error')

# + {"colab_type": "text", "id": "dWu_EtTGzZuu", "cell_type": "markdown"}
# ### 10.3 Feature Importance Plot

# + {"colab": {}, "colab_type": "code", "id": "7Yh852PPzZux", "outputId": "38295169-5000-4a07-e71d-76de9877ab42"}
plot_model(tuned_lightgbm, plot='feature')

# + {"colab_type": "text", "id": "qI_tk-8RzZu8", "cell_type": "markdown"}
# *Another* way to analyze the performance of models is to use the `evaluate_model()` function which displays a user interface for all of the available plots for a given model. It internally uses the `plot_model()` function.

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 398, "referenced_widgets": ["2f905c9057e849cfb68d8b4a73a9ae2c", "4705ae97c5c341db8a7efcc7e4334c79", "202f5f0dfbea4f6cb6a75e4aebd370c7", "4c0b5f15356140e9bd6ed3a2e0850baa", "63e7b6e04e3d43118698abaca0973960", "b7731479f5f141289a4939cb73adfb28", "5eb7949ac4a140118e9ee6be49921284", "19ea86ac89a349e281011fbd327c29c0", "c9bbc67e75d1477e8cd7a64502b98704", "5334e09f4bed4a0ba45635300e76027e", "14244d0fca4f40a8b3552675869748a3", "baf068abf86f44a9a9502f79dfe17eb5", "3495c4e24f1d4b22af1117ea2403a4fa", "83a10da0205742d9a383671a0be02606", "4ae774bce02d4c1ab1351cd75b25cf17", "ee5416de1bb1462e87b2fa6f9065d646", "36a9c708b65e498586a07d526e6a5d06", "aeb87dce945f4d588807575e9f6e64c3"]}, "colab_type": "code", "id": "J4ryBACHzZu_", "outputId": "246ab553-8043-4cbe-fe99-7ea00bb13aed"}
evaluate_model(tuned_lightgbm)

# + {"colab_type": "text", "id": "CxKARgKAzZvJ", "cell_type": "markdown"}
# # 11.0 Predict on Test / Hold-out Sample

# + {"colab_type": "text", "id": "r8k_rmplzZvL", "cell_type": "markdown"}
# Before finalizing the model, it is advisable to perform one final check by predicting the test/hold-out set and reviewing the evaluation metrics. If you look at the information grid in Section 6 above, you will see that 30% (1621 samples) of the data has been separated out as a test/hold-out sample. All of the evaluation metrics we have seen above are cross-validated results based on training set (70%) only. Now, using our final trained model stored in the `tuned_lightgbm` variable we will predict the hold-out sample and evaluate the metrics to see if they are materially different than the CV results.

# + {"colab": {}, "colab_type": "code", "id": "ozTyeSjCzZvY", "outputId": "1cdf25f4-0988-40fa-c8e9-5695fec11f05"}
predict_model(tuned_lightgbm);

# + {"colab_type": "text", "id": "UouOXHaxzZvo", "cell_type": "markdown"}
# The R2 on the test/hold-out set is **`0.9652`** compared to **`0.9708`** achieved on `tuned_lightgbm` CV results (in section 9.2 above). This is not a significant difference. If there is a large variation between the test/hold-out and CV results, then this would normally indicate over-fitting but could also be due to several other factors and would require further investigation. In this case, we will move forward with finalizing the model and predicting on unseen data (the 10% that we had separated in the beginning and never exposed to PyCaret).
#
# (TIP : It's always good to look at the standard deviation of CV results when using `create_model`.)

# + {"colab_type": "text", "id": "J0PmhEQFzZvr", "cell_type": "markdown"}
# # 12.0 Finalize Model for Deployment

# + {"colab_type": "text", "id": "Rtaj0uWgzZvx", "cell_type": "markdown"}
# Model finalization is the last step in the experiment. A normal machine learning workflow in PyCaret starts with `setup()`, followed by comparing all models using `compare_models()` and shortlisting a few candidate models (based on the metric of interest) to perform several modeling techniques such as hyperparameter tuning, ensembling, stacking etc. This workflow will eventually lead you to the best model for use in making predictions on new and unseen data. The `finalize_model()` function fits the model onto the complete dataset including the test/hold-out sample (30% in this case). The purpose of this function is to train the model on the complete dataset before it is deployed in production.

# + {"colab": {}, "colab_type": "code", "id": "UPk310pezZv0"}
final_lightgbm = finalize_model(tuned_lightgbm)

# + {"colab": {}, "colab_type": "code", "id": "IGFUGDAPzZwF", "outputId": "ddaa33c4-f728-444a-fc33-47da5c37304a"}
print(final_lightgbm)

# + {"colab_type": "text", "id": "QmJjaIkrzZwQ", "cell_type": "markdown"}
# **Caution:** One final word of caution. Once the model is finalized using `finalize_model()`, the entire dataset including the test/hold-out set is used for training. As such, if the model is used for predictions on the hold-out set after `finalize_model()` is used, the information grid printed will be misleading as you are trying to predict on the same data that was used for modeling. In order to demonstrate this point only, we will use `final_lightgbm` under `predict_model()` to compare the information grid with the one above in section 11. 

# + {"colab": {}, "colab_type": "code", "id": "bmYJRTAyzZwU", "outputId": "58e2f57b-abda-4166-c82b-af52ab6f18b8"}
predict_model(final_lightgbm);

# + {"colab_type": "text", "id": "5NkpL1ZHzZwr", "cell_type": "markdown"}
# Notice how the R2 in the `final_lightgbm` has increased to **`0.9891`** from **`0.9652`**, even though the model is same. This is because the `final_lightgbm` variable is trained on the complete dataset including the test/hold-out set.

# + {"colab_type": "text", "id": "CgKSkSsZzZwv", "cell_type": "markdown"}
# # 13.0 Predict on Unseen Data

# + {"colab_type": "text", "id": "6n7QFM94zZwy", "cell_type": "markdown"}
# The `predict_model()` function is also used to predict on the unseen dataset. The only difference from section 11 above is that this time we will pass the `data_unseen` parameter. `data_unseen` is the variable created at the beginning of the tutorial and contains 10% (600 samples) of the original dataset which was never exposed to PyCaret. (see section 5 for explanation)

# + {"colab": {}, "colab_type": "code", "id": "YdlpJUx0zZw4", "outputId": "5b45a2b5-c9a1-4d20-80f7-28211f07d586"}
unseen_predictions = predict_model(final_lightgbm, data=data_unseen)
unseen_predictions.head()

# + {"colab_type": "text", "id": "wZnpuHoDzZxG", "cell_type": "markdown"}
# The `Label` column is added onto the `data_unseen` set. Label is the predicted value using the `final_lightgbm` model. If you want predictions to be rounded, you can use `round` parameter inside `predict_model()`. You can also check the metrics on this since you have actual target column `Price` available. To do that we will use pycaret.utils module. See example below:
# -

from pycaret.utils import check_metric
check_metric(unseen_predictions.Price, unseen_predictions.Label, 'R2')

# + {"colab_type": "text", "id": "os2dbiIrzZxJ", "cell_type": "markdown"}
# # 14.0 Saving the Model

# + {"colab_type": "text", "id": "46CV19RlzZxL", "cell_type": "markdown"}
# We have now finished the experiment by finalizing the `tuned_lightgbm` model which is now stored in `final_lightgbm` variable. We have also used the model stored in `final_lightgbm` to predict `data_unseen`. This brings us to the end of our experiment, but one question is still to be asked: What happens when you have more new data to predict? Do you have to go through the entire experiment again? The answer is no, PyCaret's inbuilt function `save_model()` allows you to save the model along with entire transformation pipeline for later use.

# + {"colab": {}, "colab_type": "code", "id": "tXl6hkG9zZxN", "outputId": "4af19b19-6c0c-4b1e-a4f7-249e729091bd"}
save_model(final_lightgbm,'Final LightGBM Model 25Nov2020')

# + {"colab_type": "text", "id": "q2wgdZ5ozZxX", "cell_type": "markdown"}
# (TIP : It's always good to use date in the filename when saving models, it's good for version control.)

# + {"colab_type": "text", "id": "9LsyznpCzZxb", "cell_type": "markdown"}
# # 15.0 Loading the Saved Model

# + {"colab_type": "text", "id": "7ZH-4EMLzZxd", "cell_type": "markdown"}
# To load a saved model at a future date in the same or an alternative environment, we would use PyCaret's `load_model()` function and then easily apply the saved model on new unseen data for prediction.

# + {"colab": {}, "colab_type": "code", "id": "2hsqdgn3zZxg", "outputId": "9db6b97b-ea93-452b-e4b4-121bea203bd3"}
saved_final_lightgbm = load_model('Final LightGBM Model 25Nov2020')

# + {"colab_type": "text", "id": "NBAXt62nzZx5", "cell_type": "markdown"}
# Once the model is loaded in the environment, you can simply use it to predict on any new data using the same `predict_model()` function. Below we have applied the loaded model to predict the same `data_unseen` that we used in section 13 above.

# + {"colab": {}, "colab_type": "code", "id": "y7debJpCzZx8"}
new_prediction = predict_model(saved_final_lightgbm, data=data_unseen)

# + {"colab": {}, "colab_type": "code", "id": "8hT1v3N0zZyD", "outputId": "c5f9994b-1859-49d0-9bb0-4d7f6c506b98"}
new_prediction.head()

# + {"colab_type": "text", "id": "cuVEPftKzZyK", "cell_type": "markdown"}
# Notice that the results of `unseen_predictions` and `new_prediction` are identical.
# -

from pycaret.utils import check_metric
check_metric(new_prediction.Price, new_prediction.Label, 'R2')

# + {"colab_type": "text", "id": "uE3tuIUHzZyL", "cell_type": "markdown"}
# # 16.0 Wrap-up / Next Steps?

# + {"colab_type": "text", "id": "ct8oOjuvzZyS", "cell_type": "markdown"}
# This tutorial has covered the entire machine learning pipeline from data ingestion, pre-processing, training the model, hyperparameter tuning, prediction and saving the model for later use. We have completed all of these steps in less than 10 commands which are naturally constructed and very intuitive to remember such as `create_model()`, `tune_model()`, `compare_models()`. Re-creating the entire experiment without PyCaret would have taken well over 100 lines of code in most libraries.
#
# We have only covered the basics of `pycaret.regression`. In following tutorials we will go deeper into advanced pre-processing, ensembling, generalized stacking and other techniques that allow you to fully customize your machine learning pipeline and are must know for any data scientist.
#
# See you at the next tutorial. Follow the link to __[Regression Tutorial (REG102) - Level Intermediate](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Intermediate%20-%20REG102.ipynb)__
