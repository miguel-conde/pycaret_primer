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

# + {"colab_type": "text", "id": "opRJcZi01RRO", "cell_type": "markdown"}
# #  <span style="color:orange">Regression Tutorial (REG102) - Level Intermediate</span>

# + {"colab_type": "text", "id": "wi1fabUC1RRY", "cell_type": "markdown"}
# **Created using: PyCaret 2.2** <br />
# **Date Updated: November 25, 2020**
#
# # 1.0 Tutorial Objective
# Welcome to the regression tutorial **(REG102)** - Level Intermediate. This tutorial assumes that you have completed __[Regression Tutorial (REG101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Beginner%20-%20REG101.ipynb)__. If you haven't used PyCaret before and this is your first tutorial, we strongly recommend you to go back and progress through the beginner tutorial to understand the basics of working in PyCaret.
#
# In this tutorial we will use the `pycaret.regression` module to learn:
#
# * **Normalization:**  How to normalize and scale the dataset
# * **Transformation:**  How to apply transformations that make the data linear and approximately normal
# * **Target Transformation:**  How to apply transformations to the target variable
# * **Combine Rare Levels:**  How to combine rare levels in categorical features
# * **Bin Numeric Variables:**  How to bin numeric variables and transform numeric features into categorical ones using 'sturges' rule
# * **Model Ensembling and Stacking:**  How to boost model performance using several ensembling techniques such as Bagging, Boosting, Voting and Generalized Stacking.
# * **Experiment Logging:** How to log experiments in PyCaret using MLFlow backend
#
# Read Time : Approx 60 Minutes
#
#
# ## 1.1 Installing PyCaret
# If you haven't installed PyCaret yet. Please follow the link to __[Beginner's Tutorial](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Beginner%20-%20REG101.ipynb)__ for instructions on how to install pycaret.
#
# ## 1.2 Pre-Requisites
# - Python 3.6 or greater
# - PyCaret 2.0 or greater
# - Internet connection to load data from pycaret's repository
# - Completion of Regression Tutorial (REG101) - Level Beginner
#
# ## 1.3 For Google colab users:
# If you are running this notebook on Google colab, run the following code at top of your notebook to display interactive visuals.<br/>
# <br/>
# `from pycaret.utils import enable_colab` <br/>
# `enable_colab()`
#
# ## 1.4 See also:
# - __[Regression Tutorial (REG101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Beginner%20-%20REG101.ipynb)__
# - __[Regression Tutorial (REG103) - Level Expert](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Expert%20-%20REG103.ipynb)__

# + {"colab_type": "text", "id": "h1iXlqnO1RRd", "cell_type": "markdown"}
# # 2.0 Brief overview of techniques covered in this tutorial
# Before we into the practical execution of the techniques mentioned above in Section 1, it is important to understand what are these techniques are and when to use them. More often than not most of these techniques will help linear and parametric algorithms, however it is not surprising to also see performance gains in tree-based models. The Below explanations are only brief and we recommend that you do extra reading to dive deeper and get a more thorough understanding of these techniques.
#
# - **Normalization:** Normalization / Scaling (often used interchangeably with standardization) is used to transform the actual values of numeric variables in a way that provides helpful properties for machine learning. Many algorithms such as Linear Regression, Support Vector Machine and K Nearest Neighbors assume that all features are centered around zero and have variances that are at the same level of order. If a particular feature in a dataset has a variance that is larger in order of magnitude than other features, the model may not understand all features correctly and could perform poorly. __[Read more](https://sebastianraschka.com/Articles/2014_about_feature_scaling.html#z-score-standardization-or-min-max-scaling)__ <br/>
# <br/>
# - **Transformation:** While normalization transforms the range of data to remove the impact of magnitude in variance, transformation is a more radical technique as it changes the shape of the distribution so that transformed data can be represented by a normal or approximate normal distirbution. In general, you should transform the data if using algorithms that assume normality or a gaussian distribution. Examples of such models are Linear Regression, Lasso Regression and Ridge Regression. __[Read more](https://en.wikipedia.org/wiki/Power_transform)__<br/>
# <br/>
# - **Target Transformation:** This is similar to the `transformation` technique explained above with the exception that this is only applied to the target variable. __[Read more](https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html)__ to understand the effects of transforming the target variable in regression.<br/>
# <br/>
# - **Combine Rare Levels:** Sometimes categorical features have levels that are insignificant in the frequency distribution. As such, they may introduce noise into the dataset due to a limited sample size for learning. One way to deal with rare levels in categorical features is to combine them into a new class. <br/>
# <br/>
# - **Bin Numeric Variables:** Binning or discretization is the process of transforming numerical variables into categorical features. An example would be `Carat Weight` in this experiment. It is a continious distribution of numeric values that can be discretized into intervals. Binning may improve the accuracy of a predictive model by reducing the noise or non-linearity in the data. PyCaret automatically determines the number and size of bins using Sturges rule.  __[Read more](https://www.vosesoftware.com/riskwiki/Sturgesrule.php)__<br/>
# <br/>
# - **Model Ensembling and Stacking:** Ensemble modeling is a process where multiple diverse models are created to predict an outcome. This is achieved either by using many different modeling algorithms or using different samples of training data sets. The ensemble model then aggregates the predictions of each base model resulting in one final prediction for the unseen data. The motivation for using ensemble models is to reduce the generalization error of the prediction. As long as the base models are diverse and independent, the prediction error of the model decreases when the ensemble approach is used. The two most common methods in ensemble learning are `Bagging` and `Boosting`. Stacking is also a type of ensemble learning where predictions from multiple models are used as input features for a meta model that predicts the final outcome. __[Read more](https://blog.statsbot.co/ensemble-learning-d1dcd548e936)__<br/>
# <br/>
# - **Tuning Hyperparameters of Ensemblers:** Similar to hyperparameter tuning for a single machine learning model, we will also learn how to tune hyperparameters for an ensemble model.

# + {"colab_type": "text", "id": "zrNWkm4v1RRh", "cell_type": "markdown"}
# # 3.0 Dataset for the Tutorial

# + {"colab_type": "text", "id": "n_N5u8MZ1RRm", "cell_type": "markdown"}
# For this tutorial we will be using the same dataset that was used in __[Regression Tutorial (REG101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Beginner%20-%20REG101.ipynb)__.
#
# #### Dataset Acknowledgements:
# This case was prepared by Greg Mills (MBA ’07) under the supervision of Phillip E. Pfeifer, Alumni Research Professor of Business Administration. Copyright (c) 2007 by the University of Virginia Darden School Foundation, Charlottesville, VA. All rights reserved.
#
# The original dataset and description can be __[found here.](https://github.com/DardenDSC/sarah-gets-a-diamond)__ 

# + {"colab_type": "text", "id": "BuL2VDPq1RRq", "cell_type": "markdown"}
# # 4.0 Getting the Data

# + {"colab_type": "text", "id": "OljCicBG1RRu", "cell_type": "markdown"}
# You can download the data from the original source __[found here](https://github.com/DardenDSC/sarah-gets-a-diamond)__ and load it using the pandas read_csv function or you can use PyCaret's data respository to load the data using the get_data function (This will require internet connection).

# + {"colab": {}, "colab_type": "code", "id": "p84oCDfz1RRz", "outputId": "41972d81-7bf0-4023-9a62-8a75fb059d4b"}
from pycaret.datasets import get_data
dataset = get_data('diamond', profile=True)

# + {"colab_type": "text", "id": "M_odMvI51RSI", "cell_type": "markdown"}
# Notice that when the `profile` parameter is to `True`, it displays a data profile for exploratory data analysis. Several pre-processing steps as discussed in section 2 above will be performed in this experiment based on this analysis. Let's summarize how the profile has helped make critical pre-processing choices with the data.
#
# - **Missing Values:** There are no missing values in the data. However, we still need imputers in our pipeline just in case the new unseen data has missing values (not applicable in this case). When you execute the `setup()` function, imputers are created and stored in the pipeline automatically. By default, it uses a mean imputer for numeric values and a constant imputer for categorical. This can be changed using the `numeric_imputation` and `categorical_imputation` parameters in `setup()`. <br/>
# <br/>
# - **Combine Rare Levels:** Notice the distribution of the `Clarity` feature in the dataset. It has 7 distinct classes of which `FL` only appears 4 times. Similarly in the `Cut` feature, the `Fair` level only appears `2.1%` of the time in the training dataset. We will use the `combine_rare_categories` parameter in the setup to combine the rare levels. <br/>
# <br/>
# - **Data Scale / Range:** Notice how the scale / range of `Carat Weight` is significantly different than the `Price` variable. Carat Weight ranges from between 0.75 to 2.91 while Price ranges from 2,184 all the way up to 101,561. We will deal with this problem by using the `normalize` parameter in setup. <br/>
# <br/>
# - **Target Transformation:** The target variable `Price` is not normally distributed. It is right skewed with high kurtosis. We will use the `transform_target` parameter in the setup to apply a linear transformation on the target variable. `<br/>
# <br/>
# - **Bin Numeric Features:** `Carat Weight` is the only numeric feature. When looking at its histogram, the distribution seems to have natural breaks. Binning will convert it into a categorical feature and create several levels using sturges' rule. This will help remove the noise for linear algorithms. <br/>
# <br/>

# + {"colab": {}, "colab_type": "code", "id": "4FBfpsdK1RSM", "outputId": "7d7ccdd1-5a13-4bbc-e158-2d6d477a8835"}
#check the shape of data
dataset.shape

# + {"colab_type": "text", "id": "8pp-pq_I1RSb", "cell_type": "markdown"}
# In order to demonstrate the `predict_model()` function on unseen data, a sample of 600 has been withheld from the original dataset to be used for predictions. This should not be confused with a train/test split as this particular split is performed to simulate a real life scenario. Another way to think about this is that these 600 records were not available at the time when the machine learning experiment was performed.

# + {"colab": {}, "colab_type": "code", "id": "NGqMrv8U1RSf", "outputId": "96f2ccee-68fa-4df0-cfb8-edbdcbf9eb54"}
data = dataset.sample(frac=0.9, random_state=786)
data_unseen = dataset.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions ' + str(data_unseen.shape))

# + {"colab_type": "text", "id": "vZbKDK261RSt", "cell_type": "markdown"}
# # 5.0 Setting up Environment in PyCaret

# + {"colab_type": "text", "id": "NW9Lw4l91RSz", "cell_type": "markdown"}
# In the previous tutorial __[Regression (REG101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Beginner%20-%20REG101.ipynb)__ we learned how to initialize the environment in pycaret using `setup()`. No additional parameters were passed in our last example as we did not perform any pre-processing steps (other than those that are imperative for machine learning experiments which were performed automatically by PyCaret). In this example we will take it to the next level by customizing the pre-processing pipeline using `setup()`. Let's look at how to implement all the steps discussed in section 4 above.
# -

from pycaret.regression import *

# + {"colab": {}, "colab_type": "code", "id": "Atcjop5M1RTE", "outputId": "9f17e438-1144-4063-ed18-7ff2a07c9d07"}
exp_reg102 = setup(data = data, target = 'Price', session_id=123,
                  normalize = True, transformation = True, transform_target = True, 
                  combine_rare_levels = True, rare_level_threshold = 0.05,
                  remove_multicollinearity = True, multicollinearity_threshold = 0.95, 
                  bin_numeric_features = ['Carat Weight'],
                  log_experiment = True, experiment_name = 'diamond1') 

# + {"colab_type": "text", "id": "OgDwm-V41RTR", "cell_type": "markdown"}
# Note that this is the same setup grid that was shown in __[Regression Tutorial (REG101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Beginner%20-%20REG101.ipynb)__. The only difference here is the customization parameters that were passed to `setup()` are now set to `True`. Also notice that the `session_id` is the same as the one used in the beginner tutorial, which means that the effect of randomization is completely isolated. Any improvements we see in this experiment are solely due to the pre-processing steps taken in `setup()` or any other modeling techniques used in later sections of this tutorial.

# + {"colab_type": "text", "id": "E5ZGIV_61RTV", "cell_type": "markdown"}
# # 6.0 Comparing All Models

# + {"colab_type": "text", "id": "4gCEovkI1RTZ", "cell_type": "markdown"}
# Similar to __[Regression Tutorial (REG101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Beginner%20-%20REG101.ipynb)__ we will also begin this tutorial with `compare_models()`. We will then compare the below results with the last experiment.

# + {"colab": {}, "colab_type": "code", "id": "veolaq_w1RTd", "outputId": "57f4988a-64f4-4dcb-fe21-1c89293ac94d"}
top3 = compare_models(exclude = ['ransac'], n_select = 3)
# -

# Notice that we have used `n_select` parameter within `compare_models`. In last tutorial you have seen that compare_models by default returns the best performing model (single model based on default sort order). However you can use `n_select` parameter to return top N models. In this example `compare_models` has returned Top 3 models.

type(top3)

print(top3)

# + {"colab_type": "text", "id": "E3t7LNHf1RTr", "cell_type": "markdown"}
# For the purpose of comparison we will use the `RMSLE` score. Notice how drastically a few of the algorithms have improved after we performed a few pre-processing steps in `setup()`. 
# - Linear Regression RMSLE improved from `0.6690` to `0.0973`
# - Ridge Regression RMSLE improved from `0.6689` to `0.0971`
# - Huber Regression RMSLE improved from `0.4333` to `0.0972`
#
# To see results for all of the models from the previous tutorial refer to Section 7 in __[Regression Tutorial (REG101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Beginner%20-%20REG101.ipynb)__.

# + {"colab_type": "text", "id": "OYkCSO7n1RTv", "cell_type": "markdown"}
# # 7.0 Create a Model

# + {"colab_type": "text", "id": "n8OqV9po1RTy", "cell_type": "markdown"}
# In the previous tutorial __[Regression (REG101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Beginner%20-%20REG101.ipynb)__ we learned how to create a model using the `create_model()` function. Now we will learn about a few other parameters that may come in handy. In this section, we will create all models using 5 fold cross validation. Notice how the `fold` parameter is passed inside `create_model()` to achieve this.

# + {"colab_type": "text", "id": "6TVGAA7Q1RT2", "cell_type": "markdown"}
# ### 7.1 Create Model (with 5 Fold CV)

# + {"colab": {}, "colab_type": "code", "id": "Re4yQ6rI1RT6", "outputId": "ba0fc954-c37d-4901-8557-5f90fc455a53"}
dt = create_model('dt', fold = 5)

# + {"colab_type": "text", "id": "axAmmrbZ1RUI", "cell_type": "markdown"}
# ### 7.2 Create Model (Metrics rounded to 2 decimals points)

# + {"colab": {}, "colab_type": "code", "id": "uYPm9kEK1RUM", "outputId": "713a2016-1d7b-4cc8-854f-bd20c3f1c38b"}
rf = create_model('rf', round = 2)

# + {"colab_type": "text", "id": "2ukwthju1RUb", "cell_type": "markdown"}
# Notice how passing the `round` parameter inside `create_model()` has rounded the evaluation metrics to 2 decimals.
# -

# ### 7.3 Create Model (KNN)

knn = create_model('knn')

print(knn)

# + {"colab_type": "text", "id": "Jlef9kIH1RUk", "cell_type": "markdown"}
# # 8.0 Tune a Model

# + {"colab_type": "text", "id": "5d9IwoCE1RUo", "cell_type": "markdown"}
# In the previous tutorial __[Regression (REG101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Beginner%20-%20REG101.ipynb)__ we learned how to automatically tune the hyperparameters of a model using pre-defined grids. Here we will introduce the the `n_iter` parameter in `tune_model()`. `n_iter` is the number of iterations within a random grid search. For every iteration, the model randomly selects one value from a pre-defined grid of hyperparameters. By default, the parameter is set to `10` which means there would be a maximum of 10 iterations to find the best value for hyperparameters. Increasing the value may improve the performance but will also increase the training time. See the example below:

# + {"colab": {}, "colab_type": "code", "id": "2RL-Efjf1RUt", "outputId": "4fbfa307-964d-40e0-956d-5029f5db5ee7"}
tuned_knn = tune_model(knn)

# + {"colab": {}, "colab_type": "code", "id": "ETYEYBDO1RU8", "outputId": "63eb436d-ce8e-4d4d-986e-98b6e10164cb"}
tuned_knn2 = tune_model(knn, n_iter = 50)

# + {"colab_type": "text", "id": "iLFFpJnU1RVL", "cell_type": "markdown"}
# Notice how two tuned K Nearest Neighbors were created based on the `n_iter` parameter. In `tuned_knn`, the `n_iter` parameter is left to the default value and resulted in R2 of `0.6504`. In `tuned_knn2`, the `n_iter` parameter was set to `50` and the R2 improved to `0.6689`. Observe the differences between the hyperparameters of `tuned_knn` and `tuned_knn2` below:

# + {"colab": {}, "colab_type": "code", "id": "9xDXsAYt1RVO", "outputId": "642a84d9-2046-4b8e-f963-5acc481527d5"}
plot_model(tuned_knn, plot = 'parameter')

# + {"colab": {}, "colab_type": "code", "id": "9K_webQ11RVa", "outputId": "468d764b-edd6-47d1-936c-d9f07ed6ae3a"}
plot_model(tuned_knn2, plot = 'parameter')

# + {"colab_type": "text", "id": "pwyjV3KW1RVo", "cell_type": "markdown"}
# # 9.0 Ensemble a Model

# + {"colab_type": "text", "id": "sr64qPJg1RVt", "cell_type": "markdown"}
# Ensembling is a common machine learning technique used to improve the performance of models (mostly tree based). There are various techniques for ensembling that we will cover in this section. These include Bagging and Boosting __[(Read More)](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205)__. We will use the `ensemble_model()` function in PyCaret which ensembles the trained base estimators using the method defined in the `method` parameter.

# + {"colab": {}, "colab_type": "code", "id": "3TQio5301RVx", "outputId": "91a74209-e9f7-4958-eac9-554249c0f708"}
# lets create a simple dt
dt = create_model('dt')

# + {"colab_type": "text", "id": "oN1T29bo1RV-", "cell_type": "markdown"}
# ### 9.1 Bagging

# + {"colab": {}, "colab_type": "code", "id": "fXN5tYyE1RWB", "outputId": "e6a354e5-1a2c-47cb-f6e7-643fcba344c3"}
bagged_dt = ensemble_model(dt)

# + {"colab": {}, "colab_type": "code", "id": "rZnQhUoJ1RWN", "outputId": "43d56e9d-2eec-45c8-b8c6-a8ae64b17ad7"}
# check the parameter of bagged_dt
print(bagged_dt)

# + {"colab_type": "text", "id": "QMKjtHZA1RWY", "cell_type": "markdown"}
# Notice how ensembling has improved the `RMSLE` from `0.1082` to `0.0938`. In the above example we have used the default parameters of `ensemble_model()` which uses the `Bagging` method. Let's try `Boosting` by changing the `method` parameter in `ensemble_model()`. See example below: 

# + {"colab_type": "text", "id": "TFk_UbV51RWc", "cell_type": "markdown"}
# ### 9.2 Boosting

# + {"colab": {}, "colab_type": "code", "id": "lMP9Uj801RWf", "outputId": "442145ff-1a0a-4c72-ec3c-cc29f5091477"}
boosted_dt = ensemble_model(dt, method = 'Boosting')

# + {"colab_type": "text", "id": "TUJdkdR-1RWp", "cell_type": "markdown"}
# Notice how easy it is to ensemble models in PyCaret. By simply changing the `method` parameter you can do bagging or boosting which would otherwise have taken multiple lines of code. Note that `ensemble_model()` will by default build `10` estimators. This can be changed using the `n_estimators` parameter. Increasing the number of estimators can sometimes improve results. See an example below:

# + {"colab": {}, "colab_type": "code", "id": "S73V76Fr1RWu", "outputId": "a9b7c767-7f66-45c0-ee0e-fd2a620f1125"}
bagged_dt2 = ensemble_model(dt, n_estimators=50)

# + {"colab_type": "text", "id": "UuCvIXA21RW4", "cell_type": "markdown"}
# Notice how increasing the n_estimators parameter has improved the result. The bagged_dt model with the default `10` estimators resulted in a RMSLE of `0.0996` whereas in bagged_dt2 where `n_estimators = 50` the RMSLE improved to `0.0911`.

# + {"colab_type": "text", "id": "RdYEDmAQ1RXT", "cell_type": "markdown"}
# ### 9.3 Blending

# + {"colab_type": "text", "id": "-Q7HPxJY1RXW", "cell_type": "markdown"}
# Blending is another common technique for ensembling that can be used in PyCaret. It creates multiple models and then averages the individual predictions to form a final prediction. Let's see an example below:
# -

# train individual models to blend
lightgbm = create_model('lightgbm', verbose = False)
dt = create_model('dt', verbose = False)
lr = create_model('lr', verbose = False)

# blend individual models
blender = blend_models(estimator_list = [lightgbm, dt, lr])

# + {"colab": {}, "colab_type": "code", "id": "443dCPZN1RXZ", "outputId": "b415d161-dfad-4282-fb6b-89381e325fb9"}
# blend top3 models from compare_models
blender_top3 = blend_models(top3)
# -

print(blender_top3.estimators_)

# + {"colab_type": "text", "id": "g6KDkbVu1RXq", "cell_type": "markdown"}
# Now that we have created a `VotingRegressor` using the `blend_models()` function. The model returned by the `blend_models` function is just like any other model that you would create using `create_model()` or `tune_model()`. You can use this model for predictions on unseen data using `predict_model()` in the same way you would for any other model.

# + {"colab_type": "text", "id": "5hwtKpxV1RYa", "cell_type": "markdown"}
# ### 9.4 Stacking

# + {"colab_type": "text", "id": "3ZDMp0KO1RYi", "cell_type": "markdown"}
# Stacking is another popular technique for ensembling but is less commonly implemented due to practical difficulties. Stacking is an ensemble learning technique that combines multiple models via a meta-model. Another way to think about stacking is that multiple models are trained to predict the outcome and a meta-model is created that uses the predictions from those models as an input along with the original features. The implementation of `stack_models()` is based on Wolpert, D. H. (1992b). Stacked generalization __[(Read More)](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)__. 
#
# Let's see an example below using the top 3 models we have obtained from `compare_models`:

# + {"colab": {}, "colab_type": "code", "id": "wDdutrXy1RYp", "outputId": "f3c092c3-c307-488b-ee6f-cf104248f828"}
stacker = stack_models(top3)

# + {"colab_type": "text", "id": "roI0qUd61RY8", "cell_type": "markdown"}
# By default, the meta model (final model to generate predictions) is Linear Regression. The meta model can be changed using the `meta_model` parameter. See an example below:

# + {"colab": {}, "colab_type": "code", "id": "OOZr6EzJ1RZB", "outputId": "c29a338c-fda0-4c04-a384-ad821adf6013"}
xgboost = create_model('xgboost')
stacker2 = stack_models(top3, meta_model = xgboost)

# + {"colab_type": "text", "id": "cnbiUV5H1RZX", "cell_type": "markdown"}
# Before we wrap up this section, there is another parameter in `stack_models()` that we haven't seen yet called `restack`. This parameter controls the ability to expose the raw data to the meta model. When set to `True`, it exposes the raw data to the meta model along with all the predictions of the base level models. By default it is set to `True`. See the example below with the `restack` parameter changed to `False`.
# -

# # 10.0 Experiment Logging

# PyCaret 2.0 embeds MLflow Tracking component as a backend API and UI for logging parameters, code versions, metrics, and output files when running your machine learning code and for later visualizing the results. To log your experiments in pycaret simply use log_experiment and experiment_name parameter in the setup function, as we did in this example.
#
# You can start the UI on `localhost:5000`. Simply initiate the MLFlow server from command line or from notebook. See example below:

# to start the MLFlow server from notebook:
# !mlflow ui

# ### Open localhost:5000 on your browser (below is example of how UI looks like)
# ![title](https://i2.wp.com/pycaret.org/wp-content/uploads/2020/07/classification_mlflow_ui.png?resize=1080%2C508&ssl=1)

# + {"colab_type": "text", "id": "AYIcCby_1RdN", "cell_type": "markdown"}
# # 11.0 Wrap-up / Next Steps?

# + {"colab_type": "text", "id": "D94EyMJo1RdQ", "cell_type": "markdown"}
# We have covered a lot of new concepts in this tutorial. Most importantly we have seen how to use exploratory data analysis to customize a pipeline in `setup()` which has improved the results considerably when compared to what we saw earlier in __[Regression Tutorial (REG101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Beginner%20-%20REG101.ipynb)__. WWe have also learned how to perform and tune ensembling in PyCaret.
#
# There are however a few more advanced things to cover in `pycaret.regression` which include interpretating more complex tree based models using shapley values, advanced ensembling techniques such as multiple layer stacknet and more pre-processing pipeline methods. We will cover all of this in our next and final tutorial in the `pycaret.regression` series. 
#
# See you at the next tutorial. Follow the link to __[Regression Tutorial (REG103) - Level Expert](https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Expert%20-%20REG103.ipynb)__
