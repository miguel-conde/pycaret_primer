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

# + {"colab_type": "text", "id": "tuhTO0jP4EeE", "cell_type": "markdown"}
# #  <span style="color:orange">Binary Classification Tutorial (CLF102) - Level Intermediate</span>

# + {"colab_type": "text", "id": "uHu4QCDf4Eeh", "cell_type": "markdown"}
# **Created using: PyCaret 2.2** <br />
# **Date Updated: November 20, 2020**
#
# # 1.0 Tutorial Objective
# Welcome to the Binary Classification Tutorial **(CLF102)** - Level Intermediate. This tutorial assumes that you have completed __[Binary Classification Tutorial (CLF101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Beginner%20-%20%20CLF101.ipynb)__. If you haven't used PyCaret before and this is your first tutorial, we strongly recommend you to go back and progress through the beginner tutorial to understand the basics of working in PyCaret.
#
# In this tutorial we will use the `pycaret.classification` module to learn:
#
# * **Normalization:**  How to normalize and scale the dataset
# * **Transformation:**  How to apply transformations that make the data linear and approximately normal
# * **Ignore Low Variance:**  How to remove features with statistically insignificant variances to make the experiment more efficient
# * **Remove Multi-collinearity:**  How to remove multi-collinearity from the dataset to boost performance of Linear algorithms
# * **Group Features:**  How to extract statistical information from related features in the dataset
# * **Bin Numeric Variables:**  How to bin numeric variables and transform numeric features into categorical ones using 'sturges' rule
# * **Model Ensembling and Stacking:**  How to boost model performance using several ensembling techniques such as Bagging, Boosting, Soft/hard Voting and Generalized Stacking
# * **Model Calibration:**  How to calibrate probabilities of a classification model
# * **Experiment Logging:**  How to log experiments in PyCaret using MLFlow backend
#
# Read Time : Approx 60 Minutes
#
#
# ## 1.1 Installing PyCaret
# If you haven't installed PyCaret yet, please follow the link to __[Beginner's Tutorial](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Beginner%20-%20%20CLF101.ipynb)__ for instructions on how to install.
#
# ## 1.2 Pre-Requisites
# - Python 3.6 or greater
# - PyCaret 2.0 or greater
# - Internet connection to load data from pycaret's repository
# - Completion of Binary Classification Tutorial (CLF101) - Level Beginner
#
# ## 1.3 For Google colab users:
# If you are running this notebook on Google colab, run the following code at top of your notebook to display interactive visuals.<br/>
# <br/>
# `from pycaret.utils import enable_colab` <br/>
# `enable_colab()`
#
# ## 1.4 See also:
# - __[Binary Classification Tutorial (CLF101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Beginner%20-%20%20CLF101.ipynb)__
# - __[Binary Classification Tutorial (CLF103) - Level Expert](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Expert%20-%20CLF103.ipynb)__

# + {"colab_type": "text", "id": "0hNS3XgV4Eer", "cell_type": "markdown"}
# # 2.0 Brief overview of techniques covered in this tutorial
# Before we get into the practical execution of the techniques mentioned above in the Section 1, it is important to understand what these techniques are and when to use them. More often than not most of these techniques will help linear and parametric algorithms, however it is not suprising to also see performance gains in tree-based models. The below explanations are only brief and we recommend that you to do extra reading to dive deeper and get a more thorough understanding of these techniques.
#
# - **Normalization:** Normalization / Scaling (often used interchangeably with standardization) is used to transform the actual values of numeric variables in a way that provides helpful properties for machine learning. Many algorithms such as Logistic Regression, Support Vector Machine, K Nearest Neighbors and Naive Bayes assume that all features are centered around zero and have variances that are at at the same level of order. If a particular feature in a dataset has a variance that is larger in order of magnitude than other features, the model may not understand all features correctly and could perform poorly. For instance, in the dataset we are using for this example the `AGE` feature ranges between 21 to 79 while other numeric features range from 10,000 to 1,000,000. __[Read more](https://sebastianraschka.com/Articles/2014_about_feature_scaling.html#z-score-standardization-or-min-max-scaling)__ <br/>
# <br/>
# - **Transformation:** While normalization transforms the range of data to remove the impact of magnitude in variance, transformation is a more radical technique as it changes the shape of the distribution so that transformed data can be represented by a normal or approximate normal distirbution. In general, you should transform the data if using algorithms that assume normality or a gaussian distribution. Examples of such models are Logistic Regression, Linear Discriminant Analysis (LDA) and Gaussian Naive Bayes. (Pro tip: any method with “Gaussian” in the name probably assumes normality.) __[Read more](https://en.wikipedia.org/wiki/Power_transform)__<br/>
# <br/>
# - **Ignore Low Variance:** Datasets can sometimes contain categorical features that have a single unique or small number of values across samples. This kind of features are not only non-informative and add no value but are also sometimes harmful for few algorithms. Imagine a feature with only one unique value or few dominant unique values accross samples, they can be removed from the dataset by using the ignore low variance feature in PyCaret. <br/>
# <br/>
# - **Multi-collinearity:** Multi-collinearity is a state of very high intercorrelations or inter-associations among the independent features in the dataset. It is a type of disturbance in the data that is not handled well by machine learning models (mostly linear algorithms). Multi-collinearity may reduce overall coefficient of the model and cause unpredictable variance. This will lead to overfitting where the model may do great on a known training set but will fail with an unknown testing set. __[Read more](https://towardsdatascience.com/multicollinearity-in-data-science-c5f6c0fe6edf)__<br/>
# <br/>
# - **Group Features:** Sometimes datasets may contain features that are related at a sample level. For example in the `credit` dataset there are features called `BILL_AMT1 .. BILL_AMT6` which are related in such a way that `BILL_AMT1` is the amount of the bill 1 month ago and `BILL_AMT6` is the amount of the bill 6 months ago. Such features can be used to extract additional features based on the statistical properties of the distribution such as mean, median, variance, standard deviation etc. <br/>
# <br/>
# - **Bin Numeric Variables:** Binning or discretization is the process of transforming numerical variables into categorical features. An example would be the Age variable which is a continious distribution of numeric values that can be discretized into intervals (10-20 years, 21-30 etc.). Binning may improve the accuracy of a predictive model by reducing the noise or non-linearity in the data. PyCaret automatically determines the number and size of bins using Sturges rule.  __[Read more](https://www.vosesoftware.com/riskwiki/Sturgesrule.php)__<br/>
# <br/>
# - **Model Ensembling and Stacking:** Ensemble modeling is a process where multiple diverse models are created to predict an outcome. This is achieved either by using many different modeling algorithms or using different samples of training data sets. The ensemble model then aggregates the predictions of each base model resulting in one final prediction for the unseen data. The motivation for using ensemble models is to reduce the generalization error of the prediction. As long as the base models are diverse and independent, the prediction error of the model decreases when the ensemble approach is used. The two most common methods in ensemble learning are `Bagging` and `Boosting`. Stacking is also a type of ensemble learning where predictions from multiple models are used as input features for a meta model that predicts the final outcome. __[Read more](https://blog.statsbot.co/ensemble-learning-d1dcd548e936)__<br/>
# <br/>

# + {"colab_type": "text", "id": "2B2swEt84Ee1", "cell_type": "markdown"}
# # 3.0 Dataset for the Tutorial

# + {"colab_type": "text", "id": "rfr5f_GF4Ee7", "cell_type": "markdown"}
# For this tutorial we will be using the same dataset that was used in __[Binary Classification Tutorial (CLF101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Beginner%20-%20%20CLF101.ipynb)__
#
# #### Dataset Acknowledgements:
# Lichman, M. (2013). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
#
# The original dataset and data dictionary can be __[found here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)__ at the UCI Machine Learning Repository.

# + {"colab_type": "text", "id": "fGHO07ET4EfB", "cell_type": "markdown"}
# # 4.0 Getting the Data

# + {"colab_type": "text", "id": "yeUS_YV24EfG", "cell_type": "markdown"}
# You can download the data from the original source __[found here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)__ and load it using the pandas read_csv function or you can use PyCaret's data respository to load the data using the get_data function (This will require an internet connection).

# + {"colab": {}, "colab_type": "code", "id": "SwAC9YKj4EgZ", "outputId": "aac7ee4c-f6bb-421a-cecc-ac72ae86b81f"}
from pycaret.datasets import get_data
dataset = get_data('credit', profile=True)

# + {"colab_type": "text", "id": "NBjUo6XM4EhH", "cell_type": "markdown"}
# Notice that when the `profile` parameter is to `True`, it displays a data profile for exploratory data analysis. Several pre-processing steps as discussed in section 2 above will be performed in this experiment based on this analysis. Let's summarize how the profile has helped make critical pre-processing choices with the data.
#
# - **Missing Values:** There are no missing values in the data. However, we still need imputers in our pipeline just in case the new unseen data has missing values (not applicable in this case). When you execute the `setup()` function, imputers are created and stored in the pipeline automatically. By default, it uses a mean imputer for numeric values and a constant imputer for categorical. This can be changed using the `numeric_imputation` and `categorical_imputation` parameters in `setup()`. <br/>
# <br/>
# - **Multicollinearity:** There are high correlations between `BILL_AMT1 ... BIL_AMT6` which introduces multicollinearity into the data. We will remove multi-collinearity by using the `remove_multicollinearity` and `multicollinearity_threshold` parameters in setup. <br/>
# <br/>
# - **Data Scale / Range:** Notice how the scale / range of numeric features are different. For example the `AGE` feature ranges from between 21 to 79 and `BILL_AMT1` ranges from -165,580 to 964,511. This may cause problems for algorithms that assume all features have variance within the same order. In this case, the order of magnitude for `BILL_AMT1` is widely different than `AGE`. We will deal with this problem by using the `normalize` parameter in setup. <br/>
# <br/>
# - **Distribution of Feature Space:** Numeric features are not normally distributed. Look at the distributions of `LIMIT_BAL`, `BILL_AMT1` and `PAY_AMT1 ... PAY_AMT6`. A few features are also highly skewed such as `PAY_AMT1`. This may cause problems for algorithms that assume normal or approximate normal distributions of the data. Examples include Logistic Regression, Linear Discriminant Analysis (LDA) and Naive Bayes.  We will deal with this problem by using the `transformation` parameter in setup. <br/>
# <br/>
# - **Group Features:** From the data description we know that certain features are related with each other such as `BILL_AMT1 ... BILL_AMT6` and `PAY_AMT1 ... PAY_AMT6`. We will use the `group_features` parameter in setup to extract statistical information from these features.  <br/>
# <br/>
# - **Bin Numeric Features:** When looking at the correlations between the numeric features and the target variable, we that `AGE` and `LIMIT_BAL` are weak. We will use the `bin_numeric_features` parameter to remove the noise from these variables which may help linear algorithms. <br/>

# + {"colab": {}, "colab_type": "code", "id": "HuMuCeN44EhL", "outputId": "9d6210f4-d8e2-45ac-d92a-1b939220c02a"}
#check the shape of data
dataset.shape

# + {"colab_type": "text", "id": "WHyZrNCX4EhZ", "cell_type": "markdown"}
# In order to demonstrate the `predict_model()` function on unseen data, a sample of 1200 rows has been withheld from the original dataset to be used for predictions. This should not be confused with a train/test split as this particular split is performed to simulate a real life scenario. Another way to think about this is that these 1200 records were not available at the time when the machine learning experiment was performed.

# + {"colab": {}, "colab_type": "code", "id": "yYoi1mdi4Ehk", "outputId": "bd125cbe-1ea7-4f25-af39-e4d10e53e8ac"}
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)

data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions ' + str(data_unseen.shape))

# + {"colab_type": "text", "id": "GlddJmh74Eh4", "cell_type": "markdown"}
# # 5.0 Setting up Environment in PyCaret

# + {"colab_type": "text", "id": "bHjLZoXc4Eh-", "cell_type": "markdown"}
# In the previous tutorial __[Binary Classification (CLF101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Beginner%20-%20%20CLF101.ipynb)__ we learned how to initialize the environment in pycaret using `setup()`. No additional parameters were passed in our last example as we did not perform any pre-processing steps. In this example we will take it to the next level by customizing the pre-processing pipeline using `setup()`. Let's look at how to implement all the steps discussed in section 4 above.

# + {"colab": {}, "colab_type": "code", "id": "x34RqY6W4EiC"}
from pycaret.classification import *

# + {"colab": {}, "colab_type": "code", "id": "KBqDTo754EiR", "outputId": "f89b4933-30ec-43ea-ca35-81abb12a977f"}
exp_clf102 = setup(data = data, target = 'default', session_id=123,
                  normalize = True, 
                  transformation = True, 
                  ignore_low_variance = True,
                  remove_multicollinearity = True, multicollinearity_threshold = 0.95,
                  bin_numeric_features = ['LIMIT_BAL', 'AGE'],
                  group_features = [['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'],
                                   ['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']],
                  log_experiment = True, experiment_name = 'credit1')

# + {"colab_type": "text", "id": "zz6rype14Eie", "cell_type": "markdown"}
# Note that this is the same setup grid that was shown in __[Binary Classification Tutorial (CLF101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Beginner%20-%20%20CLF101.ipynb)__. The only difference here is the customization parameters that were passed to `setup()` are now set to `True`. Also notice that the `session_id` is the same as the one used in the beginner tutorial, which means that the effect of randomization is completely isolated. Any improvements we see in this experiment are solely due to the pre-processing steps taken in `setup()` or any other modeling techniques used in later sections of this tutorial.
#
# Another difference you may have noticed is the `log_experiment` and `experiment_name` parameter we have used within `setup`. This is to log all the modeling activity in this experiment. You will see at the end of this notebook on how you can benefit from this functionality.

# + {"colab_type": "text", "id": "B4o2brP84Eih", "cell_type": "markdown"}
# # 6.0 Comparing All Models

# + {"colab_type": "text", "id": "dZ5LGtuL4Eil", "cell_type": "markdown"}
# Similar to __[Binary Classification Tutorial (CLF101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Beginner%20-%20%20CLF101.ipynb)__ we will also begin this tutorial with `compare_models()`. We will then compare the below results with the last experiment.

# + {"colab": {}, "colab_type": "code", "id": "ARFsODlw4Eiq", "outputId": "09c08c76-9099-425b-97eb-9bf5044805da"}
top3 = compare_models(n_select = 3)
# -

# Notice that we have used `n_select` parameter within `compare_models`. In last tutorial you have seen that `compare_models` by default returns the best performing model (single model based on default sort order). However you can use `n_select` parameter to return top N models. In this example `compare_models` has returned Top 3 models.

type(top3)

print(top3)

# + {"colab_type": "text", "id": "JUQyii3p4Ei9", "cell_type": "markdown"}
# For the purposes of comparison we will use the `AUC` score. Notice how drastically a few of the algorithms have improved after we performed the pre-processing in `setup()`. 
# - Logistic Regression AUC improved from `0.6410` to `0.6957`
# - Naives Bayes AUC improved from `0.6442` to `0.6724`
# - K Nearest Neighbors AUC improved from `0.5939` to `0.6173`
#
# To see results for all of the models from the previous tutorial refer to Section 7 in __[Binary Classification Tutorial (CLF101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Beginner%20-%20%20CLF101.ipynb)__. <br/>

# + {"colab_type": "text", "id": "QxHq04mC4EjB", "cell_type": "markdown"}
# # 7.0 Create a Model

# + {"colab_type": "text", "id": "8Psn3Lc34EjF", "cell_type": "markdown"}
# In the previous tutorial __[Binary Classification (CLF101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Beginner%20-%20%20CLF101.ipynb)__ we learned how to create a model using the `create_model()` function. Now we will learn about a few other parameters that may come in handy. In this section, we will create all models using 5 fold cross validation. Notice how the `fold` parameter is passed inside `create_model()` to achieve this.

# + {"colab_type": "text", "id": "ft5E6kE24EjJ", "cell_type": "markdown"}
# ### 7.1 Create Model (with 5 fold CV)

# + {"colab": {}, "colab_type": "code", "id": "Mt7h_Ds24EjN", "outputId": "79a6681f-53b0-4446-a8d5-b57d725b0bbc"}
dt = create_model('dt', fold = 5)

# + {"colab_type": "text", "id": "7EBE1vve4EjZ", "cell_type": "markdown"}
# ### 7.2 Create Model (Round Metrics to 2 decimals points)

# + {"colab": {}, "colab_type": "code", "id": "m-QLALWP4Ejd", "outputId": "426f9d08-102b-4b65-b8ff-b201ba44d3c0"}
rf = create_model('rf', round = 2)

# + {"colab_type": "text", "id": "89GKYflR4Ejt", "cell_type": "markdown"}
# Notice how passing the `round` parameter inside `create_model()` has rounded the evaluation metrics to 2 decimals.

# + {"colab_type": "text", "id": "n_Hp176K4Ejw", "cell_type": "markdown"}
# # 8.0 Tune a Model

# + {"colab_type": "text", "id": "KRkuk-YS4Ej1", "cell_type": "markdown"}
# In the previous tutorial __[Binary Classification (CLF101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Beginner%20-%20%20CLF101.ipynb)__ we learned how to automatically tune the hyperparameters of a model using pre-defined grids. Here we will introduce the use of the `optimize` parameter in `tune_model()` which can be thought of as an objective function. In `pycaret.classification` all hyperparameter tuning is set to optimize for `Accuracy` by default which can be changed using the `optimize` parameter. See the example below:

# + {"colab": {}, "colab_type": "code", "id": "PYn9xux-4Ej7", "outputId": "19de0001-2b95-42c5-bc5c-517e3ed909ac"}
tuned_rf = tune_model(rf, optimize = 'AUC')

# + {"colab": {}, "colab_type": "code", "id": "MtW1pv2c4EkI", "outputId": "a342070d-b3cc-4eef-9e7e-ebac7315c893"}
tuned_rf2 = tune_model(rf, optimize = 'Recall')

# + {"colab_type": "text", "id": "Pa3Us5Bc4EkU", "cell_type": "markdown"}
# Notice how the results of tuning Random Forest Classifiers differ with the `optimize` param. In `tuned_rf` we optimized `AUC` resulting in `0.7691` in `AUC` and `0.6446` in `Recall`. However, in `tuned_rf2` when we set the `optimize` parameter to `Recall`, it actually resulted in a better model in terms of `Recall`. However, `AUC` was compromised. Observe the differences between the hyperparameters of `tuned_rf` and `tuned_rf2` below:

# + {"colab": {}, "colab_type": "code", "id": "R0V597844EkX"}
plot_model(tuned_rf, plot = 'parameter')

# + {"colab": {}, "colab_type": "code", "id": "8a1AxLAe4Ekj"}
plot_model(tuned_rf2, plot = 'parameter')

# + {"colab_type": "text", "id": "eXWxVxQb4Ek5", "cell_type": "markdown"}
# # 9.0 Ensemble a Model

# + {"colab_type": "text", "id": "8HawXCoP4Ek9", "cell_type": "markdown"}
# Ensembling is a common machine learning technique used to improve the performance of models (mostly tree based). There are various techniques for ensembling that we will cover in this section. These include Bagging and Boosting __[(Read More)](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205)__. We will use the `ensemble_model()` function in PyCaret which ensembles the trained base estimators using the method defined in the `method` parameter.

# + {"colab": {}, "colab_type": "code", "id": "JeYoFXe94ElA", "outputId": "9e35fb8e-e52f-4e6a-d7b2-6a498110e2b7"}
# lets create a simple decision tree model that we will use for ensembling 
dt = create_model('dt')

# + {"colab_type": "text", "id": "7e_JvjjK4ElM", "cell_type": "markdown"}
# ### 9.1 Bagging

# + {"colab": {}, "colab_type": "code", "id": "Ro9lchLm4ElR", "outputId": "fed57385-c786-4ea7-c0f6-23cd27492134"}
bagged_dt = ensemble_model(dt)

# + {"colab": {}, "colab_type": "code", "id": "haFEnRB74Ele", "outputId": "ecb1fbb6-c309-41eb-926e-218bd199fb1f"}
# check the parameters of bagged_dt
print(bagged_dt)

# + {"colab_type": "text", "id": "JcMBA-_H4Elo", "cell_type": "markdown"}
# Notice how ensembling has improved the `AUC` of Decision Tree Classifier. In the above example we have used the default parameters of `ensemble_model()` which uses the `Bagging` method. Let's try `Boosting` by changing the `method` parameter in `ensemble_model()`. See example below: 

# + {"colab_type": "text", "id": "oz-x8Ms84Elr", "cell_type": "markdown"}
# ### 9.2 Boosting

# + {"colab": {}, "colab_type": "code", "id": "y1jVQf_q4Elv", "outputId": "a5ab5675-85cd-419b-8876-5aa27e5e249e"}
boosted_dt = ensemble_model(dt, method = 'Boosting')

# + {"colab_type": "text", "id": "7Ou4NDKb4El9", "cell_type": "markdown"}
# Notice how easy it is to ensemble models in PyCaret. By simply changing the `method` parameter you can do bagging or boosting which would otherwise have taken multiple lines of code. Note that `ensemble_model()` will by default build `10` estimators. This can be changed using the `n_estimators` parameter. Increasing the number of estimators can sometimes improve results. See an example below:

# + {"colab": {}, "colab_type": "code", "id": "GIiqIzQi4EmB", "outputId": "b45686c4-63e1-442f-bd18-ccc716753adf"}
bagged_dt2 = ensemble_model(dt, n_estimators=50)

# + {"colab_type": "text", "id": "s95TgoAT4EmL", "cell_type": "markdown"}
# Notice how increasing the n_estimators parameter has improved the performance of ensembled model. The bagged_dt model with the default `10` estimators resulted in an AUC of `0.7282` whereas in bagged_dt2 where `n_estimators = 50` the AUC improved to `0.7503`.

# + {"colab_type": "text", "id": "pDrdubjb4Emm", "cell_type": "markdown"}
# ### 9.3 Blending

# + {"colab_type": "text", "id": "G-Mub1qC4Emp", "cell_type": "markdown"}
# Blending is another common technique for ensembling that can be used in PyCaret. It uses predictions from multiple models to generate a final set of predictions using voting / majority consensus from all of the models passed in the `estimator_list` parameter. The `method` parameter can be used to define the type of voting. When set to `hard`, it uses labels for majority rule voting. When set to `soft` it uses the sum of predicted probabilities instead of the label. Default value of method is set to `auto` which means it tries to use `soft` method and fall back to `hard` if the former is not supported. Let's see an example below:
# -

# train individual models to blend
lightgbm = create_model('lightgbm', verbose = False)
dt = create_model('dt', verbose = False)
lr = create_model('lr', verbose = False)

# blend individual models
blend_soft = blend_models(estimator_list = [lightgbm, dt, lr], method = 'soft')

# blend individual models
blend_hard = blend_models(estimator_list = [lightgbm, dt, lr], method = 'hard')

# blend top3 models from compare_models
blender_top3 = blend_models(top3)

print(blender_top3.estimators_)

# + {"colab_type": "text", "id": "HZw7sl394EoG", "cell_type": "markdown"}
# ### 9.4 Stacking

# + {"colab_type": "text", "id": "1d2f_8no4EoM", "cell_type": "markdown"}
# Stacking is another popular technique for ensembling but is less commonly implemented due to practical difficulties. Stacking is an ensemble learning technique that combines multiple models via a meta-model. Another way to think about stacking is that multiple models are trained to predict the outcome and a meta-model is created that uses the predictions from those models as an input along with the original features. The implementation of `stack_models()` is based on Wolpert, D. H. (1992b). Stacked generalization __[(Read More)](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)__. 
#
# Let's see an example below using the top 3 models we have obtained from `compare_models`:

# + {"colab": {}, "colab_type": "code", "id": "9J975uiz4EoO", "outputId": "98f0c6bc-384a-457a-e674-ead66fff0bc1"}
stack_soft = stack_models(top3)

# + {"colab": {}, "colab_type": "code", "id": "WWInYCWI4Eos", "outputId": "68f44383-c03f-4793-9835-5e7b644bd3d2"}
xgboost = create_model('xgboost')
stack_soft2 = stack_models(top3, meta_model=xgboost)

# + {"colab_type": "text", "id": "Yom7721D4Eo0", "cell_type": "markdown"}
# Selecting which `method` and `models` to use in stacking depends on the statistical properties of the dataset. Experimenting with different models and methods is the best way to find out which configuration will work best. However as a general rule of thumb, models with strong yet diverse performance tend to improve results when used in stacking. </br>
#
# Before we wrap up this section, there is another parameter in `stack_models()` that we haven't seen yet called `restack`. This parameter controls the ability to expose the raw data to the meta model. When set to `True`, it exposes the raw data to the meta model along with all the predictions of the base level models. By default it is set to `True`.

# + {"colab_type": "text", "id": "bhtJHidg4EpR", "cell_type": "markdown"}
# # 10.0 Model Calibration

# + {"colab_type": "text", "id": "91jxBYw-4EpV", "cell_type": "markdown"}
# When performing classification you often not only want to predict the class label (outcome such as 0 or 1), but also obtain the probability of the respective outcome which provides a level of confidence on the prediction. Some models can give you poor estimates of the class probabilities and some do not even support probability prediction. Well calibrated classifiers are probabilistic and provide outputs in the form of probabilities that can be directly interpreted as a confidence level. PyCaret allows you to calibrate the probabilities of a given model through the `calibrate_model()` function. See an example below:

# + {"colab": {}, "colab_type": "code", "id": "-_SM3psW4Epi", "outputId": "0589cc46-384f-4461-8ad1-8b857c271937"}
rf = create_model('rf')

# + {"colab": {}, "colab_type": "code", "id": "NS2vjyxl4Ep4", "outputId": "45762432-0575-49e2-e5a9-712b02d4d916"}
plot_model(rf, plot='calibration')

# + {"colab": {}, "colab_type": "code", "id": "mVB60azT4EqJ", "outputId": "ec5f17d2-267e-44ad-d828-40d9b91b70f1"}
calibrated_rf = calibrate_model(rf)

# + {"colab": {}, "colab_type": "code", "id": "hE6y4LlG4Eqb", "outputId": "25c6a67d-da1c-465d-ab8f-9953664adda8"}
plot_model(calibrated_rf, plot='calibration')

# + {"colab_type": "text", "id": "k9pb2gWR4Eqw", "cell_type": "markdown"}
# Notice how different the above 2 plots look. One is before calibration and one is after. A perfectly calibrated classifier will follow the black dotted line in the above plots. Not only is `calibrated_rf` better calibrated but the `AUC` has also improved from `0.7591` to `0.7646`. By default, `calibrate_model()` uses the `sigmoid` method which corresponds to Platt's approach. The other available method is `isotonic` which is a non-parametric approach. See an example of calibration using the `isotonic` method below:  

# + {"colab": {}, "colab_type": "code", "id": "I0WKZOK14Eqy", "outputId": "a7144f54-0260-470b-dd0a-0ebb512894d4"}
calibrated_rf_isotonic = calibrate_model(rf, method = 'isotonic')

# + {"colab": {}, "colab_type": "code", "id": "HIoNaLU14Eq_", "outputId": "05551e6f-f1e9-43c7-dcc2-96617424dcab"}
plot_model(calibrated_rf_isotonic, plot='calibration')
# -

# # 11.0 Experiment Logging

# PyCaret >= 2.0 embeds MLflow Tracking component as a backend API and UI for logging parameters, code versions, metrics, and output files when running your machine learning code and for later visualizing the results. To log your experiments in pycaret simply use `log_experiment` and `experiment_name` parameter in the `setup` function, as we did in this example.
#
# You can start the UI on `https://localhost:5000`. Simply initiate the MLFlow server from command line or from notebook. See example below:

# to start the MLFlow server from notebook:
# !mlflow ui 

# ### Open localhost:5000 on your browser (below is example of how UI looks like)
# ![title](https://i2.wp.com/pycaret.org/wp-content/uploads/2020/07/classification_mlflow_ui.png?resize=1080%2C508&ssl=1)

# + {"colab_type": "text", "id": "uWKoxcgK4EtG", "cell_type": "markdown"}
# # 12.0 Wrap-up / Next Steps?

# + {"colab_type": "text", "id": "RnmdZxsq4EtI", "cell_type": "markdown"}
# We have covered a lot of new concepts in this tutorial. Most importantly we have seen how to use exploratory data analysis to customize a pipeline in `setup()` which has improved the results considerably when compared to what we saw earlier in __[Binary Classification Tutorial (CLF101) - Level Beginner](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Beginner%20-%20%20CLF101.ipynb)__. We have also learned how to perform and tune ensembling in PyCaret.
#
# There are however a few more advanced things to cover in `pycaret.classification` which include defining and optimizing custom cost functions, interpreting more complex tree based models using shapley values, advanced ensembling techniques such as multiple layer stacknet and more pre-processing pipeline methods. We will cover all of this in our next and final tutorial in the `pycaret.classification` series. 
#
# See you at the next tutorial. Follow the link to __[Binary Classification Tutorial (CLF103) - Level Expert](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Expert%20-%20CLF103.ipynb)__
