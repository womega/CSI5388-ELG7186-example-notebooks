#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the imbalanced-learn library
# install it with ! pip install imbalanced-learn
import imblearn


# In[2]:


# import the scikit-learn library
# install it with ! pip install scikit-learn
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_validate


# In[3]:


# import other necessary libraries
import numpy as np
import pandas as pd
from collections import Counter


# In[4]:


# function to append the averages and
# standard deviations of scores,
# row 11 are the averages and
# row 12 are the stds
def append_avgs(dict):
    avgs = []
    stds = []
    for i in dict:
        avg = np.average(dict[i])
        std = np.std(dict[i])
        avgs.append(avg)
        stds.append(std)
    df = pd.DataFrame(dict)
    df.loc[11] = avgs
    df.loc[12] = stds
    return df


# In[5]:


# create an imbalaced toy dataset with 10000 samples and 10 features
# class 0: 7500 samples, class 1: 2500 samples 
X, y = make_classification(n_samples=10000, weights=[0.75], n_features = 10,
                           flip_y=0, random_state=42)

# define the Decision Tree Classifier model
model = sklearn.tree.DecisionTreeClassifier()

# define crossvalidation strategy
# this will perform a stratified 10-fold crossvalidation
# parameter n_splits defines the number of folds
cv = StratifiedKFold(n_splits=10)

# define the scoring dictionary to evaluate the crossvalidation
scoring = {'f1': 'f1', 'precision': 'precision', 'accuracy': 'accuracy',
           'recall': 'recall', 'roc_auc': 'roc_auc'}


# # Cross-validation without sampling

# In[7]:


# crossvalidate on original data set
scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)


# **Row 11 contains average scores and 12 contains the standard deviation**

# In[8]:


# use previously defined function "append_avgs" to append averages and stds
scores = append_avgs(scores)
scores


# On average the DTC scores are **F-1 0.712143**; **Precision 0.833571**; **Accuracy 0.974000**; **Recall 0.660000** and **AUC 0.825263**

# # Cross-validation with sampling

# **We will use _SMOTE_ and _Random Undersampling_ as sampling techniques**

# In[9]:


# define SMOTE to oversample minority class to contain
# half the amount of samples in the majority class
# by setting sampling_strategy to 0.5
smote = imblearn.over_sampling.SMOTE(sampling_strategy = 0.5)

# define RandomUnderSampler to undersample majority class
# to contain twice the amount of samples in the minority class
# by setting sampling_strategy to 0.5
under = imblearn.under_sampling.RandomUnderSampler(sampling_strategy = 0.5)


# ## Cross validation with sampling -- DONE WRONG

# ### Cross-validation with oversampling -- DONE WRONG

# In[10]:


# oversample minority class
X_smote, y_smote = smote.fit_resample(X, y)


# In[11]:


print("Distribution in original data set", Counter(y))
print("Distribution in data set after SMOTE", Counter(y_smote))


# In[12]:


# crossvalidate on oversampled data set using the crossvalidation technique defined in "cv"
wrong_scores_o = cross_validate(model, X_smote, y_smote, scoring=scoring, cv=cv, n_jobs=-1)


# In[13]:


# use previously defined function "append_avgs" to append averages and stds
wrong_scores_o = append_avgs(wrong_scores_o)
wrong_scores_o


# ### Cross-validation with undersampling -- DONE WRONG

# In[14]:


# undersample majority class
X_under, y_under = under.fit_resample(X, y)


# In[15]:


print("Distribution in original data set", Counter(y))
print("Distribution in data set after undersampling", Counter(y_under))


# In[16]:


# crossvalidate on undersampled data set using the crossvalidation technique defined in "cv"
wrong_scores_u = cross_validate(model, X_under, y_under, scoring=scoring, cv=cv, n_jobs=-1)


# In[17]:


# use previously defined function "append_avgs" to append averages and stds
wrong_scores_u = append_avgs(wrong_scores_u)
wrong_scores_u


# ## Cross validation with sampling -- DONE RIGHT

# ### Cross-validation with oversampling -- DONE RIGHT

# In[18]:


# define the SMOTE oversampling pipeline
smote_steps = [('over', smote), ('model', model)]
smote_pipeline = imblearn.pipeline.Pipeline(steps=smote_steps)


# This Pipeline first oversamples the training dataset with SMOTE then fits the model.

# In[19]:


# show the SMOTE pipeline
print(smote_pipeline)


# In[20]:


# evaluate the SMOTE pipeline using the crossvalidation technique defined in cv
smote_scores = cross_validate(smote_pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)


# In[21]:


# use previously defined function "append_avgs" to append averages and stds
smote_scores = append_avgs(smote_scores)
smote_scores


# ### Cross-validation with undersampling -- DONE RIGHT

# In[22]:


# define RandomUnderSampler undersampling pipeline
under_steps = [('under', under), ('model', model)]
under_pipeline = imblearn.pipeline.Pipeline(steps=under_steps)


# This Pipeline first undersamples the training dataset with RandomUnderSampler then fits the model.

# In[23]:


# show outher RandomUnderSampler pipeline
print(under_pipeline)


# In[24]:


# evaluate the RandomUnderSampler pipeline using the crossvalidation technique defined in cv
under_scores = cross_validate(under_pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)


# In[25]:


# use previously defined function "append_avgs" to append averages and stds
under_scores = append_avgs(under_scores)
under_scores


# # Compare scores

# In[26]:


# create an empty dataframe to compare the scores
df = pd.DataFrame()

# add the average scores of each sampling strategy
# to the empty dataframe
df = df.append([scores.loc[11], wrong_scores_o.loc[11], wrong_scores_u.loc[11],
           smote_scores.loc[11], under_scores.loc[11]]).reset_index(drop=True)

# rename the indices
df = df.rename(index={0:'NO SAMPLING', 1:'WRONG SMOTE', 2:'WRONG UNDERSAMPLING',
                 3: 'RIGHT SMOTE', 4:'RIGHT UNDERSAMPLING'})

df


# In[ ]:




