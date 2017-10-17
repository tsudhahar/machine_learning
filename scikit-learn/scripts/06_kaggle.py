# # Advanced Machine Learning Techniques

# ## Agenda
# 
# 1. Reading in the Kaggle data and adding features
# 2. Using a **`Pipeline`** for proper cross-validation
# 3. Combining **`GridSearchCV`** with **`Pipeline`**
# 4. Efficiently searching for tuning parameters using **`RandomizedSearchCV`**
# 5. Adding features to a document-term matrix (using SciPy)
# 6. Adding features to a document-term matrix (using **`FeatureUnion`**)
# 7. Ensembling models
# 8. Locating groups of similar cuisines
# 9. Model stacking

# for Python 2: use print only as a function
from __future__ import print_function


# ## Part 1: Reading in the Kaggle data and adding features
# 
# - Our goal is to predict the **cuisine** of a recipe, given its **ingredients**.
# - **Feature engineering** is the process through which you create features that don't natively exist in the dataset.

import pandas as pd
import numpy as np


# define a function that accepts a DataFrame and adds new features
def make_features(df):
    
    # number of ingredients
    df['num_ingredients'] = df.ingredients.apply(len)
    
    # mean length of ingredient names
    df['ingredient_length'] = df.ingredients.apply(lambda x: np.mean([len(item) for item in x]))
    
    # string representation of the ingredient list
    df['ingredients_str'] = df.ingredients.astype(str)
    
    return df


# create the same features in the training data and the new data
train = make_features(pd.read_json('../data/train.json'))
new = make_features(pd.read_json('../data/test.json'))


train.head()


train.shape


new.head()


new.shape


# ## Part 2: Using a `Pipeline` for proper cross-validation

# define X and y
X = train.ingredients_str
y = train.cuisine


# X is just a Series of strings
X.head()


# replace the regex pattern that is used for tokenization
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(token_pattern=r"'([a-z ]+)'")


# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# [make_pipeline documentation](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)

# create a pipeline of vectorization and Naive Bayes
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(vect, nb)


# examine the pipeline steps
pipe.steps


# **Proper cross-validation:**
# 
# - By passing our pipeline to **`cross_val_score`**, features will be created from **`X`** (via **`CountVectorizer`**) within each fold of cross-validation.
# - This process simulates the real world, in which your out-of-sample data will contain **features that were not seen** during model training.

# cross-validate the entire pipeline
from sklearn.cross_validation import cross_val_score
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()


# ## Part 3: Combining `GridSearchCV` with `Pipeline`
# 
# - We use **`GridSearchCV`** to locate optimal tuning parameters by performing an "exhaustive grid search" of different parameter combinations, searching for the combination that has the best cross-validated accuracy.
# - By passing a **`Pipeline`** to **`GridSearchCV`** (instead of just a model), we can search tuning parameters for both the vectorizer and the model.

# pipeline steps are automatically assigned names by make_pipeline
pipe.named_steps.keys()


# create a grid of parameters to search (and specify the pipeline step along with the parameter)
param_grid = {}
param_grid['countvectorizer__token_pattern'] = [r"\b\w\w+\b", r"'([a-z ]+)'"]
param_grid['multinomialnb__alpha'] = [0.5, 1]
param_grid


# [GridSearchCV documentation](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html)

# pass the pipeline (instead of the model) to GridSearchCV
from sklearn.grid_search import GridSearchCV
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')


# run the grid search
grid.fit(X, y)


# examine the score for each combination of parameters
grid.grid_scores_


# print the single best score and parameters that produced that score
print(grid.best_score_)
print(grid.best_params_)


# ## Part 4: Efficiently searching for tuning parameters using `RandomizedSearchCV`
# 
# - When there are many parameters to tune, searching all possible combinations of parameter values may be **computationally infeasible**.
# - **`RandomizedSearchCV`** searches a sample of the parameter values, and you control the computational "budget".
# 
# [RandomizedSearchCV documentation](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.RandomizedSearchCV.html)

from sklearn.grid_search import RandomizedSearchCV


# [scipy.stats documentation](http://docs.scipy.org/doc/scipy/reference/stats.html)

# for any continuous parameters, specify a distribution instead of a list of options
import scipy as sp
param_grid = {}
param_grid['countvectorizer__token_pattern'] = [r"\b\w\w+\b", r"'([a-z ]+)'"]
param_grid['countvectorizer__min_df'] = [1, 2, 3]
param_grid['multinomialnb__alpha'] = sp.stats.uniform(scale=1)
param_grid


# set a random seed for sp.stats.uniform
np.random.seed(1)


# additional parameters are n_iter (number of searches) and random_state
rand = RandomizedSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_iter=5, random_state=1)


# run the randomized search
rand.fit(X, y)


rand.grid_scores_


print(rand.best_score_)
print(rand.best_params_)


# ### Making predictions for new data

# define X_new as the ingredient text
X_new = new.ingredients_str


# print the best model found by RandomizedSearchCV
rand.best_estimator_


# RandomizedSearchCV/GridSearchCV automatically refit the best model with the entire dataset, and can be used to make predictions
new_pred_class_rand = rand.predict(X_new)
new_pred_class_rand


# create a submission file (score: 0.75342)
pd.DataFrame({'id':new.id, 'cuisine':new_pred_class_rand}).set_index('id').to_csv('sub3.csv')


# ## Part 5: Adding features to a document-term matrix (using SciPy)
# 
# - So far, we've trained models on either the **document-term matrix** or the **manually created features**, but not both.
# - To train a model on both types of features, we need to **combine them into a single feature matrix**.
# - Because one of the matrices is **sparse** and the other is **dense**, the easiest way to combine them is by using SciPy.

# create a document-term matrix from all of the training data
X_dtm = vect.fit_transform(X)
X_dtm.shape


type(X_dtm)


# [scipy.sparse documentation](http://docs.scipy.org/doc/scipy/reference/sparse.html)

# create a DataFrame of the manually created features
X_manual = train.loc[:, ['num_ingredients', 'ingredient_length']]
X_manual.shape


# create a sparse matrix from the DataFrame
X_manual_sparse = sp.sparse.csr_matrix(X_manual)
type(X_manual_sparse)


# combine the two sparse matrices
X_dtm_manual = sp.sparse.hstack([X_dtm, X_manual_sparse])
X_dtm_manual.shape


# - This was a relatively easy process.
# - However, it does not allow us to do **proper cross-validation**, and it doesn't integrate well with the rest of the **scikit-learn workflow**.

# ## Part 6: Adding features to a document-term matrix (using `FeatureUnion`)
# 
# - Below is an alternative process that does allow for proper cross-validation, and does integrate well with the scikit-learn workflow.
# - To use this process, we have to learn about transformers, **`FunctionTransformer`**, and **`FeatureUnion`**.

# ### What are "transformers"?
# 
# Transformer objects provide a `transform` method in order to perform **data transformations**. Here are a few examples:
# 
# - **`CountVectorizer`**
#     - `fit` learns the vocabulary
#     - `transform` creates a document-term matrix using the vocabulary
# - **`Imputer`**
#     - `fit` learns the value to impute
#     - `transform` fills in missing entries using the imputation value
# - **`StandardScaler`**
#     - `fit` learns the mean and scale of each feature
#     - `transform` standardizes the features using the mean and scale
# - **`HashingVectorizer`**
#     - `fit` is not used, and thus it is known as a "stateless" transformer
#     - `transform` creates the document-term matrix using a hash of the token

# ### Converting a function into a transformer

# define a function that accepts a DataFrame returns the manually created features
def get_manual(df):
    return df.loc[:, ['num_ingredients', 'ingredient_length']]


get_manual(train).head()


# [FunctionTransformer documentation](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) (new in 0.17)

from sklearn.preprocessing import FunctionTransformer


# create a stateless transformer from the get_manual function
get_manual_ft = FunctionTransformer(get_manual, validate=False)
type(get_manual_ft)


# execute the function using the transform method
get_manual_ft.transform(train).head()


# define a function that accepts a DataFrame returns the ingredients string
def get_text(df):
    return df.ingredients_str


# create and test another transformer
get_text_ft = FunctionTransformer(get_text, validate=False)
get_text_ft.transform(train).head()


# ### Combining feature extraction steps
# 
# - **`FeatureUnion`** applies a list of transformers in parallel to the input data (not sequentially), then **concatenates the results**.
# - This is useful for combining several feature extraction mechanisms into a single transformer.
# 
# ![Pipeline versus FeatureUnion](06_pipeline_versus_featureunion.jpg)

# [make_union documentation](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_union.html)

from sklearn.pipeline import make_union


# create a document-term matrix from all of the training data
X_dtm = vect.fit_transform(X)
X_dtm.shape


# this is identical to a FeatureUnion with just one transformer
union = make_union(vect)
X_dtm = union.fit_transform(X)
X_dtm.shape


# try to add a second transformer to the Feature Union (what's wrong with this?)
# union = make_union(vect, get_manual_ft)
# X_dtm_manual = union.fit_transform(X)


# properly combine the transformers into a FeatureUnion
union = make_union(make_pipeline(get_text_ft, vect), get_manual_ft)
X_dtm_manual = union.fit_transform(train)
X_dtm_manual.shape


# ![Pipeline in a FeatureUnion](06_pipeline_in_a_featureunion.jpg)

# ### Cross-validation

# slightly improper cross-validation
cross_val_score(nb, X_dtm_manual, y, cv=5, scoring='accuracy').mean()


# create a pipeline of the FeatureUnion and Naive Bayes
pipe = make_pipeline(union, nb)


# properly cross-validate the entire pipeline (and pass it the entire DataFrame)
cross_val_score(pipe, train, y, cv=5, scoring='accuracy').mean()


# ### Alternative way to specify `Pipeline` and `FeatureUnion`

# reminder of how we created the pipeline
union = make_union(make_pipeline(get_text_ft, vect), get_manual_ft)
pipe = make_pipeline(union, nb)


# [Pipeline documentation](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and [FeatureUnion documentation](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)

# duplicate the pipeline structure without using make_pipeline or make_union
from sklearn.pipeline import Pipeline, FeatureUnion
pipe = Pipeline([
    ('featureunion', FeatureUnion([
            ('pipeline', Pipeline([
                    ('functiontransformer', get_text_ft),
                    ('countvectorizer', vect)
                    ])),
            ('functiontransformer', get_manual_ft)
        ])),
    ('multinomialnb', nb)
])


# ### Grid search of a nested `Pipeline`

# examine the pipeline steps
pipe.steps


# create a grid of parameters to search (and specify the pipeline step along with the parameter)
param_grid = {}
param_grid['featureunion__pipeline__countvectorizer__token_pattern'] = [r"\b\w\w+\b", r"'([a-z ]+)'"]
param_grid['multinomialnb__alpha'] = [0.5, 1]
param_grid


grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')


grid.fit(train, y)


print(grid.best_score_)
print(grid.best_params_)


# ## Part 7: Ensembling models
# 
# Rather than combining features into a single feature matrix and training a single model, we can instead create separate models and "ensemble" them.

# ### What is ensembling?
# 
# Ensemble learning (or "ensembling") is the process of combining several predictive models in order to produce a combined model that is **better than any individual model**.
# 
# - **Regression:** average the predictions made by the individual models
# - **Classification:** let the models "vote" and use the most common prediction, or average the predicted probabilities
# 
# For ensembling to work well, the models must have the following characteristics:
# 
# - **Accurate:** they outperform the null model
# - **Independent:** their predictions are generated using different "processes", such as:
#     - different types of models
#     - different features
#     - different tuning parameters
# 
# **The big idea:** If you have a collection of individually imperfect (and independent) models, the "one-off" mistakes made by each model are probably not going to be made by the rest of the models, and thus the mistakes will be discarded when averaging the models.
# 
# **Note:** There are also models that have built-in ensembling, such as Random Forests.

# ### Model 1: KNN model using only manually created features

# define X and y
feature_cols = ['num_ingredients', 'ingredient_length']
X = train[feature_cols]
y = train.cuisine


# use KNN with K=800
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=800)


# train KNN on all of the training data
knn.fit(X, y)


# define X_new as the manually created features
X_new = new[feature_cols]


# calculate predicted probabilities of class membership for the new data
new_pred_prob_knn = knn.predict_proba(X_new)
new_pred_prob_knn.shape


# print predicted probabilities for the first row only
new_pred_prob_knn[0, :]


# display classes with probabilities
zip(knn.classes_, new_pred_prob_knn[0, :])


# predicted probabilities will sum to 1 for each row
new_pred_prob_knn[0, :].sum()


# ### Model 2: Naive Bayes model using only text features

# print the best model found by RandomizedSearchCV
rand.best_estimator_


# define X_new as the ingredient text
X_new = new.ingredients_str


# calculate predicted probabilities of class membership for the new data
new_pred_prob_rand = rand.predict_proba(X_new)
new_pred_prob_rand.shape


# print predicted probabilities for the first row only
new_pred_prob_rand[0, :]


# ### Ensembling models 1 and 2

# calculate the mean of the predicted probabilities for the first row
(new_pred_prob_knn[0, :] + new_pred_prob_rand[0, :]) / 2


# calculate the mean of the predicted probabilities for all rows
new_pred_prob = pd.DataFrame((new_pred_prob_knn + new_pred_prob_rand) / 2, columns=knn.classes_)
new_pred_prob.head()


# for each row, find the column with the highest predicted probability
new_pred_class = new_pred_prob.apply(np.argmax, axis=1)
new_pred_class.head()


# create a submission file (score: 0.75241)
pd.DataFrame({'id':new.id, 'cuisine':new_pred_class}).set_index('id').to_csv('sub4.csv')


# **Note:** [VotingClassifier](http://scikit-learn.org/stable/modules/ensemble.html#votingclassifier) (new in 0.17) makes it easier to ensemble classifiers, though it is limited to the case in which all of the classifiers are fit to the same data.

# ## Part 8: Locating groups of similar cuisines

# for each cuisine, combine all of the recipes into a single string
cuisine_ingredients = train.groupby('cuisine').ingredients_str.sum()
cuisine_ingredients


# examine the brazilian ingredients
cuisine_ingredients['brazilian'][0:500]


# confirm that they match the brazilian recipes
train.loc[train.cuisine=='brazilian', 'ingredients_str'].head()


# create a document-term matrix from cuisine_ingredients
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
cuisine_dtm = vect.fit_transform(cuisine_ingredients)
cuisine_dtm.shape


# [How to calculate document similarity](http://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity/12128777#12128777) (Stack Overflow)

# calculate the cosine similarity between each cuisine and all other cuisines
from sklearn import metrics
cuisine_similarity = []
for idx in range(cuisine_dtm.shape[0]):
    similarity = metrics.pairwise.linear_kernel(cuisine_dtm[idx, :], cuisine_dtm).flatten()
    cuisine_similarity.append(similarity)


# convert the results to a DataFrame
cuisine_list = cuisine_ingredients.index
cuisine_similarity = pd.DataFrame(cuisine_similarity, index=cuisine_list, columns=cuisine_list)
cuisine_similarity


# display the similarities as a heatmap
import seaborn as sns
sns.heatmap(cuisine_similarity)


# hand-selected cuisine groups
group_1 = ['chinese', 'filipino', 'japanese', 'korean', 'thai', 'vietnamese']
group_2 = ['british', 'french', 'irish', 'russian', 'southern_us']
group_3 = ['greek', 'italian', 'moroccan', 'spanish']
group_4 = ['brazilian', 'cajun_creole', 'indian', 'jamaican', 'mexican']


# ## Part 9: Model stacking
# 
# - The term "model stacking" is used any time there are **multiple "levels" of models**, in which the outputs from one level are used as inputs to another level.
# - In this case, we will create one model that predicts the **cuisine group** for a recipe. Within each of the four groups, we will create another model that predicts the actual **cuisine**.
# - Our theory is that each of these five models may need to be **tuned differently** for maximum accuracy, but will ultimately result in a process that is more accurate than a single-level model.

# create a dictionary that maps each cuisine to its group number
cuisines = group_1 + group_2 + group_3 + group_4
group_numbers = [1]*len(group_1) + [2]*len(group_2) + [3]*len(group_3) + [4]*len(group_4)
cuisine_to_group = dict(zip(cuisines, group_numbers))
cuisine_to_group


# map the cuisines to their group numbers
train['group'] = train.cuisine.map(cuisine_to_group)
train.head()


# confirm that all recipes were assigned a cuisine group
train.group.isnull().sum()


# calculate the cross-validated accuracy of using text to predict cuisine group
X = train.ingredients_str
y = train.group
pipe_main = make_pipeline(CountVectorizer(), MultinomialNB())
cross_val_score(pipe_main, X, y, cv=5, scoring='accuracy').mean()


# define an X and y for each cuisine group
X1 = train.loc[train.group==1, 'ingredients_str']
y1 = train.loc[train.group==1, 'cuisine']
X2 = train.loc[train.group==2, 'ingredients_str']
y2 = train.loc[train.group==2, 'cuisine']
X3 = train.loc[train.group==3, 'ingredients_str']
y3 = train.loc[train.group==3, 'cuisine']
X4 = train.loc[train.group==4, 'ingredients_str']
y4 = train.loc[train.group==4, 'cuisine']


# define a pipeline for each cuisine group
pipe_1 = make_pipeline(CountVectorizer(), MultinomialNB())
pipe_2 = make_pipeline(CountVectorizer(), MultinomialNB())
pipe_3 = make_pipeline(CountVectorizer(), MultinomialNB())
pipe_4 = make_pipeline(CountVectorizer(), MultinomialNB())


# within each cuisine group, calculate the cross-validated accuracy of using text to predict cuisine
print(cross_val_score(pipe_1, X1, y1, cv=5, scoring='accuracy').mean())
print(cross_val_score(pipe_2, X2, y2, cv=5, scoring='accuracy').mean())
print(cross_val_score(pipe_3, X3, y3, cv=5, scoring='accuracy').mean())
print(cross_val_score(pipe_4, X4, y4, cv=5, scoring='accuracy').mean())


# **Note:** Ideally, each of the five pipelines should be **individually tuned** from start to finish, including feature engineering, model selection, and parameter tuning.

# ### Making predictions for the new data

# fit each pipeline with the relevant X and y
pipe_main.fit(X, y)
pipe_1.fit(X1, y1)
pipe_2.fit(X2, y2)
pipe_3.fit(X3, y3)
pipe_4.fit(X4, y4)


# for the new data, first make cuisine group predictions
X_new = new.ingredients_str
new_pred_group = pipe_main.predict(X_new)
new_pred_group


# then within each predicted cuisine group, make cuisine predictions
new_pred_class_1 = pipe_1.predict(X_new[new_pred_group==1])
new_pred_class_2 = pipe_2.predict(X_new[new_pred_group==2])
new_pred_class_3 = pipe_3.predict(X_new[new_pred_group==3])
new_pred_class_4 = pipe_4.predict(X_new[new_pred_group==4])
print(new_pred_class_1)
print(new_pred_class_2)
print(new_pred_class_3)
print(new_pred_class_4)


# add the cuisine predictions to the DataFrame of new data
new.loc[new_pred_group==1, 'pred_class'] = new_pred_class_1
new.loc[new_pred_group==2, 'pred_class'] = new_pred_class_2
new.loc[new_pred_group==3, 'pred_class'] = new_pred_class_3
new.loc[new_pred_group==4, 'pred_class'] = new_pred_class_4


new.head()


# create a submission file (score: 0.70475)
pd.DataFrame({'id':new.id, 'cuisine':new.pred_class}).set_index('id').to_csv('sub5.csv')

