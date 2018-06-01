
# coding: utf-8

# # Capstone Project - Cuisine Categorization by the recipe of ingredients

# In[1]:

# for Python 2: use print only as a function
from __future__ import print_function


# ## Data Exploration and Adding features
# 

# In[3]:

import pandas as pd 


# In[4]:

train = pd.read_json('../data/train.json')
test = pd.read_json('../data/test.json')


# In[5]:

train.head()


# In[6]:

train.shape


# In[7]:

test.shape


# In[8]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

train['cuisine'].value_counts().plot(kind='barh')


# ##Exploratory Visualization

# Next, let me explore on the data exploration. especially on
# 
# * how many ingredients there are? Out them, find the unique ingredients.
# * how these ingredients made up of individual words?

# In[9]:

import re

total_ingredients = []Exploratory Visualization
for lst_ingredients in train.ingredients:
    total_ingredients += [ingredient.lower() for ingredient in lst_ingredients]

no_of_ingredients = len(total_ingredients)
uniq_ingredients = len(set(total_ingredients))

print(no_of_ingredients)
print(uniq_ingredients)


# In[10]:

word_split = re.compile('[,. ]+')
total_ingredient_words = []

for ingredients in total_ingredients:
    total_ingredient_words += re.split(word_split, ingredients)
    
no_of_ingredients_words = len(total_ingredient_words)
uniq_ingredients_words = len(set(total_ingredient_words))

print(no_of_ingredients_words)
print(uniq_ingredients_words)


# In[12]:

from collections import Counter
import seaborn as sns
import numpy as np

cntr = {}
for c in train['cuisine'].unique():
    cntr[c] = Counter()
    idx = (train['cuisine'] == c)
    for ingredients in train[idx]['ingredients']:
        cntr[c].update(ingredients)

common_ingredients = pd.DataFrame([[items[0] for items in cntr[c].most_common(5)] for c in cntr],
            index=[c for c in cntr],
            columns=['Most_Common_{}'.format(i) for i in range(1, 6)])

common_ingredients

counter1 = Counter(common_ingredients['Most_Common_1'])

ing_name1 = counter1.keys()
ing_count1 = counter1.values()

counter2 = Counter(common_ingredients['Most_Common_2'])
ing_name2 = counter2.keys()
ing_count2 = counter2.values()

counter3 = Counter(common_ingredients['Most_Common_3'])
ing_name3 = counter3.keys()
ing_count3 = counter3.values()

counter4 = Counter(common_ingredients['Most_Common_4'])
ing_name4 = counter4.keys()
ing_count4 = counter4.values()

fig = plt.figure(figsize=(5, 5))
#fig, ax = plt.subplots(4, 4, figsize=(40, 40))

# Plot histogram using matplotlib bar().
indexes = np.arange(len(ing_name1))
width = 0.2
plt.bar(indexes, ing_count1, width)
plt.xticks(indexes + width * 0.5, ing_name1,rotation='vertical')
plt.title('Most Common Ingredients Rank 1')
plt.show()
fig.savefig('rank1.png')

fig = plt.figure(figsize=(5, 5))
indexes = np.arange(len(ing_name2))
width = 0.2
plt.bar(indexes, ing_count2, width)
plt.xticks(indexes + width * 0.5, ing_name2,rotation='vertical')
plt.title('Most Common Ingredients Rank 2')
plt.show()
fig.savefig('rank2.png')

fig = plt.figure(figsize=(5, 5))
indexes = np.arange(len(ing_name3))
width = 0.2
plt.bar(indexes, ing_count3, width)
plt.xticks(indexes + width * 0.5, ing_name3,rotation='vertical')
plt.title('Most Common Ingredients Rank 3')
plt.show()
fig.savefig('rank3.png')

fig = plt.figure(figsize=(5, 5))
indexes = np.arange(len(ing_name4))
width = 0.2
plt.bar(indexes, ing_count4, width)
plt.xticks(indexes + width * 0.5, ing_name4,rotation='vertical')
plt.title('Most Common Ingredients Rank 4')
plt.show()
fig.savefig('rank4.png')


# In[13]:

list_ingredients = np.unique(common_ingredients.values.ravel())

train['total_ingredients'] = train['ingredients'].map(":".join)

fig, ax = plt.subplots(5, 5, figsize=(40, 40))
for ingredient, ax_idx in zip(list_ingredients, range(25)):
    indexes = train['total_ingredients'].str.contains(ingredient)
    ingredient_occur = (train[indexes]['cuisine'].value_counts() / train['cuisine'].value_counts())
    ingredient_occur.plot(kind='bar', ax=ax.ravel()[ax_idx], fontsize=10, title=ingredient)

fig.savefig('ingredient_occur.plot.png')


# ## Benchmark Model

# In[14]:

from text_unidecode import unidecode

def xform_string(str_list):
    return ", ".join([
        unidecode(str).lower()
        for str in str_list
    ])


# In[15]:

# Import train_test_split
from sklearn.cross_validation import train_test_split

train_features= train.drop('cuisine', axis = 1)
train_cuisine = pd.DataFrame(train['cuisine'])

# Split the 'features' and 'Yummly' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_features, 
                                                    train_cuisine , 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))
print (y_train.shape[0])
print (y_test.shape[0])
X_train.shape



# In[16]:

X_test.shape
#X_test.head


# In[17]:

y_train.shape
y_train.head


# In[18]:

y_test.shape


# In[19]:

from sklearn.feature_extraction.text import CountVectorizer

# Preprocessing
vector = CountVectorizer(
    preprocessor = xform_string,
    analyzer = "word",
    token_pattern = r"(?u)\b[a-z]{2,40}\b",
    max_features = 4500
)

vector.fit(np.concatenate([X_train.ingredients, X_test.ingredients]))

print ("Total No. of features:", len(vector.get_feature_names()))


# In[20]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Train
clf_A = RandomForestClassifier(
    n_estimators = 50,
    oob_score = True,
    verbose = 10,
    n_jobs = 5
)


# Train
clf_B =  DecisionTreeClassifier(random_state = 42)


# In[21]:

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#RandomForest
benchmark_model_A = Pipeline([
    ("vector", vector),
    ("scl", StandardScaler(with_mean=False)),
    ("clf_A", clf_A)
])

benchmark_model_A.fit(X_train.ingredients,y_train.cuisine)

print ("#")
print ("# Best score:", benchmark_model_A.named_steps["clf_A"].oob_score_)
print ("#")

#DecisionTree
benchmark_model_B = Pipeline([
    ("vector", vector),
    ("scl", StandardScaler(with_mean=False)),
    ("clf_B", clf_B)
])

benchmark_model_B.fit(X_train.ingredients,y_train.cuisine)


# In[22]:

#Metrics

from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score

pred_results_A = benchmark_model_A.predict(X_test.ingredients)
print(pred_results_A)

# Train and Test Accuracy for Random Forest
print ("Train Accuracy for Random Forest :: ", accuracy_score(y_train, benchmark_model_A.predict(X_train.ingredients)))
print ("Test Accuracy for Random Forest  :: ", accuracy_score(y_test, pred_results_A))

print ("F-Score on Test for Random Forest :: ",fbeta_score(y_test, pred_results_A,average=None, beta = 0.5))
  
pred_results_B = benchmark_model_B.predict(X_test.ingredients)
print(pred_results_B)

# Train and Test Accuracy for  Decision Tree
print ("Train Accuracy for Decision Tree :: ", accuracy_score(y_train, benchmark_model_B.predict(X_train.ingredients)))
print ("Test Accuracy for Decision Tree  :: ", accuracy_score(y_test, pred_results_B))

print ("F-Score on Test for Decision Tree :: ",fbeta_score(y_test, pred_results_B,average=None, beta = 0.5))


# In[23]:

#Metrics

from sklearn.metrics import log_loss

clf_probs_A =  benchmark_model_A.predict_proba(X_test.ingredients)
clf_probs_B =  benchmark_model_B.predict_proba(X_test.ingredients)

#print(clf_probs_A)

score_A = log_loss(y_test, clf_probs_A)
print("Log Loss for Random Forest :: ",score_A)

score_B = log_loss(y_test, clf_probs_B)
print("Log Loss for Decision Tree :: ",score_B)


# In[24]:

#Kaggle submission files

pred_results_A = benchmark_model_A.predict(test.ingredients)
test['cuisine'] = pred_results_A
print(pred_results_A)

out_file= "../data/bench_mark_random_forest.csv"
op = pd.DataFrame(data={
        "id": test.id,
        "cuisine": test.cuisine
        })
op.sort_values(by="id", inplace=True)
op.to_csv(out_file, columns=["id", "cuisine"], index=False, quoting=3)
print ("Submission for bench_mark Random Forest written to", out_file)


pred_results_B = benchmark_model_B.predict(test.ingredients)
test['cuisine'] = pred_results_B
print(pred_results_B)

out_file= "../data/bench_mark_decision_tree.csv"
op = pd.DataFrame(data={
        "id": test.id,
        "cuisine": test.cuisine
        })
op.sort_values(by="id", inplace=True)
op.to_csv(out_file, columns=["id", "cuisine"], index=False, quoting=3)
print ("Submission for bench_mark Decision Tree written to", out_file)


# In[25]:

# a function that adds new features and the dataframe

def add_features(df): 
    
    # no of ingredients
    df['no_ingredients'] = df.ingredients.apply(len)
    
    # average length of ingredient names
    df['ingredient_len'] = df.ingredients.apply(lambda x: np.mean([len(item) for item in x]))
    
    # make strings of the ingredients list
    df['ingredients_string'] = df.ingredients.astype(str)
    
    return df


# In[26]:

# create the same features in the training data and the new data
train = add_features(pd.read_json('../data/train.json'))
new = add_features(pd.read_json('../data/test.json'))


# In[27]:

train.head()


# In[28]:

train.shape


# In[29]:

new.head()


# In[30]:

new.shape


# ## Optimize Cross Validation using Pipeline

# In[31]:

# assign X and y
X = train.ingredients_string
y = train.cuisine


# In[32]:

# X is just an ingredient series
X.head()


# In[33]:

# define the regex pattern for teh purpose of tokenization
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(token_pattern=r"'([a-z ]+)'")


# In[34]:

# import and declare the Multinomial Naive Bayes along with the default parameters
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()


# In[35]:

# Assign the Naive Bayes with a pipeline of vectorization
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(vector, mnb)


# In[36]:

# explore the pipeline steps
pipeline.steps


# In[37]:

# cross validate the full pipeline
from sklearn.cross_validation import cross_val_score
cross_val_score(pipeline, X, y, cv=7, scoring='accuracy').mean()


# In[38]:

# pipeline steps are automatically assigned names by make_pipeline
pipeline.named_steps.keys()


# In[39]:

# create a grid of parameters to search (and specify the pipeline step along with the parameter)
parameters_grid = {}
parameters_grid['countvectorizer__token_pattern'] = [r"\b\w\w+\b", r"'([a-z ]+)'"]
parameters_grid['multinomialnb__alpha'] = [0.5, 1]
parameters_grid


# In[40]:

# pass the pipeline (instead of the model) to GridSearchCV
from sklearn.grid_search import GridSearchCV
gridCV = GridSearchCV(pipeline, parameters_grid , cv=7, scoring='accuracy')


# In[41]:

# time the grid search
get_ipython().magic(u'time gridCV.fit(X, y)')


# In[42]:

# examine the score for each combination of parameters
gridCV.grid_scores_


# In[43]:

# print the single best score and parameters that produced that score
print(gridCV.best_score_)
print(gridCV.best_params_)


# In[44]:

from sklearn.grid_search import RandomizedSearchCV


# In[45]:

# for the continuous parameters, distribution is always prefeered when compared to a list of options
import scipy as sp
parameters_grid = {}
parameters_grid['countvectorizer__token_pattern'] = [r"\b\w\w+\b", r"'([a-z ]+)'"]
parameters_grid['countvectorizer__min_df'] = [1, 2, 3]
parameters_grid['multinomialnb__alpha'] = sp.stats.uniform(scale=1)
parameters_grid


# In[46]:

# define a random seed
np.random.seed(1)


# In[47]:

# additional parameters are achieved thru number of searches (n_tier) and random_state
rdm = RandomizedSearchCV(pipeline, parameters_grid, cv=5, scoring='accuracy', n_iter=5, random_state=1)


# In[48]:

# time the randomized search
get_ipython().magic(u'time rdm.fit(X, y)')


# In[49]:

rdm.grid_scores_


# In[50]:

print(rdm.best_score_)
print(rdm.best_params_)


# ### Making predictions for test data

# In[62]:

# Assign X_new as the ingredients string
X_new = new.ingredients_string
X_new


# In[52]:

# what is the best model identified by RandomizedSearchCV
rdm.best_estimator_


# In[53]:

# RandomizedSearchCV/GridSearchCV now refits the best model and ready to make predictions for all the dataset
new_pred_class_rdm = rdm.predict(X_new)
new_pred_class_rdm


# In[66]:

# train_test_split

train_features_new= train.drop('cuisine', axis = 1)
train_cuisine_new = pd.DataFrame(train['cuisine'])

# Split the 'features' and 'Yummly' data into training and testing sets
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(train_features_new, 
                                                    train_cuisine_new , 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print ("Training set has {} samples.".format(X_train_new.shape[0]))
print ("Testing set has {} samples.".format(X_test_new.shape[0]))
print (y_train_new.shape[0])
print (y_test_new.shape[0])
X_train_new.shape


# In[68]:

X_test_new.ingredients_string


# In[69]:

#Metrics

#test['cuisine'] = new_pred_class_rdm
pred_results_mnb = rdm.predict(X_test_new.ingredients_string)

# Test Accuracy for Naive Bayes
print ("Test Accuracy for Naive Bayes  :: ", accuracy_score(y_test, pred_results_mnb))

print ("F-Score on Test for Naive Bayes :: ",fbeta_score(y_test, pred_results_mnb,average=None, beta = 0.5))

clf_probs_mnb =   rdm.predict_proba(X_test_new.ingredients_string)
print(clf_probs_mnb)

score_mnb = log_loss(y_test, clf_probs_mnb)
print("Log Loss for Naive Bayes :: ",score_mnb)


# In[70]:

# create a submission file (score: 0.75341)
pd.DataFrame({'id':new.id, 'cuisine':new_pred_class_rdm}).set_index('id').to_csv('../data/actual1_naive_bayes.csv')


# In[71]:

# create a document term matrix using the entire training data
X_dtm = vector.fit_transform(X)
X_dtm.shape


# In[72]:

type(X_dtm)


# In[73]:

# DF of the custom created features
X_custom = train.loc[:, ['no_ingredients', 'ingredient_len']]
X_custom.shape


# In[74]:

# a sparse matrix from the above DF
X_custom_sparse = sp.sparse.csr_matrix(X_custom)
type(X_custom_sparse)


# In[75]:

# combine the two sparse matrices
X_dtm_custom = sp.sparse.hstack([X_dtm, X_custom_sparse])
X_dtm_custom.shape


# ### Converting a function into a transformer

# In[76]:

# Create a function that takes a DataFrame & returns the custom created features
def get_custom(df):
    return df.loc[:, ['no_ingredients', 'ingredient_len']]


# In[77]:

get_custom(train).head()


# In[78]:

from sklearn.preprocessing import FunctionTransformer


# In[79]:

# create a stateless transformer from the get_custom function
get_custom_ft = FunctionTransformer(get_custom, validate=False)
type(get_custom_ft)


# In[80]:

# execute the function using the transform method
get_custom_ft.transform(train).head()


# In[81]:

# create a function that takes DF and returns the ingredients string
def get_txt(df):
    return df.ingredients_string


# In[82]:

# create and test another transformer
get_txt_ft = FunctionTransformer(get_txt, validate=False)
get_txt_ft.transform(train).head()


# In[83]:

from sklearn.pipeline import make_union


# In[84]:

# create a document term matrix using the entire training data
X_dtm = vector.fit_transform(X)
X_dtm.shape


# In[88]:

# Replicate it as a FeatureUnion by  using transformer
f_union = make_union(vector)
X_dtm = f_union.fit_transform(X)
X_dtm.shape


# In[89]:

# properly combine the transformers into a FeatureUnion
f_union = make_union(make_pipeline(get_txt_ft, vector), get_custom_ft)
X_dtm_custom = f_union.fit_transform(train)
X_dtm_custom.shape


# ### Cross Validation

# In[90]:

# is this proper cross validation?
cross_val_score(mnb, X_dtm_custom, y, cv=5, scoring='accuracy').mean()


# In[91]:

# define a pipeline of the FeatureUnion and Naive Bayes
pipeline = make_pipeline(f_union, mnb)


# In[92]:

# do proper cross validate the entire pipeline and pass it the DF
cross_val_score(pipeline, train, y, cv=5, scoring='accuracy').mean()


# ### Alternative way to specify `Pipeline` and `FeatureUnion`

# In[93]:

# quick rewind to the pipeline I did earlier
f_union = make_union(make_pipeline(get_txt_ft, vector), get_custom_ft)
pipeline = make_pipeline(f_union, mnb)


# In[94]:

# repicate the pipeline without using the make_union or make_pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
pipeline = Pipeline([
    ('featureunion', FeatureUnion([
            ('pipeline', Pipeline([
                    ('functiontransformer', get_txt_ft),
                    ('countvectorizer', vector)
                    ])),
            ('functiontransformer', get_custom_ft)
        ])),
    ('multinomialnb', mnb)
])


# ###  The Nested Pipeline's Grid search 

# In[95]:

# explore the pipeline steps
pipeline.steps


# In[96]:

# define a grid of parameters to search & create the pipeline steps along with the parameters

parameters_grid = {}
parameters_grid['featureunion__pipeline__countvectorizer__token_pattern'] = [r"\b\w\w+\b", r"'([a-z ]+)'"]
parameters_grid['multinomialnb__alpha'] = [0.5, 1]
parameters_grid


# In[97]:

gridCV = GridSearchCV(pipeline, parameters_grid, cv=5, scoring='accuracy')


# In[98]:

get_ipython().magic(u'time gridCV.fit(train, y)')


# In[99]:

print(gridCV.best_score_)
print(gridCV.best_params_)


# ### Model 1: KNN model using only custom features

# In[100]:

# define X and y
feature_columns = ['no_ingredients', 'ingredient_len']
X = train[feature_columns]
y = train.cuisine


# In[101]:

# use KNN with K=800
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=800)


# In[102]:

# train KNN on all of the training data
knn.fit(X, y)


# In[103]:

# create X_new as the custom created features
X_new = new[feature_columns]


# In[104]:

# find predicted probabilities for the new data
new_pred_proba_knn = knn.predict_proba(X_new)
new_pred_proba_knn.shape


# In[105]:

# display the sample of predicted probabilities
new_pred_proba_knn[0, :]


# In[106]:

# model classes
zip(knn.classes_, new_pred_proba_knn[0, :])


# ### Model 2: Naive Bayes model using default features

# In[107]:

# the best model earlier found by RandomizedSearchCV
rdm.best_estimator_


# In[108]:

# X_new as the ingredients string
X_new = new.ingredients_string


# In[109]:

# calculate predicted probabilities of class membership for the new data
new_pred_proba_rdm = rdm.predict_proba(X_new)
new_pred_proba_rdm.shape


# In[110]:

# sampel of predicted probabilities
new_pred_proba_rdm[0, :]


# ### Ensembling models 1 and 2

# In[111]:

# calculate the mean of the predicted probabilities for all rows
new_pred_proba = pd.DataFrame((new_pred_proba_knn + new_pred_proba_rdm) / 2, columns=knn.classes_)
new_pred_proba.head()


# In[112]:

# find the field with the highest predicted probability
new_pred_proba_class = new_pred_proba.apply(np.argmax, axis=1)
new_pred_proba_class.head()


# In[113]:

# create a submission file
pd.DataFrame({'id':new.id, 'cuisine':new_pred_proba_class}).set_index('id').to_csv('../data/actual2_ensembled_models.csv')

