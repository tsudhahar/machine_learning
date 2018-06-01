#Machine Learning Engineer Nanodegree
##Capstone Proposal

Sudhahar Thiagarajan
February 19, 2018

##Cuisine Categorization by the recipe of ingredients

###Datasets and Inputs

The whole train dataset contains thirty thousand plus recipes along with their
ingredients and cuisine categories. The test data contains close to ten thousand
recipes. An example of a recipe node in train data would be:

{

"id": 24717,

"cuisine": "indian",

"ingredients": [

"turmeric",

"vegetable stock",

"tomatoes",

"garam masala",

"naan",

"red lentils",

"red chili peppers",

"onions",

"spinach",

"sweet potatoes"

]

},

I’m seeing some more challenges on the training dataset such as

1. High Bias
2. Some cuisine categories have more data while some other categories have less data
3. Similar descriptions
4. Multi word strings
5.Plural form
6.etc

The train data, test data and the sample submission files can be downloaded from
<https://www.kaggle.com/c/whats-cooking/data>
