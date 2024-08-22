#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import nltk
import re
import pickle

# Download necessary NLTK data
nltk.download('punkt')       # Download the 'punkt' tokenizer data
nltk.download('wordnet')     # Download the 'wordnet' corpus for lemmatization
nltk.download('stopwords')   # Download stopwords if you plan to use them


# In[2]:


# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table('InsertTableName', engine)

# Define feature and target variables X and Y
X = df['message']
Y = df.iloc[:, 4:]  # Assuming the first 4 columns are not target categories


# ### 2. Write a tokenization function to process your text data

# In[3]:


def tokenize(text):
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    
    # Lemmatize and remove stop words
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token).strip() for token in tokens]
    
    return tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[4]:


# This pipeline uses TF-IDF and MultiOutputClassifier with RandomForest
pipeline = Pipeline([
    ('vect', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[5]:


# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[6]:


# Train pipeline
pipeline.fit(X_train, Y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[7]:


# Predict on test data
Y_pred = pipeline.predict(X_test)


# In[8]:


# Report the f1 score, precision, and recall for each output category
for i, column in enumerate(Y.columns):
    print(f'Category: {column}\n', classification_report(Y_test[column], Y_pred[:, i]))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[9]:


# Use grid search to find better parameters
parameters = {
    'clf__estimator__n_estimators': [10, 20],
    'clf__estimator__min_samples_split': [2, 4]
}

cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=3, n_jobs=-1)

# Train the grid search model
cv.fit(X_train, Y_train)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[10]:


# Predict on test data using the best model
Y_pred_cv = cv.predict(X_test)

# Report the accuracy, precision, and recall of the tuned model
for i, column in enumerate(Y.columns):
    print(f'Category: {column}\n', classification_report(Y_test[column], Y_pred_cv[:, i]))

## 8. Try improving your model further

# Ideas for improvement:
# - Try different machine learning algorithms like AdaBoost or Gradient Boosting
# - Add other features such as POS tagging, named entity recognition, or using pre-trained word embeddings

## 9. Export your model as a pickle file

# Export the model to a pickle file
with open('classifier.pkl', 'wb') as file:
    pickle.dump(cv.best_estimator_, file)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# ### 9. Export your model as a pickle file

# In[11]:


with open('classifier.pkl', 'wb') as file:
    pickle.dump(cv.best_estimator_, file)


# ### 10. Use this notebook to complete `train_classifier.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




