#!/usr/bin/env python
# coding: utf-8

# In[27]:


import nltk
nltk.download('averaged_perceptron_tagger')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

#For lemmatizaton
import spacy

# Plotting tools

import matplotlib.pyplot as plt

reviews = pd.read_csv('principal_data.csv')


# In[28]:


# converting all the columns name to lower case for easy access
reviews.columns  = reviews.columns.str.lower()
from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop_words = stopwords.words('english')
    not_stopwords = {'no', 'not', 'but', 'all', 'think', 'of'}
    final_stop_words = set([word for word in stop_words if word not in not_stopwords])
    text = [x for x in text if x not in final_stop_words]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# clean text data
reviews["pros"] = reviews["pros"].apply(lambda x: clean_text(x))
reviews["cons"] = reviews["cons"].apply(lambda x: clean_text(x))

reviews['pros'] = [x.replace("\r\n","") for x in reviews['pros']]

reviews['cons'] = [x.replace("\r\n","") for x in reviews['cons']]
reviews['feedback'] = reviews['pros'] + " " + reviews['cons']
reviews['sentiment']=reviews['overall_rating'].apply(lambda x: 2 if int(x)>3 else (1 if int(x) ==3 else 0))
reviews.drop(['pros','location','review_date', 'cons', 'author_active', 'overall_rating', 'review_title', 'outlook', 'recommends', 'review_text'], axis = 1, inplace = True)


# In[29]:




# add tf-idfs columns
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(reviews["feedback"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = reviews.index
reviews = pd.concat([reviews, tfidf_df], axis=1)


# In[30]:


reviews.shape


# In[31]:


import pickle
pickle.dump(tfidf, open('transform.pkl', 'wb'))


# In[32]:


label = "sentiment"
ignore_cols = [label, "feedback"]
features = [c for c in reviews.columns if c not in ignore_cols]
X_train, X_test, y_train, y_test = train_test_split(reviews[features], reviews[label], test_size = 0.20, random_state = 42)
from sklearn.ensemble import RandomForestClassifier
# train a random forest classifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)

# show feature importance
feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)
feature_importances_df.head(20)

rf.score(X_test, y_test)
filename = 'senti_model.pkl'
pickle.dump(rf,open(filename, 'wb'))


# In[ ]:




