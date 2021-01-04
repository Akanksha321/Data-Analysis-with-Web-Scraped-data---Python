#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

import pickle

import joblib


# load the model from disk
filename = 'senti_model.pkl'
rf = pickle.load(open(filename, 'rb'))
tfidf=pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():



	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = tfidf.transform(data).toarray()
        
		my_prediction = rf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(host='0.0.0.0', port = 8080)





# In[2]:


get_ipython().run_line_magic('tb', '')


# In[ ]:




