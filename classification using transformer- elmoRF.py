# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 13:22:20 2021

@author: aai00920
"""
#%%
######------------------ Sentence extraction using elmo-------------------------------
# remove all warnings
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import os
import numpy as np
import pandas as pd
import tensorflow as tf;
import tensorflow_hub as hub
from sklearn import preprocessing

import spacy
from spacy.lang.en import English
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

import logging
logging.getLogger('tensorflow').disabled = True #OPTIONAL - to disable outputs from Tensorflow

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

#%%

os.chdir(r'C:\Users\AAI00920\OneDrive - ARCADIS\Desktop\NLP\classification')
df = pd.read_csv('Sample Dataset.csv')
df.head()

url = "https://tfhub.dev/google/elmo/3" # you can choose the version from tensorflow hub
embed = hub.Module(url)


#### the paragraph's are converted into sentences and make sure word count < 150, this is a constrain while using elmo model
text = ' '.join(df['Abstract Text']) # the column name is Abstract Text
text = text.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ')
text = ' '.join(text.split())
doc = nlp(text)
sentences = []
for i in doc.sents:
  if len(i)>1:
    sentences.append(i.string.strip())



sentences[0:5]

#### lets embed the data (make sure the sentences are in list)
sentences_1 = sentences[0:20] # to run for smaller data
embeddings = embed(
    sentences_1,
    signature="default",
    as_dict=True)["default"]
        
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  x = sess.run(embeddings)

x.shape

#### lets try to visualize this embedding in a 2D space using PCA
#The embedding dimension is now downgraded from 1024D to 2D
pca = PCA(n_components=20)
y = pca.fit_transform(x)
        
y = TSNE(n_components=2).fit_transform(y)

data = [
    go.Scatter(
        x=[i[0] for i in y],
        y=[i[1] for i in y],
        mode='markers',
        text=[i for i in sentences],
    marker=dict(
        size=16,
        color = [len(i) for i in sentences], #set color equal to a variable
        opacity= 0.8,
        colorscale='viridis',
        showscale=False
    )
    )
]
layout = go.Layout()
layout = dict(
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False)
             )
fig = go.Figure(data=data, layout=layout)
fig.update_layout(width=900,height=600, title_text='Elmo Embeddings represented in 2 dimension)')

## Sentence Extraction
"""
step1: create embedding for the corpus (done above)
step2: create keyword embedding
step3: carry out cosine similarity
step4:
"""

search_word = 'amputation'
embeddings = embed(
    [search_word],
    signature="default",
    as_dict=True)["default"]
        
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  search_vect = sess.run(embeddings)

cosine_similarities = pd.Series(cosine_similarity(search_vect, x).flatten()) 
results_returned = "2"

output =""
for i,j in cosine_similarities.nlargest(int(results_returned)).iteritems():
  for i in sentences_1[i].split():
    if i.lower() in search_word:
      output += " "+str(i)+ ","
    else:
      output += " "+str(i)

# below will work for only one value output if we need more than 1 then use iteritems
max_id = cosine_similarities.nlargest(int(results_returned)).index[0]
sentences_1[max_id]

output_list = list(output.split(".")) 

#%%
#---------------------classification (embeddings + random forest) -------------------------------------
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from sklearn.ensemble import RandomForestClassifier
os.chdir(r'C:\Users\AAI00920\OneDrive - ARCADIS\Desktop\NLP\classification')
df = pd.read_csv('Sample Dataset.csv')
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)


# Step1: preparation of data for classification model
sent = []
noi_list = []
for j in range(len(df)):
    text = df['Abstract Text'][j]
    text_1 = df['Nature of Injury'][j]
    doc = nlp(text)
    
    for i in doc.sents:
        if len(i)>1:
           sent.append(i.string.strip())
           noi_list.append(text_1)
        
training_data = pd.DataFrame(list(zip(sent,noi_list)), columns = ['Description', 'Nature of Injury'])

#Step 2: cleaning th data and EDA
#removing stopwords:

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
   
try:
    training_data['Description'] = training_data['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
except:
    print("stopwords not removed")

training_data = training_data.dropna() # remove na rows
queries = training_data['Nature of Injury'].unique() 

# create an intermediate dataframe and check for labels and subset for each
train = pd.DataFrame()
test =  pd.DataFrame()

for i in queries:
    #print(i)
    #i = 'Amputation, Crushing' 
    #subsetting because we need 75% training data for each label
    intermediate_df = training_data[training_data['Nature of Injury'] == i]
    train_int = intermediate_df.head(int(len(intermediate_df)*(75/100))) # keeping 75% of the data for training
    train = train.append(train_int)
    test_int = intermediate_df.tail(int(len(intermediate_df)*(25/100)))
    test = test.append(test_int)

# dictionary with key value pair for questions of site summary
keys= queries
values = np.arange(0,len(queries))
di = dict(zip(keys, values))
di_swap = dict(zip(values, keys))



# replacing the label values with numbers
train = train.replace({"Nature of Injury": di})
test = test.replace({"Nature of Injury": di})

# remove punctuation marks
punctuation = '!"#$%&()*+-:;<=>?@[\\]^_`{|}~'

train['Description'] = train['Description'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
test['Description'] = test['Description'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

# convert text to lowercase
train['Description'] = train['Description'].str.lower()
test['Description'] = test['Description'].str.lower()

# remove whitespaces
train['Description'] = train['Description'].apply(lambda x:' '.join(x.split()))
test['Description'] = test['Description'].apply(lambda x: ' '.join(x.split()))

# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

train['Description'] = lemmatization(train['Description'])
test['Description'] = lemmatization(test['Description'])

## creating embeddings using elmo
# training data loaded as chunks
list_train = [train[i:i+100] for i in range(0,train.shape[0],100)] 
list_test = [test[i:i+100] for i in range(0,test.shape[0],100)]

def elmo_vectors(x):
  embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))

#taking first 2 batches ((200 values) to spend less time
list_train1 = list_train[0:2]
# Extract ELMo embeddings
elmo_train = [elmo_vectors(x['Description']) for x in list_train1]
#elmo_test = [elmo_vectors(x['Description']) for x in list_test]

# concatenating into 1 array for train and test
elmo_train_new = np.concatenate(elmo_train, axis = 0)

# creating train test data
xtrain = elmo_train_new
ytrain = train["Nature of Injury"][0:200]
xvalid = elmo_train_new

# Random Forest Model
clf=RandomForestClassifier(n_estimators=500)
clf.fit(xtrain,ytrain)
pred = clf.predict(xtrain)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(ytrain, pred))

# prediting on the same data
master_df = train[0:200]
master_df['prediction'] = pred
master_df = master_df.replace({"prediction": di_swap})

#%%
# confusion matrix and classification report(precision, recall, F1-score)
print(classification_report(ytrain, pred))
print(confusion_matrix(ytrain, pred))

# Saving the model as pickle file
with open('ehselmo.pickle', 'wb') as f:
    pickle.dump(clf, f)










