# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 01:58:32 2020

@author: SIDSAG"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from textblob import TextBlob
import warnings
import time


warnings.filterwarnings("ignore",category=DeprecationWarning)
data_df = pd.read_csv("ready_corona_tweets1_30")
start_time = time.time()

from html.parser import HTMLParser
html_parser = HTMLParser()
data_df['clean_text'] = data_df['text'].apply(lambda x: html_parser.unescape(x))

def remove_pattern(input_text,pattern):
  r= re.findall(pattern,input_text)
  for i in r:
    input_text = re.sub(i,'',input_text)
  return input_text

data_df['clean_text'] = np.vectorize(remove_pattern)(data_df['clean_text'],"@[\w]*")  
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: x.lower())


def lookup_dict(text,dictionary):
  for word in text.split():
    if word.lower() in dictionary:
      if word.lower() in text.split():
        text = text.replace(word,dictionary[word.lower()])
  return text


# converting apostrophe 
import pickle
pickle_in = open("apos_dict.pickle","rb")
apostrophe_dict = pickle.load(pickle_in)
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: lookup_dict(x,apostrophe_dict))

# converting abbreivations 
pickle_in = open("short_dict.pickle","rb")
short_word_dict = pickle.load(pickle_in)

data_df['clean_text'] = data_df['clean_text'].apply(lambda x: lookup_dict(x,short_word_dict))

# converting emoticons
pickle_in = open("emot_dict.pickle","rb")
emoticon_dict = pickle.load(pickle_in)
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: lookup_dict(x,emoticon_dict))

# converting emoji's
import emoji
def rep_emoji(tweet):
  tweet = emoji.demojize(tweet)
  tweet = tweet.replace(":" , " ")
  tweet=' '.join(tweet.split())
  return tweet
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: rep_emoji(x))

data_df['clean_text'] = data_df['clean_text'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: re.sub(r'[^a-zA-Z]',' ',x))
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from textblob import TextBlob

data_df['tokenized_tweet'] = data_df['clean_text'].apply(lambda x: word_tokenize(x))
stop_words = set(stopwords.words('english'))
data_df['tweet_token_filter'] = data_df['tokenized_tweet'].apply(lambda x: [word for word in x if not word in stop_words])
lemmatizing = WordNetLemmatizer()
data_df['tweet_lemmatized'] = data_df['tweet_token_filter'].apply(lambda x: ' '.join([lemmatizing.lemmatize(i)for i in x]))
data_df['sentiment_lemmatized'] = data_df['tweet_lemmatized'].apply(lambda x: TextBlob(x).sentiment)
data_df[['sentiment_lemmatized','tweet_lemmatized']].head(10)

all_words = ' '.join([text for text in data_df['tweet_lemmatized']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Most Common words in column Tweet Lemmatized")
plt.show()

sid = SentimentIntensityAnalyzer()
data_df['sentiment_lemmatized'] = data_df['tweet_lemmatized'].apply(lambda x: sid.polarity_scores(x))
data_df['sentiment_lemmatized'].head(10)
def convert(x):
    if x < -0.05:
        return 0
    elif -0.05< x < 0.05 :
        return 1
    else:
        return 2
data_df['label_lemmatized'] = data_df['sentiment_lemmatized'].apply(lambda x: convert(x['compound']))
#data_df['label_stemmed'].head(5)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix


X= data_df['tweet_lemmatized']
tfidf_vectorizer = TfidfVectorizer(max_df=0.90,min_df=2,max_features=1000,stop_words='english')
tfidf_lemm = tfidf_vectorizer.fit_transform(X)
y= data_df['label_lemmatized']
print("data vectorized")

train_tfidf = tfidf_lemm[:319685, :]
test_tfidf  = tfidf_lemm[319685:,:]
x_train, x_test , y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

x_train = train_tfidf[y_train.index]

x_test= train_tfidf[y_test.index]

print("data split properly")
lreg = LogisticRegression(solver='saga',max_iter=10000)

lreg.fit(x_train, y_train)
print("data trained")
pkl_filename = "LR-lemm.pkl"
#with open(pkl_filename,'wb') as file:
 #pickle.dump( model ,file )
prediction = lreg.predict(x_test)
#prediction_int = prediction[:,1] >= 0.3
#prediction_int = prediction_int.astype(np.int)
print("data is predicted")
#Sliced_array = y_test(t)
print(confusion_matrix(y_test , prediction , labels = [0,1,2]))
print(classification_report(y_test, prediction))
print(f1_score(y_test, prediction, average = 'macro')) # calculating f1 score
#computing run time
end_time = time.time()
run_time = end_time - start_time
print("Run time:", run_time)
print("classification completed")

#saving and loading pickle file
start_time = time.time()
import pickle
pkl_filename = "LR-tfidf.pkl"
with open(pkl_filename,'wb') as file:
 pickle.dump(lreg ,file )
with open(pkl_filename, 'rb') as file:
 LR_Model = pickle.load(file)
#Prediction based on the data
prediction = LR_Model.predict(x_test)
print("data is predicted")
print(confusion_matrix(y_test , prediction , labels =[0,1,2]))
print(classification_report(y_test, prediction ,labels=[0,1,2],target_names=['negative','neutral','positive']))
# calculating f1 score
print(f1_score(y_test, prediction, average ='macro'))
end_time = time.time()
run_time = end_time - start_time
print("run_time:", run_time)
