# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 01:47:32 2020

@author: SIDSAG
"""
Sample code//////////////////////////

#importing dataset
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import time 
#from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
data_df = pd.read_csv("ready_corona_tweets1_30")


#data Preprocessing

#removal of Html tags
from html.parser import HTMLParser
html_parser = HTMLParser()
data_df['clean_text'] = data_df['text'].apply(lambda x:html_parser.unescape(x))
#removal of @user and converting to small letter
def remove_pattern(input_text,pattern):
 r= re.findall(pattern,input_text)
 for i in r:
  input_text = re.sub(i,'',input_text)
 return input_text

data_df['clean_text'] =np.vectorize(remove_pattern)(data_df['clean_text'],"@[\w]*")
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: x.lower())
#print(data_df)

#REPLACEMENT
#Funtion for replacement
def lookup_dict(text,dictionary):
  for word in text.split():
   if word.lower() in dictionary:
    if word.lower() in text.split():
     text = text.replace(word,dictionary[word.lower()])
  return text

#replacement of apostrophe
# Replacing Apostrophe
import pickle
pickle_in = open("apos_dict.pickle","rb")
apostrophe_dict = pickle.load(pickle_in)
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: lookup_dict(x,apostrophe_dict))


#replacing the abbrevation
# converting abbreivations
# converting abbreivations 
pickle_in = open("short_dict.pickle","rb")
short_word_dict = pickle.load(pickle_in)
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: lookup_dict(x,short_word_dict))

#replacement of emotions
pickle_in = open("emot_dict.pickle","rb")
emoticon_dict = pickle.load(pickle_in)
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: lookup_dict(x,emoticon_dict))

#funtion for replacement of emogi
#import emoji 
#def extract_emojis(s):
# return ''.join(c for c in s if c in emoji.UNICODE_EMOJI)
def rep_emoji(tweet):
 tweet = emoji.demojize(tweet)
 tweet = tweet.replace(":" , " ")
 tweet=' '.join(tweet.split())
 return tweet

# Replacement of emoji's 
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: lookup_dict(x,emoticon_dict))
#print(data_df)

# Replacing  punctuation with spaces :
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
#print(data_df)

# Replacing Special Characters with spaces :
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: re.sub(r'[^a-zA-Z]',' ',x))
#print(data_df)

# Removing words lesser than length 2 
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
#print(data_df)

#Tokenizing the words 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from textblob import TextBlob

#word tokenizing
data_df['tokenized_tweet'] = data_df['clean_text'].apply(lambda x: word_tokenize(x))
#print(data_df)

#Removing Stop Words
stop_words = set(stopwords.words('english'))
#remove stopwords
data_df['tweet_token_filter'] =data_df['tokenized_tweet'].apply(lambda x: [word for word in x if not word in stop_words])
#print(data_df)

#Stemming the words
stemming = PorterStemmer()
data_df['tweet_stemmed'] = data_df['tweet_token_filter'].apply(lambda x: ''.join([stemming.stem(i) for i in x]))
#print(data_df)

#Labeling the data and training
# Sentiment analysis using VADER:
sid = SentimentIntensityAnalyzer()
data_df['sentiment_stemmed'] = data_df['tweet_stemmed'].apply(lambda x:
sid.polarity_scores(x))
def convert(x):
 if x < -0.05:
  return 0
 elif -0.05 < x < 0.05:
  return 1
 else :
  return 2

#Labeling based on returned values:
data_df['label_stemmed'] = data_df['sentiment_stemmed'].apply(lambda x: convert(x['compound']))
#importing HashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
#hashing vectorization
X= data_df['tweet_stemmed']
hashing_vectorizer = HashingVectorizer(stop_words = 'english',alternate_sign= False)
hash_stem = hashing_vectorizer.fit_transform(X)
y= data_df['label_stemmed']
#print("Data vectorized")

#vectorization time
Vectorizing_time = time.time()
#print("Vectorizing_time :",Vectorizing_time - start_time)

#train and test set formed
hashing_trainset = hash_stem[:319685, :]
hashing_testset  = hash_stem[319685:,:]
x_train, x_test , y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
x_train = hashing_trainset[y_train.index]
x_test= hashing_trainset[y_test.index]
print("Data split into train and test set")

#multinomial naive bayes model training
from sklearn.naive_bayes import MultinomialNB



#MultinomialNB(*, alpha=1.0, fit_prior=True, class_prior=None)
model =  MultinomialNB()
model.fit(x_train, y_train)

#model training
#print("Model training time: ", time.time()-start_time)
#pkl_filename = "MNB-hash.pkl"
#with open(pkl_filename,'wb') as file:
 # pickle.dump( model ,file )
prediction = model.predict(x_test)

#multinomial naive bayes model testing
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test , prediction, labels = [0,1,2]))
print(classification_report(y_test, prediction , labels=[0,1,2],target_names=['negative' ,'neutral','positive']))
print(f1_score(y_test, prediction , average ='macro')) 

#computing run time
#end_time = time.time()
#run_time = end_time - start_time
#print("Run time:", run_time)