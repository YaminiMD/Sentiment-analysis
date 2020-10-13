# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 20:08:18 2020

@author: SIDSAG
"""
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import time
from textblob import TextBlob
import warnings

#import necessary files
warnings.filterwarnings("ignore",category=DeprecationWarning)

#read csv files
data_df = pd.read_csv("ready_corona_tweets1_30")


start_time = time.time()

#remove html tags
from html.parser import HTMLParser
html_parser = HTMLParser()
data_df['clean_text'] = data_df['text'].apply(lambda x: html_parser.unescape(x))

def remove_pattern(input_text,pattern):
  r= re.findall(pattern,input_text)
  for i in r:
    input_text = re.sub(i,'',input_text)
  return input_text

# remove @ user and convert all to lower case
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
#def extract_emojis(s):
 # return ''.join(c for c in s if c in emoji.UNICODE_EMOJI)
def rep_emoji(tweet):
  tweet = emoji.demojize(tweet)
  tweet = tweet.replace(":" , " ")
  tweet=' '.join(tweet.split())
  return tweet
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: rep_emoji(x))

# converting special characters to spaces and removing words lesser than 2
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: re.sub(r'[^a-zA-Z]',' ',x))
data_df['clean_text'] = data_df['clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

#For tokenizing and stemming
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from textblob import TextBlob

#word tokenizing
data_df['tokenized_tweet'] = data_df['clean_text'].apply(lambda x: word_tokenize(x))
stop_words = set(stopwords.words('english'))

#remove stopwords
data_df['tweet_token_filter'] = data_df['tokenized_tweet'].apply(lambda x: [word for word in x if not word in stop_words])

#stem words
#stemming = PorterStemmer()
stemming = SnowballStemmer("english")
data_df['tweet_stemmed'] = data_df['tweet_token_filter'].apply(lambda x: ' '.join([stemming.stem(i) for i in x]))
#data_df['tweet_stemmed'].head(5)


# Sentiment analysis using VADER
sid = SentimentIntensityAnalyzer()
data_df['sentiment_stemmed'] = data_df['tweet_stemmed'].apply(lambda x: sid.polarity_scores(x))

def convert(x):
    if x < -0.05:
        return 0
    elif -0.05 < x < 0.05:
        return 1
    else :
        return 2 
# Labeling based on returned values
data_df['label_stemmed'] = data_df['sentiment_stemmed'].apply(lambda x: convert(x['compound']))
#importing HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#hashing vectorization
X= data_df['tweet_stemmed']
tfidf_vectorizer = TfidfVectorizer(max_df=0.90,min_df=2,max_features=2000, stop_words = 'english')
tfidf_stem = tfidf_vectorizer.fit_transform(X)
y= data_df['label_stemmed']
#print("Data vectorized")

#vectorization time
Vectorizing_time = time.time()
print("Vectorizing_time :",Vectorizing_time - start_time)

#train and test set formed
tfidf_trainset = tfidf_stem[:319685, :]
tfidf_testset  = tfidf_stem[319685:,:]

x_train, x_test , y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
x_train = tfidf_trainset[y_train.index]
x_test= tfidf_trainset[y_test.index]

print("Data split into train and test set")

#multinomial naive bayes model training
from sklearn.naive_bayes import MultinomialNB



#MultinomialNB(*, alpha=1.0, fit_prior=True, class_prior=None)
model =  MultinomialNB()
model.fit(x_train, y_train)

#model training
print("Model training time: ", time.time()-start_time)
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
end_time = time.time()
run_time = end_time - start_time
print("Run time:", run_time)

