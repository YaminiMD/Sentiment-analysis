import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import time
from textblob import TextBlob
import warnings
import chardet
#import geopandas

#import necessary files
warnings.filterwarnings("ignore",category=DeprecationWarning)


data_df = pd.read_csv("ready_corona_tweets1_30")
#error_bad_lines=False,encoding = result['encoding'])

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
from nltk.stem import PorterStemmer
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
stemming = PorterStemmer()
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

(data_df['label_stemmed']==1).value_counts()

# saving the dataframe 
data_df.to_csv('ready_corona_tweets1_30"') 


#Training and Prediction Phase
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

#defining X
X= data_df['tweet_stemmed']

#Passing through TF - IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90,min_df=2,max_features=2000,stop_words='english')
tfidf_stem = tfidf_vectorizer.fit_transform(X)

#defining Y
y= data_df['label_stemmed']
print("data vectorized")

#Converting into sparse Matrices

train_tfidf = tfidf_stem[:319685, :]
test_tfidf  = tfidf_stem[319685:,:]
train_tfidf = preprocessing.scale(train_tfidf,with_mean=False)
testscale_tfidf  = preprocessing.scale(test_tfidf,with_mean=False)
#print("data scaled")
y= data_df['label_stemmed']
X= data_df.drop(columns=['label_stemmed'])

# Splitting into training and testing data
x_train, x_test , y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

x_train = train_tfidf[y_train.index]

x_test= train_tfidf[y_test.index]

print("data split properly")
Sliced_array = x_test[0:5]
#print(Sliced_array)

# Data Training
lreg = LogisticRegression(solver= 'lbfgs',multi_class='multinomial',max_iter=1000)




##lreg.fit(x_train, y_train)
print("data trained")

#Prediction based on the data
prediction = lreg.predict(x_test)

# Selecting probablities > 0.3
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction.astype(np.int)
print("data is predicted")
#Sliced_array = y_test(t)

# Metrics
print(confusion_matrix(y_test , prediction , labels = [0,1,2]))
print(classification_report(y_test, prediction , labels=[0,1,2],target_names=['negative' ,'neutral','positive']))
# calculating f1 score
print(f1_score(y_test, prediction, average ='macro')) 




end_time = time.time()
run_time = end_time - start_time
print("run_time:", run_time)
print("classification completed")
