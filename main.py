
import numpy as np
from sklearn.pipeline import Pipeline
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from senticnet.senticnet import SenticNet











class ItemSelector(BaseEstimator, TransformerMixin):
    """
    For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, tweets):
        feature_list =[]	#list that contains a dictionary of features for each tweet
        
        sn = SenticNet()	#senticnet object


        for tweet in tweets:
            features={}			#each tweet contains dictionary of features
            features['length'] = len(tweet)		#length of the tweet
            generate_sentiment_features(features, tweet, sn)	#generates sentiment score, contrast score , polarity, subjectivity
            feature_list.append(features)
            
            
        return feature_list


class TweetInitprocessor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, tweets):
        data_dict = np.recarray(shape=(len(tweets),), dtype=[('tweets', object),])
        
        
        tokenizer = nltk.TweetTokenizer(preserve_case=False, strip_handles=True)	#makes every tweet lower case and remove @user handles
        lemmatizer = nltk.stem.WordNetLemmatizer()	#use lemmatizer instead of stemming for better accuracy of the sentiment analysis.
        
        
        """ Tokenizes tweets and removes hashtags and urls and puts the tweets in data_dict['tweets']"""
        tokenized =[tokenizer.tokenize(tweet) for tweet in tweets]
        for i,tweet_tokens in enumerate(tokenized):
            clean = []
            for token in tweet_tokens:
                if token.startswith("#"):
                    continue
                elif "http" in token:
                    continue
                else:
                    clean.append(lemmatizer.lemmatize(token))
            data_dict['tweets'][i] = " ".join(clean)
            
        return data_dict





pipeline = Pipeline([ 	# implements a sklearn pipeline.

    #Step1 : Process tweets to remove hashtags and urls
    ('PreProcess', TweetInitprocessor()),

    #Step 2 : Use feature union to combine tfidf features and the extracted dictionary of features
    ('union', FeatureUnion(
        
            transformer_list =[
                                

            # Pipeline for pulling ad hoc features from post's body
                                ('tweet_features', Pipeline(

                                	[ 
                                	  ('selector', ItemSelector(key='tweets')), #selects the processed tweets
                                	  ('stats', TextStats()),  # returns a list of dicts 
                                	  ('vect', DictVectorizer(sparse = False)),  # list of dicts -> feature matrix
                                    ]
                                    						)
                                 
                                ),
                
                                ('tweet_ngrams', Pipeline(

                                	[
                                    
                                    ('selector', ItemSelector(key='tweets')), 
                                    ('tfidf', TfidfVectorizer(ngram_range =(1, 2), analyzer='word', max_df = 0.4, stop_words = 'english')),
                                    
                                    ]
                                    					)
                                ),

        						],

    					)

    ),

    #Step3: Run a classifier.
    ('svc', SVC(kernel='linear')),
])





