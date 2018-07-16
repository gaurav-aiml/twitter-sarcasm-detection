import numpy as np
import nltk
import scipy
import sklearn


def generate_sentiment_features(features, tweet, sn):
	"""Extracts features from a tweet and puts it into the features dictionary """


    fixed_tweet = contractions.fix(tweet) #removes contractions eg --> isn't changes to is not

    '''Tokenize tweet and conver to lowercase'''
    tokenizer = nltk.WordPunctTokenizer()
    tokens = tokenizer.tokenize(fixed_tweet)
    tokens = [t.lower() for t in tokens]


    '''extract features'''
    sentiment_score = get_mean_sent_score(tokens,sn)
    tweet_polariy, tweet_subjectivity = get_tweet_sentiment_score(fixed_tweet)
    contrast_scores = get_contrast_score(tokens, sn)
    
    '''add the extracted features to the dictionary'''
    features['sentiment_score'] = sentiment_score
    features['polarity'] = tweet_polariy
    features['subjectivity'] = tweet_subjectivity
    #features['sentiment_lable'] = 
    #features['f_half_contrast'] = contrast_scores[0]
    #features['s_half_contrast'] = contrast_scores[1]
    #features['emoji']
    features['contrast_score'] = contrast_scores[2]
    


    
def get_mean_sent_score(tokens,sn):
    sum_sent = 0
    length = 0
    for token in tokens:
        try:
            score = sn.polarity_intense(token)
            sum_sent+=float(score)
            length+=1
        except:
            continue
    if length != 0:
        return sum_sent/length
    else:
        return 0



def get_tweet_sentiment_score(tweet):
    tweet_blob = TextBlob(tweet)
    return tweet_blob.sentiment.polarity, tweet_blob.sentiment.subjectivity




def get_contrast_score(tokens, sn):
    scores = []
    
    if len(tokens)==1:
        tokens+=['.']

        
    first_half = tokens[0:int(len(tokens)/2)]
    second_half = tokens[int(len(tokens)/2):]
    
    first_half_mean_score = get_mean_sent_score(first_half, sn)
    scores.append(first_half_mean_score)
    
    second_half_mean_score = get_mean_sent_score(second_half, sn)
    scores.append(second_half_mean_score)
    
    tweet_contrast_score = np.abs(first_half_mean_score - second_half_mean_score)
    scores.append(tweet_contrast_score)
   
    
    return scores

