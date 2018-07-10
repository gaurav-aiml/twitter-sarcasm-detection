# This file is used to tokenize the tweet and perform pre-processing on each tweet

import nltk
import numpy as np

def preprocess(tweet_list):

	# Used Tweet Tokenizer of NLTK for its advantages in dealing with tweets
	#strip_handles removes the userid : @username from the tweets.
	tokenizer = nltk.TweetTokenizer(preserve_case=False, strip_handles=True)

	#create a column to store the tokenized version of each tweet: This helps in easier pre-processing
	tokenized =[tokenizer.tokenize(tweet) for tweet in tweet_list]
	
	tweets = []

	for token_list in tokenized:
		clean = []
		for token in token_list:
			if token.startswith("#"):
				continue
			elif "http" in token:
				continue
			else:
				clean.append(token)
		clean.append("\n")
		tweets.append(clean)


	cleaned_tweets =[" ".join(line) for line in tweets] #The resultant tokens join to form the cleaned tweets.

	return cleaned_tweets