import nltk,re
import pandas as pd
import numpy as np
from preprocess import *

#reading the data into a pandas datafram
df_raw = pd.read_csv("Datasets/dataset_full_balanced_CMU.csv")

df_raw.columns=['sarcasm','raw_tweet']
df_raw.dropna(inplace=True)
target = (df_raw['sarcasm']=='sarc').astype(int) #create a target list which contains 1 for sarcastic tweet and 0 for non sarcastic

cleanded_tweets = preprocess(df_raw['raw_tweet'])

print(cleanded_tweets[:10])


