import pandas as pd
from main import pipeline
from sklearn.model_selection import train_test_split









df_raw = pd.read_csv("../Datasets/dataset_full_balanced_CMU.csv")
df_raw.columns=['sarcasm','raw_tweet']
df_raw.dropna(inplace=True)
target = (df_raw['sarcasm']=='sarc').astype(int)

"""Splits the data into training and testing"""
X_train, X_test, y_train, y_test = train_test_split(df_raw['raw_tweet'][:500], target[:500], test_size=0.2)


print("Training")
pipeline.fit(X_train, y_train)


print("Predicting")
y = pipeline.predict(X_test)


print(classification_report(y, y_test))
print(accuracy_score(y, y_test))