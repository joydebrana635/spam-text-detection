## import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


## reading and making dataset

df = pd.read_csv('spam.tsv',sep = '\t')
df.isnull().sum()

ham = df[df['label']=='ham']
spam = df[df['label']=='spam']
ham = ham.sample(spam.shape[0])
data = ham.append(spam, ignore_index=True)

## data preparation

from sklearn.feature_extraction.text import TfidfVectorizer
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size = 0.3, random_state = 0, shuffle = True, stratify = data['label'])
vectorizer = TfidfVectorizer()
X_train1 = vectorizer.fit_transform(X_train)

## applying svm

clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SVC(C = 1000, gamma = 'auto'))])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

## taking input

string1 = input("Enter text message: ")

print(clf.predict([string1]))
## print(clf.predict(["Hey! Whatsup?"]))
## print(clf.predict(["Congratulations!, you have won free ticketsto the USA this summer. TEXT 'WON' to 445566"]))
## print(clf.predict(["Joydeb Congrats! Your Dhani OneFreedom Credit Limit has been increased to Rs. 15000 @ just Rs. 299 per month."]))
## print(clf.predict(['Dear customer, welcome savings with Freecharge Pay Later. Get Rs.200 assured cashback + save up to Rs.1000 on top brands. T&C apply: g.frcg.info/pjDMDYRohdx']))
print(type(clf.predict([string1])))