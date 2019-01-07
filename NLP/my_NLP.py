#NLP

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset # basically tsv files
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting =3)#quoting 3 is ignoring double quotes

#cleaning texts
import re 
import nltk
nltk.download('stopwords') #remove unnecessary words like 'this'  
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #removing wildcharac n all n keepng only characters
    review = review.lower()#convert to lower case
    review = review.split()
    #stemming 
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #taking irrelevant words from english, removes 'this'
    # after addind stemming part it converts the relevant words to their root form, eg: loved becomes love
    review = ' '.join(review)# join back the characters to one string
    corpus.append(review)
#now putting it the whole thing in for loop

# you could still see some unnecessary names and all, which will be filtered
#using bag of words model
#Creating bag of models
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #1500 moost frequent words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#using classification now, Naive Bayes

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)