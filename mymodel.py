import pandas as pd
import sklearn
import numpy as np
messages=pd.read_csv('smsspamcollection/smsspamcollection',
                     sep='\t',names=["label","message"])

import gensim
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps=PorterStemmer()
lem=WordNetLemmatizer()

corpus1=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-z]'," ",messages["message"][i])
    review=review.lower()
    review=review.split()
    
    review = [word for  word in review if not word in stopwords.words('english')]
    review=" ".join(review)
    corpus1.append(review)
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000)

pickle.dump(cv,open('bag_of_words_transform.pkl','wb'))
x_bag=cv.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
pickle.dump(tfidf,open('tfidf_transform.pkl','wb'))

x_tf=tfidf.fit_transform(corpus1).toarray()

y_bag=pd.get_dummies(messages['label'])  
y_bag=y_bag.iloc[:,1].values

y_tfidf=pd.get_dummies(messages['label'])
y_tfidf=y_tfidf.iloc[:,1].values


np.random.seed(42)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_tf,y_tfidf,test_size=0.2)


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()  
model.fit(x_train,y_train)

y_pred_tfidf=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion1=confusion_matrix(y_test,y_pred_tfidf)


from sklearn.metrics import accuracy_score
accuracy1=accuracy_score(y_test,y_pred_tfidf)

import pickle
filename='spam-model_tfidf.pkl'
pickle.dump(model,open(filename,'wb'))
