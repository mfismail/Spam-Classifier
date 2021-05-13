import pandas as pd
import sklearn
import numpy as np

messages=pd.read_csv('C:/Users/DELL/Desktop/Machine_Learning/NLP/SpamClassifier-master/smsspamcollection/smsspamcollection',
                     sep='\t',names=["label","message"])

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-z]'," ",messages["message"][i])
    review=review.lower()
    review=review.split()
    
    review = [ps.stem(word) for  word in review if not word in stopwords.words('english')]
    review=" ".join(review)
    corpus.append(review)
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
x=cv.fit_transform(corpus).toarray()


y=pd.get_dummies(messages['label'])  
y=y.iloc[:,1].values

np.random.seed(42)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()  
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
