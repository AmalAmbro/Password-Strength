# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 09:14:53 2021

@author: Admin
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#load data
#I'm having trouble while handling the dataset provided by you guys,I'm using another datset from kaggle,which will be loaded along with assignmnets
#if there are any bad lines in csv data, drop them.
df=pd.read_csv("I:/Ensemble Technique/data.csv",error_bad_lines=False,engine='python')
#Taking only 1999 rows
df=df.iloc[0:1999,]
df=df.dropna(axis=0)
df.shape
df.head(10)
#The data from kaggle has strength values 0,1,2 encoding it to the 0,1
for i in range(0,1999):
    if df.iloc[i,1]==2:
        df.iloc[i,1]=1
#Shuffle data
from sklearn.utils import shuffle
df1=shuffle(df)
df1.head()
#reset index
df1=df1.reset_index(drop=True)
x=df1['password']
y=df1['strength']
sns.countplot(y,data=df1)
#1 is more times present in the data than 0 and 2
df1.groupby(['strength']).count()/len(df1)
#Let us make a list of characters of password
def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character
#convert password into vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vector=TfidfVectorizer(tokenizer=word)
x_vec=vector.fit_transform(df1.password)
#dictionary
vector.vocabulary_
#getting  tf-idf vector for first password

feature_names=vector.get_feature_names()
first_password=x_vec[0]
vec=pd.DataFrame(first_password.T.todense(),index=feature_names,columns=['tfidf'])
vec.sort_values(by=['tfidf'],ascending=False)
#split the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_vec,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

classifier=[]
classifier.append(LogisticRegression(multi_class='ovr'))
classifier.append(LogisticRegression(multi_class='multinomial',solver='newton-cg'))
classifier.append(xgb.XGBClassifier())
classifier.append(MultinomialNB())
#result
result=[]
for model in classifier:
    a=model.fit(x_train,y_train)
    result.append(a.score(x_test,y_test))

result1=pd.DataFrame({'score':result,
                      'algorithms':['logistic_regr_ovr',
                                    'logistic_regr_mutinomial',
                                    'xgboost','naive bayes']})
a=sns.barplot('score','algorithms',data=result1)
a.set_label('accuracy')
a.set_title('cross-val-score')
#Here we can see xgboost has greater accuracy
#prediction
#Using data provided from aispry
D=pd.read_excel("I:/Ensemble Technique/Assignment/Ensemble_Password_Strength.xlsx")
x_pred=np.array(D.iloc[56:67,0])
x_pred=vector.transform(x_pred)
model=xgb.XGBClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_pred)
from sklearn.metrics import accuracy_score
accuracy_score(D.iloc[56:67,1],y_pred)
#So I get an accuracy of 90.9%%
pd.crosstab(D.iloc[56:67,1],y_pred,rownames=['Actual'],colnames=['Predicted'])
#Only 1 weak is predicted as good
