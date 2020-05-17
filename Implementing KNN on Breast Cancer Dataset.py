#!/usr/bin/env python
# coding: utf-8

# In[64]:


from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


# In[54]:


cancer_ds=datasets.load_breast_cancer() ## Loading the dataset


# In[55]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(cancer_ds.data,cancer_ds.target) ## Splitting the dataset into
                                                                                                ## training and testing data.


# In[56]:


scaler=preprocessing.StandardScaler() ## Loading it to do feature scaling which is a must before working with KNN algorithm


# In[57]:


## Here we are doing feature scaling .Note:-We are fitting the testing data on the same parameters as we have done for training
## data.

scaler.fit(x_train) 
x_train=scaler.transform(x_train) 
x_test=scaler.transform(x_test)


# In[58]:


## This function is finding the best value of no_of_neighbours with the help of cross_val_score.

def fit(x_train,y_train):
    first_run=True
    no_of_neighbour=0
    max_val=0.0
    
    for i in range(1,50,1):
        k=0.0
        clf=KNeighborsClassifier(n_neighbors=i)
        score=cross_val_score(clf,x_train,y_train, cv=KFold(3,True,0))
        
        for j in range(len(score)):
            k=k+score[j]
            
        k=k/(len(score))
        if(max_val<k):
            max_val=k
            no_of_neighbours=i
        
    return no_of_neighbours


# In[59]:


## Here we are predicting the class of the individual data point of the testing data.

def predict_one(x_train,y_train,x_test,k):
    distances=[]
    for i in range(len(x_train)):
        distance=((x_train[i,:]-x_test)**2).sum()
        distances.append([distance,i])
    distances=sorted(distances)
    
    targets=[]
    
    for i in range(k):
        index_of_training_data=distances[i][1]
        targets.append(y_train[index_of_training_data])
        
        return Counter(targets).most_common(1)[0][0]


# In[60]:


## Here we are finding the predictions of each and every data point and storing it in the list(predictions).

def predict(x_train,y_train,x_test_data,k):
    predictions=[]
    for x_test in x_test_data:
        predictions.append(predict_one(x_train,y_train,x_test,k))
    return predictions


# In[61]:


k=fit(x_train,y_train)## Fit function is used to find the best value of no_of_neighbours that we have to use in KNN for
                      ## predictions.

y_predict=predict(x_train,y_train,x_test,k) ## Here we are getting the predictions .


# In[63]:


accuracy_score(y_test,y_predict) ## We are getting to know about our accuracy of our predictions.

