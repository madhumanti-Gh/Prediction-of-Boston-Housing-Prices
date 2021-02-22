#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#NAME: MADHUMANTI SAMANTA
#PROJECT: PREDICTION OF BOSTON HOUSING PRICE USING MACHINE LEARNING


# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[12]:


#Load the dataset
boston=load_boston()
#Description of the dataset
print(boston.DESCR)


# In[13]:


#Put the data into pandas DataFrames
features=pd.DataFrame(boston.data,columns=boston.feature_names)
features


# In[14]:


features["AGE"]


# In[15]:


features["LSTAT"]


# In[16]:


target=pd.DataFrame(boston.target,columns=["target"])
target


# In[17]:


max(target["target"])      #max(Dataframe_object[column_name])


# In[18]:


min(target["target"])     #min(["target"])


# In[19]:


#Concatenate features and target into a single datafarme
#axis=1 makes it concatenate column (axis=0, which is default, i.e., concatenate by rows)
DataFrame2=pd.concat([features,target],axis=1)
DataFrame2


# In[20]:


#Pandas attribute 'describe()', i.e., "DataFrame.describe()": Generates descriptive statistics that summerize the central tendency, dispersion and shape of a dataset's distribution
#Use round(decimals=2) to set the precesion to 2 decimal places
DataFrame2.describe().round(decimals=2)


# In[22]:


#Correlation is a statistical technique that shows wheather and how strongly pairs of variables are related.
#function 'corr()' from pandas
#DataFrame.corr(method='pearson')
'''
Compute pairwise correlation
method:{'pearson','kendall','spearman'} or callable
    pearson: standard correlation coefficient
    kendall: Kendall Tau correlation coefficient
    spearman: Spearman rank correlation
    callable: callable with input two 1d ndarrays
'''
corr=DataFrame2.corr('pearson')
#take absolute values of correlations
corrs=[abs(corr[attr]['target']) for attr in list (features)]

#Make a list of pairs[(corr, features)]
l=list(zip(corrs,list(features)))

#sort the list of pairs in reverse/descending order.
#with the correlation value at the key for sorting.
l.sort(key=lambda x:x[0], reverse=True)

#"Unzip" pairs to two lists
#zip(*l) - takes a list that looks like [[a,b,c],[d,e,f],[g,h,i]]
#and returns [[a,d,g],[b,e,h],[c,f,i]]
corrs,labels=list(zip((*l)))

#Plot correlations with resoect to the target variable as a bar graph
index=np.arange(len(labels))
plt.figure(figsize=(15,5))
plt.bar(index,corrs,width=0.6)
plt.xlabel('Attributes')
plt.ylabel('Correlation with the target variation')
plt.xticks(index,labels)
plt.show()


# In[ ]:


#lambda function:
#A lambda function is a 'sort' function with no name, i.e.,Anonymous function.


# In[25]:


X=DataFrame2['LSTAT'].values
Y=DataFrame2['target'].values


# In[26]:


#before normalization
print(Y[:5])


# In[27]:


#Normalisation
x_scaler=MinMaxScaler()
X=x_scaler.fit_transform(X.reshape(-1,1))
X=X[:,-1]
y_scaler=MinMaxScaler()
Y=y_scaler.fit_transform(Y.reshape(-1,1))
Y=Y[:,-1]


# In[28]:


#After noramlization
print(Y[:5])


# In[30]:


#Splitting Data
#0.2 indicates 20% of the data is randomly sampled as testing data
xtrain,xtest, ytrain, ytest=train_test_split(X,Y,test_size=0.2)


# In[29]:


def error(m,x,c,t):
    N=x.size
    e=sum(((m*x+c)-t)**2)
    return e*1/(2*N)


# In[31]:


def update(m,x,c,t,learning_rate):
    grad_m=sum(2*((m*x+c)-t)*x)
    grad_c=sum(2*((m*x+c)-t))
    m=m-grad_m*learning_rate
    c=c-grad_c*learning_rate
    return m,c
# when use this function, remember to unpack the result as it is given of more than one value.


# In[32]:


def gradient_descent(init_m,init_c,x,t,learning_rate, iterations,error_threshold):
    m=init_m
    c=init_c
    error_values=list()
    mc_values=list()
    for i in range(iterations):
        e=error(m,x,c,t)
        if e<error_threshold:
            print("Error less than the thresold. Stopping gradient descent")
            break
        error_values.append(e)
        m,c=update(m,x,c,t,learning_rate)
        mc_values.append((m,c))
    return m,c,error_values, mc_values


# In[33]:


get_ipython().run_cell_magic('time', '', 'init_m=0.9\ninit_c=0\nlearning_rate=0.001\niterations=250\nerror_threshold=0.001\n\nm,c,error_values,mc_values=gradient_descent(init_m,init_c,xtrain,ytrain,learning_rate,iterations,error_threshold)')


# In[87]:


#%%time


# In[34]:


#As the number of iterations increases, change in the line are less noticable.
#In order to reduce the processing time for the annimation it is advised to choose smaller values.
mc_values_anim=mc_values[0:250:5]


# In[35]:


fig,ax=plt.subplots()
ln,=plt.plot([],[],'-ro',animated=True)

def init():
    plt.scatter(xtest,ytest,color='b')
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    return ln,

def update_frame(frame):
    m,c=mc_values_anim[frame]
    x1,y1=-0.5,m*-.5+c
    x2,y2=1.5,m*1.5+c
    ln.set_data([x1,x2],[y1,y2])
    return ln,

anim=FuncAnimation(fig,update_frame, frames =range(len(mc_values_anim)),init_func=init, blit=True)

HTML(anim.to_html5_video())


# In[90]:


#Visualization of the learning process


# In[36]:


plt.scatter(xtrain,ytrain,color='g')
plt.plot(xtrain,(m*xtrain+c), color='r')
plt.show()


# In[45]:


#Plotting error values
plt.plot(np.arange(len(error_values)),error_values)
plt.xlabel('Iterations')
plt.ylabel('Error')
#plt.show()


# In[38]:


#Prediction
#Calculate the predictions on the test set as a vectorised operation
predicted=(m*xtest)+c


# In[39]:


#Compute MSE for the predicted values on the testing set
mean_squared_error(ytest,predicted)


# In[40]:


#Put xtest,ytest and predicted values into a single DataFrame so that we can see the predicted values alongside the testing set
p=pd.DataFrame(list(zip(xtest,ytest,predicted)), columns=['x','target_t','predicted_y'])
p.head()


# In[41]:


plt.scatter(xtest,ytest,color='b')
plt.plot(xtest,predicted,color='r')
plt.show()


# In[43]:


#Reshape to change the shape to the shape that is required by the scaler
predicted=np.array(predicted).reshape(-1,1)


# In[44]:


#Reshape to change the shape to the shape that is required by the scaler
predicted=predicted.reshape(-1,1)
xtest=xtest.reshape(-1,1)
ytest=ytest.reshape(-1,1)

xtest_scaled=x_scaler.inverse_transform(xtest)
ytest_scaled=y_scaler.inverse_transform(ytest)
predicted_scaled=y_scaler.inverse_transform(predicted)

#This is to remove the extra dimension
xtest_scaled=xtest_scaled[:,-1]
ytest_scaled=ytest_scaled[:,-1]
predicted_scaled=predicted_scaled[:,-1]

p=pd.DataFrame(list(zip(xtest_scaled,ytest_scaled,predicted_scaled)),columns=['x','target_y','predicted_y'])
p=p.round(decimals=2)
p.head()


# In[ ]:





# In[ ]:





# In[ ]:











