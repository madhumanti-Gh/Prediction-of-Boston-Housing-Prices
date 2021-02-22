#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error


# In[2]:


n=200
x=np.linspace(0,2*np.pi,n)
sine_values=np.sin(x)
plt.plot(x,sine_values,)
#plt.show()


# In[3]:


noise=0.5
noisy_sine_values=sine_values+np.random.uniform(-noise,noise,n)
plt.plot(x,noisy_sine_values,'r')
plt.plot(x,sine_values,linewidth=3)
plt.show()


# In[4]:


#Calculate MSE using general formula
error_value=(1/n)*sum(np.power(sine_values- noisy_sine_values,2))
error_value


# In[5]:


#Calculate MSE using the library function 'sklearn'
mean_squared_error(sine_values,noisy_sine_values)
#mean_squared_error


# In[6]:


#from Ipython import HTML
#HTML(anim.to_html5_video())


# In[ ]:




