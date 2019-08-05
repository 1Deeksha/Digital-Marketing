#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Digital Marketing.csv")
df.head(5)


# In[ ]:


df = df.drop("Time", axis=1)


# In[4]:


df.head(5)


# In[5]:


df.shape


# In[6]:


# Seaborn visualization library
import seaborn as sns
sns.pairplot(df)


# In[7]:


df['sales'].describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.corr()


# In[ ]:


#split dataset

X = df.drop('sales', axis=1)
Y = df.sales


# In[11]:


X.head(5)


# In[12]:


Y.head(5)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.3)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers


# In[19]:


#adding L2 regularizer in neural network

model = Sequential()

# The Input Layer :
model.add(Dense(20,kernel_regularizer=regularizers.l2(0.0001), input_dim = 12, activation='relu'))

# The Hidden Layers :
model.add(Dense(20,kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
model.add(Dense(10,kernel_regularizer=regularizers.l2(0.0001), activation='relu'))

# The Output Layer :
model.add(Dense(1,activation='linear'))

# Compile the network :
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

#fir the model
history= model.fit(X_train, Y_train, epochs=200, batch_size=5, verbose=1, validation_data=(X_test, Y_test))


# In[25]:


import matplotlib.pyplot as plt

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs

epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();


# In[21]:


predictions = model.predict(X_test)
predictions


# In[ ]:


#Identify Top 3 platforms to investment for Digital Mrketing
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


# In[23]:


model = LinearRegression()
rfe = RFE(model,3)
rfe = rfe.fit(X, Y)
print(rfe.support_)
print(rfe.ranking_)


# In[ ]:


#'radio', 'Magazines', 'Search' are top 3 digital marketing platforms where 
#company would be able to increase sales by investing into them.

