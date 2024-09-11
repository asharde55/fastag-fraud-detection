#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[2]:


df = pd.read_csv("C:/Users/abhis/Downloads/FastagFraudDetection.csv")


# In[3]:


df.tail()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.shape


# In[7]:


df1 = df.copy   ##creating copy of Data
print(df1)


# In[8]:


# Convert 'Timestamp' column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])


# In[9]:


df


# In[10]:


df.isnull().sum()


# In[11]:


print("Missing value in FastagID :", df['FastagID'].isnull().sum())


# In[12]:


df = df.dropna(subset =['FastagID'])


# In[13]:


df.isnull().sum()


# In[14]:


df.info()


# In[15]:


df.hist(figsize=(20, 16),bins= 20)
plt.show()


# In[16]:


sns.countplot(x = 'Fraud_indicator',data = df)
plt.show()


# In[17]:


# Pairwise scatterplot for numerical variables


# In[18]:


sns.pairplot(df ,vars=['Transaction_Amount','Amount_paid','Vehicle_Speed'])
plt.show()


# In[19]:


# Box plot for 'Transaction_Amount' and 'Amount_paid'
sns.boxplot(x = 'Fraud_indicator', y = 'Transaction_Amount', data = df)


# In[20]:


# # Correlation matrix and heatmap for numerical variables


# In[21]:


correlation_matrix = df[['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[22]:


# Histogram of 'Transaction_Amount'
plt.hist(df['Transaction_Amount'], bins=30, edgecolor='black')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.show()


# In[23]:


sns.regplot( x = 'Transaction_Amount', y ='Amount_paid', data = df )


# In[24]:


sns.boxplot(x = 'Transaction_Amount', data = df)


# In[25]:


# #Select features Transaction_Amount, Amount_paid


# In[26]:


selected_feature = ['Transaction_Amount', 'Amount_paid']
X = df[selected_feature]
y = df['Fraud_indicator']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42) 


# In[27]:


# #Scaling and encoding output 
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# label_encoder = LabelEncoder()
# y_train_encoded = label_encoder.fit_transform(y_train)
# y_test_encoded = label_encoder.transform(y_test)


# In[28]:


# #Scaling and encoding output 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# In[34]:


#Neural network model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[35]:


model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[36]:


model.fit(X_train_scaled, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)


# In[39]:


y_pred_prob = model.predict(X_test_scaled)

# Convert probabilities to binary predictions
y_pred = np.round(y_pred_prob)

# Print accuracy metrics
accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred)
recall = recall_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred)

# Print accuracy metrics
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))


# In[40]:


#Accuracy metrics

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.ylabel('Score')
plt.title('Model Metrics')
plt.ylim(0, 1)  
plt.show()


# In[ ]:




