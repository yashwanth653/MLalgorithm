#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


x,y=make_blobs(n_samples=100,
               n_features=2,
               centers=3,
               cluster_std=1,
               random_state=0)


# In[17]:


x


# In[55]:


y


# In[18]:


plot=plt.scatter(x[:,0],x[:,1],color="white",edgecolor="black")


# In[19]:


kmeans=KMeans(n_clusters=3,init='random',n_init=1,max_iter=4,tol=1e-04,random_state=2)


# In[20]:


y_km=kmeans.fit_predict(x)


# In[53]:


plt.scatter(x[y_km==0,0],x[y_km==0,1],s=40,c='lightgreen',marker='o',edgecolor="black",label='cluster1')
plt.scatter(x[y_km==1,0],x[y_km==1,1],s=40,c='orange',marker='o',edgecolor="black",label='cluster2')
plt.scatter(x[y_km==2,0],x[y_km==2,1],s=40,c='blue',marker='o',edgecolor="black",label='cluster3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='red',marker='o',edgecolor="black",label='centroids')
plt.legend(scatterpoints=1)


# In[ ]:




