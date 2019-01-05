
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


np.random.seed(1)


# In[3]:


vec1 = np.array([0, 0, 0])
mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


# In[4]:


sample_for_class1 = np.random.multivariate_normal(vec1, mat1, 20).T


# In[5]:


sample_for_class1


# In[6]:


sample_for_class1.shape


# In[7]:


assert sample_for_class1.shape == (3,20)


# In[8]:


vec2 = np.array([1, 1, 1])
mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
sample_for_class2 = np.random.multivariate_normal(vec2, mat2, 20).T


# In[9]:


sample_for_class2


# In[10]:


sample_for_class2.shape


# In[11]:


all_data = np.concatenate((sample_for_class1, sample_for_class2), axis=1)


# In[12]:


all_data.shape


# In[13]:


all_data


# In[14]:


mean_dim1 = np.mean(all_data[0, :])
mean_dim2 = np.mean(all_data[1, :])
mean_dim3 = np.mean(all_data[2, :])

mean_vector = np.array([[mean_dim1], [mean_dim2], [mean_dim3]])

print('The Mean Vector:\n', mean_vector)


# In[15]:


(all_data[:,1].reshape(3,1)).shape


# In[16]:


scatter_matrix = np.zeros((3,3))
for i in range(all_data.shape[1]):
    scatter_matrix += np.dot((all_data[:, i].reshape(3, 1) - mean_vector),(all_data[:, i].reshape(3, 1) - mean_vector).T)
print('The Scatter Matrix is :\n', scatter_matrix)


# In[17]:


eig_val, eig_vec = np.linalg.eig(scatter_matrix)


# In[18]:


eig_val


# In[19]:


eig_vec


# In[20]:


for ev in eig_vec:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))


# In[22]:


eig_pairs = []
for i in range(len(eig_val)):
    eig_pairs += [(np.abs(eig_val[i]), eig_vec[:,i])]


# In[23]:


eig_pairs


# In[24]:


eig_pairs.sort(key=lambda x: x[0], reverse=True)


# In[25]:


eig_pairs


# In[26]:


matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
print('Matrix W:\n', matrix_w)


# In[30]:


transformed = np.dot(matrix_w.T,all_data)


# In[31]:


transformed


# In[32]:


transformed.shape


# In[36]:


import matplotlib.pyplot as plt


# In[37]:


plt.scatter(transformed[:,0],transformed[:,1])

