# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.autograd as Variable

#Importing dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#Prepare the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

#Getting the number of users and movies 
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

#Converting the data into an array with users in lines and movies in columns
#Using list of lists becasue we are going to use Torch afterwards
def convert(data):
    new_data = []
    for id_user in range(1, nb_users + 1):
        id_movie = data[:,1][data[:,0] == id_user]
        id_ratings = data[:,2][data[:,0] == id_user]
        ratings = np.zeros(nb_movies)
        ratings[id_movie - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)
        
#Converting the data into Torch tensors
#Tensors are arrays of a single data type, a multi-dimensional matrix(not Numpy array but Pytorch array)  
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#Converting the ratings into binary ratings 1(Liked) or 0(Not liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
  
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1  

#Creating the architecure of the Neural Network

class RBM():
    
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk),0)
        self.a += torch.sum((ph0 - phk),0)
    
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv,nh) 

#Training the RBM
#We need a loss function to measure the error between prediction and real ratings
#we will use the simple difference in absolute value
nb_epoch = 10
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[ id_user: id_user+batch_size]
        v0 = training_set[ id_user: id_user+batch_size]
        ph0, _ = rbm.sample_h(v0)
        #k steps of contrastive divergence
        for k in range(10):
            _, hk = rbm.sample_h(vk) #v0 is the target(no change), so we update vk
            _, vk = rbm.sample_v(hk)
            #we put -1 to the nodes that were not originally rated by the user
            #the training is not done at this ratings that never existent
            vk[v0<0] = v0[v0<0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[vk>=0]))
        s +=1.
    print('Epoch: '+str(epoch)+' Train loss: '+str(train_loss/s))
    
#Testing the RBM


test_loss = 0
s = 0.
for id_user in range(nb_users):
    #we re using the inputs of the training set to activate the neurons of the RBM to get the predicted ratings of the test set
    v = training_set[ id_user: id_user + 1]
    #vt is the target
    vt = test_set[ id_user: id_user + 1]
    #k steps of contrastive divergence
    if len(vt[vt>=0]) > 0:
        _, h = rbm.sample_h(v) #v0 is the target(no change), so we update vk
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
        s +=1.
print('Test loss: '+str(test_loss/s))
        

#Check that 0.25 corresponds to 75% of success
u = np.random.choice([0,1], 100000)
r = np.random.choice([0,1], 100000)
u[:50000] = r[:50000]
print('Accuracy : '+str(sum(u==r)/float(len(u)))) # -> you get 0.75
print('loss '+str(np.mean(np.abs(u-r)))) # -> you get 0.25
    
    
    