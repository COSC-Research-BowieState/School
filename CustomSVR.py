#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:29:10 2019

@author: hkyeremateng-boateng
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
import pandas as pd

ops.reset_default_graph()

def load_data(fileName):
    data = pd.read_csv(fileName, header=None,sep='\n')
    data = data[data.columns[0:]].values
    return data.transpose()
    
numberOfSamples = 500
# Create graph
sess = tf.Session()
#data = arp.generateFreqEnergy(arp,0.8,0.8,numberOfSamples)
# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

N = numberOfSamples
dLen = 100 #length of the energy detector
N_train = dLen*N//2-dLen+1; #training data length - 14901

wLen = 5*dLen; 
N_test = dLen*N//2; #test data length - 15000
N = N_train+N_test; #total data length - 29901
sample_len = 5
        
trainData = np.zeros((wLen,N_train-wLen));
testData = np.zeros((wLen,N_test-wLen));
trainLbl = np.zeros((1,N_train-wLen));
formattedData = load_data("svmDataSet.csv")
r = 0
for i in np.arange(wLen,N_train-1):
    trainLbl.itemset(r,formattedData[0,i]);
    r = r +1;
    
for s in np.arange(r):
    for k in  np.arange(wLen-1):
        trainData.itemset((k,s),np.real(formattedData[0,s+k]))
for b in np.arange(N_test-wLen):
    for t in  np.arange(wLen):
        testData.itemset((t,b),np.real(formattedData[0,N_train+t+b]));
# Split data into train/test sets
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare batch size
batch_size = 50

# Initialize placeholders
train_data = tf.placeholder(shape=[500, None], dtype=tf.float32,name="train_data")#shape - [500,24401]
trainLbl_data = tf.placeholder(shape=[1, None], dtype=tf.float32,name="train_label")#shape - [1,24401]
#x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32,name="x_data")
#y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32,name="y_target")
test_data = tf.placeholder(shape=[1, None], dtype=tf.float32,name="train_label")

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
Q = tf.Variable(tf.random_normal(shape=[24401,1]))

# Declare model operations
#model_output = tf.add(tf.matmul(x_data, A), b)

model_outputs = tf.add(tf.matmul(trainLbl_data, Q), b)
print(model_outputs)
print(trainLbl_data)
# Declare loss function
# = max(0, abs(target - predicted) + epsilon)
# 1/2 margin width parameter = epsilon
epsilon = tf.constant([0.5])
# Margin term in loss
#loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_output, y_target)), epsilon)))
loss_1 = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_outputs, train_data)), epsilon)))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.075)
#train_step = my_opt.minimize(loss)
train_step_x = my_opt.minimize(loss_1)
# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
train_loss = []
test_loss = []
for i in range(500):
     train_x = trainData
     trainLbl_x = trainLbl
     sess.run(train_step_x, feed_dict={train_data:train_x,trainLbl_data:trainLbl_x})
     sess_test = sess.run(loss_1, feed_dict={test_data:testData,trainLbl_data:trainLbl_x})
'''
for i in range(200):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    train_x = trainData
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y,train_data:train_x})
    
    temp_train_loss = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
    train_loss.append(temp_train_loss)
    
    temp_test_loss = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
    test_loss.append(temp_test_loss)
    if (i+1)%50==0:
        print('-----------')
        print('Generation: ' + str(i+1))
        print('A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Train Loss = ' + str(temp_train_loss))
        print('Test Loss = ' + str(temp_test_loss))

# Extract Coefficients
[[slope]] = sess.run(A)
[[y_intercept]] = sess.run(b)
width = sess.run(epsilon)

# Get best fit line
best_fit = []
best_fit_upper = []
best_fit_lower = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)
    best_fit_upper.append(slope*i+y_intercept+width)
    best_fit_lower.append(slope*i+y_intercept-width)

# Plot fit with data
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='SVM Regression Line', linewidth=3)
plt.plot(x_vals, best_fit_upper, 'r--', linewidth=2)
plt.plot(x_vals, best_fit_lower, 'r--', linewidth=2)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()

# Plot loss over time
plt.plot(train_loss, 'k-', label='Train Set Loss')
plt.plot(test_loss, 'r--', label='Test Set Loss')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.legend(loc='upper right')
plt.show()
'''
