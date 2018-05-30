import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

input_data = np.loadtxt('spambase.data', delimiter = ',', dtype = float)
np.random.shuffle(input_data)
X, target = input_data[:,:-1], input_data[:,-1]

train_spam = test_spam = train_probability = test_probability = 0
train_feature_spam, train_feature_not_spam = [], []
train_mean_spam , train_mean_not_spam = [], []
train_sdv_spam, train_sdv_not_spam = [], []
standard_deviation = 0.0001
 

#Splitting the train and test inputs/ targets, we get 2300 
train_input, test_input, train_target, test_target = train_test_split(X, target, test_size = 0.50, random_state = 40)