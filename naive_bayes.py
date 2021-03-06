#SUSHMITHA SHRIDHAR
#MACHINE LEARNING PROGRAMMING #2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

input_data = np.loadtxt('spambase.data', delimiter = ',', dtype = float)
np.random.shuffle(input_data)
X, target = input_data[:,:-1], input_data[:,-1]


 

#Splitting the train and test inputs/ targets, we get 2300 
train_input, test_input, train_target, test_target = train_test_split(X, target, test_size = 0.50, random_state = 40)

#CREATE PROBABILISTIC MODEL

train_spam = test_spam = train_probability = test_probability = 0
train_feature_spam, train_feature_not_spam = [], []
train_mean_spam , train_mean_not_spam = [], []
train_sdv_spam, train_sdv_not_spam = [], []
standard_deviation = 0.0001

or i in range(len(train_target)):
	if (train_target[i] == 1):
		train_spam+=1
#Obtaining the probsabilty of the spam in train set.
train_probability_spam = (train_spam*1.0)/len(train_target)
#Obtaining the probsabilty of not-spam in train set.
train_probability_not_spam = 1 - train_probability_spam


for i in range(len(test_target)):
	if (test_target[i] == 1):
		test_spam+=1
#Obtaining the probsabilty of the spam in train set.
test_probability_spam = (test_spam*1.0)/len(test_target)
#Obtaining the probsabilty of not-spam in train set.
test_probability_not_spam = 1 - test_probability_spam



for feature in range(0,total_features):
	spam_data, nspam_data = [], []
	
	for row in range(len(train_input)):
		if (train_target[row] == 1):
#If the value of the train input is 1, it is spam, hence we are seperating out the spam and not-spam
			spam_data.append(train_input[row][feature])
		else:
			nspam_data.append(train_input[row][feature])
#Finding the mean of train spam set
	train_mean_spam.append(np.mean(spam_data))
#Finding the mean of train not-spam set
	train_mean_not_spam.append(np.mean(nspam_data))

#Finding the standard deviation of train spam set
	train_sdv_spam.append(np.std(spam_data))
#Finding the standard deviation of train not-spam set
	train_sdv_not_spam.append(np.std(nspam_data))

#If the value of obtained in standard deviation is zero, change the value to 0.0001
#to avoid a divide-by- zero error in Gaussian Naive Bayes
for i in range(len(train_sdv_spam)):
	if (train_sdv_spam[i] == 0):
		train_sdv_spam[i] = standard_deviation


	if (train_sdv_not_spam[i] == 0):
		train_sdv_not_spam[i] = standard_deviation


########################################################## NAIVE BAYES ON THE TEST DATA ################################################


#Using the Gaussian Naive Bayes algorithm to classify the instances in our test set
def gaussian(x,mean,standard_deviation):
	N = float(1/(np.sqrt(2*np.pi)*standard_deviation)) * float(np.exp(-((x-mean)**2)/(2*float(standard_deviation*standard_deviation))))
	return N


for row in range(len(test_input)):
	prob_spam = np.log(train_probability_spam)
	prob_not_spam = np.log(train_probability_not_spam)

	for feature in range(0,total_features):
		x = test_input[row][feature]
		prob_spam += np.log(gaussian(x, train_mean_spam[feature], train_sdv_spam[feature]))
		prob_not_spam += np.log(gaussian(x, train_mean_not_spam[feature], train_sdv_not_spam[feature]))
	
	class_x = np.argmax([prob_not_spam, prob_spam])
	result.append(class_x)

#Confusion matrix for the test set
cfm = confusion_matrix(test_target, result)

print "\n"
print "Confusion Matrix for the test set:"
print cfm
print("\n")



for row in range(len(result)):
	if (result[row] == 1 and test_target[row] == 1):
		TP += 1
	elif (result[row] == 0 and test_target[row] == 0 ):
		TN += 1
	elif (result[row] == 1 and test_target[row] == 0 ):
		FP += 1
	else:
		FN += 1

#Calculating Accuracy , precision and recall.
accuracy = float(TP + TN)/(TP+TN+FP+FN)
print "Accuracy : "
print accuracy
print "\n"

precision = float(TP)/(TP+FP)
print "Precision: "
print precision
print "\n"

recall = float(TP)/(TP+FN)
print "Recall: "
print recall
print "\n"