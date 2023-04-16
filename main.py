import numpy
import pandas
import time
import matplotlib.pyplot as plt

from DataLoader import set_up_dataset, receiver_operating_characteristic
from SupportVectorMachines import SupportVectorMachines
from KNearestNeighbors import KNearestNeighbours
from NaiveBayes import NaiveBayes
from AdaBoost import AdaBoost


# Instantiate the classes
knn = KNearestNeighbours()
gnb = NaiveBayes()
adb = AdaBoost()
svm = SupportVectorMachines()

# Retrieve data
training_set1, training_set2, testing_set1, testing_set2 = set_up_dataset()

# # K-Nearest Neighbours
print("\n #################### K-Nearest Neighbours #################### ")
# # Find the best K value with lowest error rate based on 75% training data and 25% testing data
# # Result: Best K value is 279 with error rate of 37.38% (40 minutues code execution)
# knn.KNN_Find_Best_K(3,800,training_set1, training_set2, testing_set1, testing_set2)

# Get the Computational Times with the optimal K value
error_rate, true_positive, true_negative, false_negative, false_positive, test_pred_proba, test_labels \
 = knn.k_nearest_neighbours(training_set1, training_set2, testing_set1, testing_set2, 279)

# Plot the ROC curve
receiver_operating_characteristic(test_labels, test_pred_proba, 2)

# # Cross Validation: Find the best K with lowest error rate through 800 tests * 4 folds loops 
# # (Warning!!! About 2 hours code execution.)
# knn.KNN_cross_validation(3,800, training_set1, training_set2)


# # Naive Bayes
print("\n #################### Naive Bayes #################### ")
# Get the Computational Times
error_rate, true_positive, true_negative, false_negative, false_positive, test_pred_proba, test_labels \
 = gnb.gaussian_naive_bayes(training_set1, training_set2, testing_set1, testing_set2)

# Plot the ROC curve
receiver_operating_characteristic(test_labels, test_pred_proba, 2)

# # Cross Validation: Find the lowest error rate
# gnb.GNB_cross_validation(training_set1, training_set2)


# # AdaBoost
print("\n #################### Adaptive Boosting #################### ")
# # Find the best Estimator value with lowest error rate based on 75% training data and 25% testing data
# # Result: Best Estimator value is 16 with error rate of 40.71% (About 5 minutes code execution.)
# adb.AdaBoost_Find_Best_Estimator(1,100,training_set1, training_set2, testing_set1, testing_set2)

# Get the Computational Times with the optimal Estimator value
error_rate, true_positive, true_negative, false_negative, false_positive, test_pred_proba, test_labels \
 = adb.adaboost(training_set1, training_set2, testing_set1, testing_set2, 16)

# Plot the ROC curve
receiver_operating_characteristic(test_labels, test_pred_proba, 2)

# # Cross Valication: Find the best Estimator with lowest error rate through 100 tests * 4 folds loops 
# # (About 10 minutes of code execution.)
# AdaBoost_cross_validation(1,100, training_set1, training_set2)


# # Support Vector Machines 
print("\n #################### Support Vector Machines  #################### ")
# Get the Computational Times
error_rate, true_positive, true_negative, false_negative, false_positive, test_pred_proba, test_labels \
 = svm.support_vector_machines(training_set1, training_set2, testing_set1, testing_set2)

# Plot the ROC curve
receiver_operating_characteristic(test_labels, test_pred_proba, 2)

# # Cross Validation: Find the lowest error rate (About 1 hour code execution.)
# svm.SVM_cross_validation(training_set1, training_set2)