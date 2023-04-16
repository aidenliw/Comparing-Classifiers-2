import numpy
import pandas
import time
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from DataLoader import cross_validation
numpy.set_printoptions(suppress=True)
# Sklearn Link: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

class NaiveBayes:

    # Train the given dataset by using Gaussian Naive Bayes from sklearn library
    # Return the trained classifier
    def train_GNB_dataset_sklearn(self, data_a, data_b):
        # Merge the data set for training
        train_set = numpy.concatenate((data_a, data_b))
        # Generate category label based on the original dataset
        train_labels = numpy.concatenate((numpy.ones(len(data_a)),
                                      numpy.full(len(data_b), fill_value=2, dtype=int)))
        
        # Train the data by using the KNeighborsClassifier from sklearn, with the input number of neighbours
        classifier = GaussianNB()
        classifier.fit(train_set, train_labels)

        return classifier

    # Test the given dataset by using Gaussian Naive Bayes from sklearn library
    def test_GNB_dataset_sklearn(self, data_a, data_b, classifier):
        # Merge the data set for testing
        test_set = numpy.concatenate((data_a, data_b))
        # Generate category label based on the original dataset
        test_labels = numpy.concatenate((numpy.ones(len(data_a)),
                                      numpy.full(len(data_b), fill_value=2, dtype=int)))

        # Get a list of predictions
        prediction = classifier.predict(test_set)

        # Get a list of probability estimation of the positive class
        test_pred_proba = classifier.predict_proba(test_set)[::,1]

        # Get true_positive, false_negative, true_negative, false_positive from comparison
        true_positive = numpy.sum((prediction == test_labels) & (test_labels == 1))
        false_negative = numpy.sum((prediction != test_labels) & (test_labels == 1))
        true_negative = numpy.sum((prediction == test_labels) & (test_labels == 2))
        false_positive = numpy.sum((prediction != test_labels) & (test_labels == 2))

        return true_positive, false_negative, true_negative, false_positive, test_pred_proba, test_labels
    

    # Execute the Gaussian Naive Bayes methods
    # Input: Class A Training Set, Class B Training Set, Class A Testing Set, Class B Testing Set, Neighbours#
    # Output: Print out the values of Error Rate(%), True Positive, True Negative, False Negative, False Positive
    def gaussian_naive_bayes(self, training_set_a, training_set_b, testing_set_a, testing_set_b):
        
        print(" \n > Applying Gaussian Naive Bayes By Using Sklearn.")

        # Train the data
        start = time.time()
        cls = self.train_GNB_dataset_sklearn(training_set_a, training_set_b)
        end = time.time()
        print(" > Computational Times for Training data is %0.2f milliseconds." % ((end - start) * 1000))

        # Test the data
        start = time.time()
        true_positive, false_negative, true_negative, false_positive, test_pred_proba, test_labels = \
            self.test_GNB_dataset_sklearn(testing_set_a, testing_set_b, cls)
        end = time.time()
        print(" > Computational Times for Testing data is %0.2f milliseconds." % ((end - start) * 1000))

        print(" True Positive: %d (%0.2f%%)" % (true_positive, 100*true_positive/(true_positive+false_negative)))
        print(" False Negative: %d (%0.2f%%)" % (false_negative, 100*false_negative/(true_positive+false_negative)))
        print(" True Negative: %d (%0.2f%%)" % (true_negative, 100*true_negative/(true_negative+false_positive)))
        print(" False Positive: %d (%0.2f%%)" % (false_positive, 100*false_positive/(true_negative+false_positive)))

        # Calculate the error rate
        error_rate = 100.0 * (false_negative + false_positive) \
            / (true_positive + false_negative + true_negative + false_positive)
        print(" Error rate: %0.2f%%" % error_rate)

        return error_rate, true_positive, true_negative, false_negative, false_positive, test_pred_proba, test_labels


    # Find the average error rate by using cross validation 
    def GNB_cross_validation(self, training_set_a, training_set_b):
        lowest_error = 100
        best_index = 0
        array_index = []
        array_error = []
        array_TP = []
        array_TN = []
        array_FN = []
        array_FP = []
        for i in range(1,5):
            print("\n Traning Data on Validation Partition %d" % i)
            fold_train_a, fold_train_b, fold_validation_a, fold_validation_b \
                = cross_validation(training_set_a, training_set_b, i)
            error, true_positive, true_negative, false_negative, false_positive, test_pred_proba, test_labels \
                = self.gaussian_naive_bayes(fold_train_a, fold_train_b, fold_validation_a, fold_validation_b)
            array_index.append(i)
            array_error.append(error)
            array_TP.append(true_positive)
            array_TN.append(true_negative)
            array_FN.append(false_negative)
            array_FP.append(false_positive)
            if error < lowest_error:
                lowest_error = error
                best_index = i
        print(" > Best Partition is %d with error rate of %0.2f%%" % (best_index, lowest_error))
        df = pandas.DataFrame({
            "Fold #": array_index, "Error Rare (%)": array_error, 
            "True Positive": array_TP, "True Negative": array_TN,
            "False Negative": array_FN, "False Positive": array_FP,
            "Best Index": best_index, "Lowest Error": lowest_error })
        df.to_excel('GBN Cross Validation Partition Data.xlsx', sheet_name='sheet1', index=False) 
        return lowest_error, best_index
