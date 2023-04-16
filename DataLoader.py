import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Load data from the csv file by using pandas and convert it to numpy array
def import_data():
    # Import data from the Excel file by using pandas
    data = pandas.read_csv('accelerometer.csv', header=0)

    # Convert the dataset to a numpy array
    data_npy = numpy.asarray(data)
    return data_npy


# Separate the dataset by the class indicator value
def categorize_data(dataset):
    class_a = []
    class_b = []
    class_c = []

    # Get the index of the indicator, the indicator should be at the start of each line of the data
    indicator_index = 0
    for item in dataset:
        indicator = int(item[indicator_index])
        if indicator == 1:
            class_a.append(item[indicator_index + 1:])
        elif indicator == 2:
            class_b.append(item[indicator_index + 1:])
        elif indicator == 3:
            class_c.append(item[indicator_index + 1:])
    return numpy.asarray(class_a), numpy.asarray(class_b), numpy.asarray(class_c)


# Separate the data by first 75% for training and validation, last 25% for testing
def separate_data(data):
    data_length = len(data)
    pivot = int(data_length * 0.75)
    training_set = data[:pivot]
    testing_set = data[pivot:]
    return training_set, testing_set


# Load data, Categorize data, then Separate data
# Because the each class has different numbers of data set, we need separate them,
# and get some part of each class for traning, and rest for testing
def set_up_dataset():
    # Get data from imported csv file
    data = import_data()
    # categorize data by classes
    class_a, class_b, class_c = categorize_data(data)
    # separate data for training and testing sets
    training_set_a, testing_set_a = separate_data(class_a)
    training_set_b, testing_set_b = separate_data(class_b)
    # Get data with attribute x and y from the dataset for testing
    training_set_a = training_set_a[:, 0:4]
    training_set_b = training_set_b[:, 0:4]
    testing_set_a = testing_set_a[:, 0:4]
    testing_set_b = testing_set_b[:, 0:4]
    print("\n Loaded training set for class A with instances: " + str(len(training_set_a)))
    print(" Loaded training set for class B with instances: " + str(len(training_set_b)))
    print(" Loaded testing set for class A with instances: " + str(len(testing_set_a)))
    print(" Loaded testing set for class B with instances: " + str(len(testing_set_b)))
    return training_set_a, training_set_b, testing_set_a, testing_set_b


# Use 4-fold cross validation method to seperate the data set
# Input: Class A training data, Class A training data, partition number(1, 2, 3, or 4)
# Return: Folded training data for class A and B, Folded validation data for class A and B 
def cross_validation(training_set_a, training_set_b, partition):
    # Use the partition number to separate the data
    partition_start = (partition - 1) * 0.25
    partition_end = partition * 0.25

    # Get the separation points of each data set by the partition for class A data
    pivot_a1 = int(len(training_set_a) * partition_start)
    pivot_a2 = int(len(training_set_b) * partition_end)
    fold_validation_a = training_set_a[pivot_a1:pivot_a2]
    # Because the numpy cannot concatenate empty array, we have to set up conditions to
    # only merge the array that has values in it
    if (training_set_a[:pivot_a1].size and training_set_a[pivot_a2:].size):
        fold_train_a = numpy.concatenate((training_set_a[:pivot_a1], training_set_a[pivot_a2:]), axis=0)
    elif training_set_a[:pivot_a1].size == 0:
        fold_train_a = training_set_a[pivot_a2:]
    elif training_set_a[pivot_a2:].size == 0:
        fold_train_a = training_set_a[:pivot_a1]

    # Get the separation points of each data set by the partition class B data
    pivot_b1 = int(len(training_set_b) * partition_start)
    pivot_b2 = int(len(training_set_b) * partition_end)
    fold_validation_b = training_set_b[pivot_b1:pivot_b2]
    # Because the numpy cannot concatenate empty array, we have to set up conditions to
    # only merge the array that has values in it
    if (training_set_b[:pivot_b1].size and training_set_b[pivot_b2:].size):
        fold_train_b = numpy.concatenate((training_set_b[:pivot_b1], training_set_b[pivot_b2:]), axis=0)
    elif training_set_b[:pivot_b1].size == 0:
        fold_train_b = training_set_b[pivot_b2:]
    elif training_set_b[pivot_b2:].size == 0:
        fold_train_b = training_set_b[:pivot_b1]
   
    return fold_train_a, fold_train_b, fold_validation_a, fold_validation_b


# Ploy the ROC curve by using Sklearn metrics
def receiver_operating_characteristic(labels, pred, label_value):
    # y_true and y_pred are the true labels and predicted probabilities, respectively
    fpr, tpr, thresholds = roc_curve(labels, pred, pos_label=label_value)

    # plot the ROC curve
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()