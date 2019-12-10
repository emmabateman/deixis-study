import os
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

feature_list = []
feature_dict = {}

def confusion(true, pred):
    error_vals = abs(true-pred)
    error_indices = []
    for i in range(len(error_vals)):
        if error_vals[i]:
            error_indices.append(i)
    print("incorrect prediction on the following training cases: {0}".format(error_indices))

    matrix = confusion_matrix(true, pred)
    true_this = matrix[0][0]
    true_that = matrix[1][1]
    false_this = matrix[1][0]
    false_that = matrix[0][1]
    print("\nCONFUSION MATRIX")
    print("                 prediction\n")
    print("                this\tthat")
    print("target    this\t{0}\t{1}".format(true_this, false_that))
    print("          that\t{0}\t{1}".format(false_this, true_that))

    accuracy = (true_this + true_that)/(true_this + true_that + false_this + false_that)
    precision_this = (true_this)/(true_this+false_this)
    precision_that = (true_that)/(true_that+false_that)
    recall_this = (true_this)/(true_this+false_that)
    recall_that = (true_that)/(true_that+false_this)
    f1_this = (2 * precision_this * recall_this)/(precision_this + recall_this)
    f1_that = (2 * precision_that * recall_that)/(precision_that + recall_that)

    print("\nACCURACY:\t{0}\n".format(accuracy))
    print("\t\tthis\t\tthat")
    print("PRECISION\t{0:.6f}\t{1:.6f}".format(precision_this, precision_that))
    print("RECALL\t\t{0:.6f}\t{1:.6f}".format(recall_this, recall_that))
    print("F1\t\t{0:.6f}\t{1:.6f}".format(f1_this, f1_that))
    print("\n")

def nn(train_features, train_classes, test_features, test_classes):
    classifier = MLPClassifier(hidden_layer_sizes=(10, 10))
    print("running multi-layer perceptron with 2 hidden layers of 10 neurons")
    classifier.fit(train_features, train_classes)
    predictions = classifier.predict(test_features)
    confusion(test_classes, predictions)

def svm(train_features, train_classes, test_features, test_classes):
    classifier = SVC(kernel="linear")
    print("running state vector machine algorithm with linear kernel")
    classifier.fit(train_features, train_classes)
    predictions = classifier.predict(test_features)
    confusion(test_classes, predictions)

def kneighbors(train_features, train_classes, test_features, test_classes):
    n = 3
    classifier = KNeighborsClassifier(n_neighbors=n)
    print("performing k-nearest-neighbor algorithm with {0} neighbors".format(n))
    classifier.fit(train_features, train_classes)
    predictions = classifier.predict(test_features)
    confusion(test_classes, predictions)

def most_common_baseline(train_features, train_classes, test_features, test_classes):
    print("performing most common baseline")
    predictions = [1 for i in range(len(test_classes))]
    confusion(test_classes, predictions)

def get_features(lines):
    global feature_list, feature_dict

    for line in lines:
        feat_list = line.split()[1:]
        for feat in feat_list:
            name, value = feat.split(':',1)
            try:
                value = int(value)
            except:
                name = feat
            if not name in feature_list:
                feature_dict[name] = len(feature_list)
                feature_list.append(name)

def get_vectors(lines):
    global feature_list, feature_dict

    classes = np.zeros(len(lines))
    features = np.zeros((len(lines),len(feature_list)))

    for i in range(len(lines)):
        line = lines[i]
        classes[i] = line.split()[0] == 'that'
        feat_list = line.split()[1:]
        for feat in feat_list:
            name, value = feat.split(':', 1)
            try:
                value = int(value)
                feat_idx = feature_dict[name]
            except:
                value = 1
                if feat in feature_list:
                    feat_idx = feature_dict[feat]
                else:
                    continue
            features[i, feat_idx] = value
    return features, classes

def main():
    global feature_list, feature_dict

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="use the test set instead of the dev set", action="store_true")
    args = parser.parse_args()
    if args.test:
        print("FINAL TEST!")
        test_dir = "test/"
    else:
        test_dir = "dev/"

    #get data
    f1 = open("train/data.txt")
    f2 = open(test_dir + "data.txt")
    train_lines = f1.readlines()
    test_lines = f2.readlines()
    get_features(train_lines)
    train_features, train_classes = get_vectors(train_lines)
    test_features, test_classes = get_vectors(test_lines)

    #basic statistics
    total = len(train_classes)
    num_that = sum(train_classes)
    num_this = total - num_that
    num_ref = sum(train_features[:,feature_dict['ref']])
    num_noref = total - num_ref
    print("The training data contains {0} 'this' pronouns ({1}%) and {2} 'that' pronouns ({3}%)".format(num_this, num_this*100/total, num_that, num_that*100/total))
    print("{0} of the pronouns ({1}%) have a referent in the text".format(num_ref, num_ref*100/total))

    #predictions
    most_common_baseline(train_features, train_classes, test_features, test_classes)
    kneighbors(train_features, train_classes, test_features, test_classes)
    svm(train_features, train_classes, test_features, test_classes)
    nn(train_features, train_classes, test_features, test_classes)

    #covariance
    print("TOP 10 FEATURES\n")
    correlation = [np.corrcoef(train_classes, train_features[:,i])[0,1] for i in range(len(feature_list))]

    correlation_dict = dict(zip(feature_list, correlation))
    sorted_features = sorted(correlation_dict.items(), key=lambda x: 1/abs(x[1]))

    for tup in sorted_features[:10]:
        print("{0}{1}{2}\t{3}".format(tup[0], ' '.join(['' for i in range(20 - len(tup[0]))]), abs(tup[1]), ['this', 'that'][tup[1]>0]))

main()
