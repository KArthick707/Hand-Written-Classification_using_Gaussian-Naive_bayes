import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(train_file, test_file):
    # Here I have loaded the MNIST dataset from the CSV files
    train_data = np.genfromtxt(train_file, delimiter=",")
    test_data = np.genfromtxt(test_file, delimiter=",")

    X_train = train_data[:, 1:]  # feature of the training dataset representing the pixel value
    y_train = train_data[:, 0]  # lable of the training dataset
    X_test = test_data[:, 1:]  # feature of the testing dataset
    y_test = test_data[:, 0]  # lable of  the testing dataset

    return X_train, y_train, X_test, y_test


def precision_from_confusion_matrix(Conf_mat):
    num_class = Conf_mat.shape[0]
    precision = np.zeros(num_class)
    for i in range(num_class):
        true_positives = Conf_mat[i, i]
        false_positives = np.sum(Conf_mat[:, i]) - true_positives
        precision[i] = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) != 0 else 0

    return precision


def recall_from_confusion_matrix(Conf_mat):
    num_classes = Conf_mat.shape[0]
    recall = np.zeros(num_classes)
    for i in range(num_classes):
        true_positives = Conf_mat[i, i]
        false_negatives = np.sum(Conf_mat[i, :]) - true_positives
        recall[i] = true_positives / (true_positives + false_negatives) \
            if (true_positives + false_negatives) != 0 else 0
    return recall


# Calculating the value of F1 Score
def f1_score_from_confusion_matrix(Conf_mat):
    num_classes = Conf_mat.shape[0]
    f1_scores = np.zeros(num_classes)

    for i in range(num_classes):
        true_positives = Conf_mat[i, i]
        false_positives = np.sum(Conf_mat[:, i]) - true_positives
        false_negatives = np.sum(Conf_mat[i, :]) - true_positives
        precision = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) \
            if (true_positives + false_negatives) != 0 else 0
        f1_scores[i] = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) != 0 else 0

    return f1_scores


# Calculating Macroaverage of the Precision
def macro_average_precision_score(Precision):
    num_class = len(Precision)
    macro_average_P = sum(Precision) / num_class
    return macro_average_P


# Calculating Macroaverage of the Recall
def recall_macro_average(Recall):
    num_class = len(Recall)
    macro_average_R = sum(Recall) / num_class
    return macro_average_R


# Calculating Macroaverage of the F1_Score
def F1_macro_average(Precision, Recall):
    total_precision = sum(Precision)
    total_recall = sum(Recall)
    macro_precision = total_precision / len(Precision)
    macro_recall = total_recall / len(Recall)
    macro_average_F1 = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)

    return macro_average_F1




# 2nd way of the code
# def F1_macro_average(F1_Score):
# total_f1_score = sum(F1_Score)
# num_classes = len(F1_Score)
# macro_average_F1 = total_f1_score / num_classes
# return macro_average_F1


def train_and_evaluate_naive_bayes(X_train, y_train, X_test, y_test):
    GNB = GaussianNB()
    GNB.fit(X_train, y_train)  # trains the data with function
    # GNB.partial_fit(X_train, y_train)
    y_pred = GNB.predict(X_test)  # calculates the posterior probabilities for each class

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:\n", accuracy)

    Conf_mat = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", Conf_mat)

    Precision = precision_from_confusion_matrix(Conf_mat)
    print("Precision:\n", Precision)

    Recall = recall_from_confusion_matrix(Conf_mat)
    print("Recall:\n", Recall)

    F1_Score = f1_score_from_confusion_matrix(Conf_mat)
    print("F1 Score:\n", F1_Score)

    Macro_average_P = macro_average_precision_score(Precision)
    print("Macroaverage_Precision:\n", Macro_average_P)

    Macro_average_R = recall_macro_average(Recall)
    print("Macroaverage_Recall:\n", Macro_average_R)

    Macro_average_F1_score = F1_macro_average(Precision, Recall)
    print("Macro_average_F1_score:\n", Macro_average_F1_score)



def main():
    train_file = r"C:\\Users\\karthick\\Downloads\\mnist_train.csv"
    test_file = r"C:\\Users\\karthick\\Downloads\\mnist_test.csv"

    X_train, y_train, X_test, y_test = load_data(train_file, test_file)

    train_and_evaluate_naive_bayes(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
