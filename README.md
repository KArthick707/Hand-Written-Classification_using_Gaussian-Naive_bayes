# Hand-Written-Classification_using_Gaussian-Naive_bayes

This Python script is designed to perform handwritten digit classification using the Naive Bayes algorithm on the MNIST dataset. Here's a brief summary of its main components:

1. Data Loading: The `load_data` function reads the MNIST dataset from CSV files. It separates the features (pixel values) and labels for both the training and testing datasets.

2. Model Training and Prediction: The `train_and_evaluate_naive_bayes` function trains a Gaussian Naive Bayes model on the training data and makes predictions on the testing data.

3. Evaluation: The same function also evaluates the model's performance by calculating the accuracy and confusion matrix. It then calculates precision, recall, and F1 score from the confusion matrix.

4. Macro Averages: The `macro_average_precision_score`, `recall_macro_average`, and `F1_macro_average` functions calculate the macro-average precision, recall, and F1 score, respectively.

5. Execution: The script then loads the MNIST dataset, trains and evaluates the Naive Bayes model, and prints the evaluation metrics.

In essence, this script is a complete pipeline for performing handwritten digit classification using a Gaussian Naive Bayes model.
