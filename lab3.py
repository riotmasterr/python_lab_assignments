##A1
import numpy as np

# Function to calculate dot product
def calculate_dot_product(vector_a, vector_b):
   
    a = np.array(vector_a)
    b = np.array(vector_b)
    
    dot_prod = 0
    for i in range(len(a)):
        dot_prod += a[i] * b[i]
    
    return dot_prod


# Function to calculate Euclidean norm (length)
def calculate_euclidean_norm(vector):
   
    vec = np.array(vector)
    
    sum_squares = 0
    for i in range(len(vec)):
        sum_squares += vec[i] ** 2
    
    return np.sqrt(sum_squares)


# MAIN PROGRAM

# Define two vectors
A = [3, 4, 5, 6]
B = [1, 2, 3, 4]

print("Vector A:", A)
print("Vector B:", B)

# DOT PRODUCT
print("\n--- DOT PRODUCT ---")
my_dot = calculate_dot_product(A, B)
numpy_dot = np.dot(A, B)

print(f"My function:    {my_dot}")
print(f"NumPy function: {numpy_dot}")
print(f"Match? {np.allclose(my_dot, numpy_dot)}")

# EUCLIDEAN NORM
print("\n--- EUCLIDEAN NORM ---")
print("\nVector A:")
my_norm_a = calculate_euclidean_norm(A)
numpy_norm_a = np.linalg.norm(A)
print(f"My function:    {my_norm_a:.4f}")
print(f"NumPy function: {numpy_norm_a:.4f}")
print(f"Match? {np.allclose(my_norm_a, numpy_norm_a)}")

print("\nVector B:")
my_norm_b = calculate_euclidean_norm(B)
numpy_norm_b = np.linalg.norm(B)
print(f"My function:    {my_norm_b:.4f}")
print(f"NumPy function: {numpy_norm_b:.4f}")
print(f"Match? {np.allclose(my_norm_b, numpy_norm_b)}")


#a3
import matplotlib.pyplot as plt

#Calculates mean and variance of a feature vector
def calculate_feature_statistics(feature_vector):
    

    mean_value = np.mean(feature_vector)
    variance_value = np.var(feature_vector)
    return mean_value, variance_value

#main program
# takes the first feature
feature_index = 0
selected_feature = X_features[:, feature_index]

mean_feature, variance_feature = calculate_feature_statistics(selected_feature)

plt.hist(selected_feature, bins=20)
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.title("A3: Histogram of Feature 0")
plt.show()

print("A3: Mean of feature:", mean_feature)
print("A3: Variance of feature:", variance_feature)


#a3
import matplotlib.pyplot as plt

#Calculates mean and variance of a feature vector
def calculate_feature_statistics(feature_vector):
    

    mean_value = np.mean(feature_vector)
    variance_value = np.var(feature_vector)
    return mean_value, variance_value

#main program
# takes the first feature
feature_index = 0
selected_feature = X_features[:, feature_index]

mean_feature, variance_feature = calculate_feature_statistics(selected_feature)

plt.hist(selected_feature, bins=20)
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.title("A3: Histogram of Feature 0")
plt.show()

print("A3: Mean of feature:", mean_feature)
print("A3: Variance of feature:", variance_feature)


#A4
def minkowski_distance(vector_1, vector_2, p):
    # The function computes the Minkowski distance btw two feature vectors for a given p value
    distance = np.sum(np.abs(vector_1 - vector_2) ** p) ** (1 / p)
    return distance

# mainprogram
# selecting  two feature vectors from the dataset
feature_vector_1 = X_features[0]
feature_vector_2 = X_features[1]

# the range of p values
p_values = range(1, 11)

# list to be stored in the Minkowski distances
minkowski_distances = []

# calculate the Minkowski distance for each p
for p in p_values:
    distance = minkowski_distance(feature_vector_1, feature_vector_2, p)
    minkowski_distances.append(distance)

# plotting the Minkowski distance vs p
plt.plot(p_values, minkowski_distances, marker='o')
plt.xlabel("Value of p")
plt.ylabel("Minkowski Distance")
plt.title("A4: Minkowski Distance vs p")
plt.grid(True)
plt.show()

#A5
from scipy.spatial.distance import minkowski as scipy_minkowski
def minkowski_distance(vector_1, vector_2, p):
    # The function computes the Minkowski distance btw two feature vectors for a given p value
    distance = np.sum(np.abs(vector_1 - vector_2) ** p) ** (1 / p)
    return distance

# mainprogram

# using the same feature vectors as in A4
feature_vector_1 = X_features[0]
feature_vector_2 = X_features[1]

# selecting a p value for comparison
p_value = 3

# minkowski distance found useing the custom function
custom_minkowski_distance = minkowski_distance(
    feature_vector_1,
    feature_vector_2,
    p_value
)

# minkowski distance found  using the SciPy function
scipy_minkowski_distance = scipy_minkowski(
    feature_vector_1,
    feature_vector_2,
    p_value
)

# displaying the results
print(" Custom Minkowski Distance (p = 3):", custom_minkowski_distance)
print(" SciPy Minkowski Distance (p = 3):", scipy_minkowski_distance)


#A6
from sklearn.model_selection import train_test_split
def split_train_test(X_features, y_labels, test_size):
    # The function splits the dataset into training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y_labels,
        test_size=test_size,
        random_state=42
    )
    return X_train, X_test, y_train, y_test
#  main program

# Splitting the dataset into training set and testing set
X_train, X_test, y_train, y_test = split_train_test(
    X_features,
    y_labels,
    test_size=0.3
)

print(" Dataset split completed")
print(" Number of training samples:", len(X_train))
print(" Number of testing samples:", len(X_test))

#A7
from sklearn.neighbors import KNeighborsClassifier
def train_knn_classifier(X_train, y_train, k):
    # The function trains a kNN classifier using the training dataset forn a6
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    return knn_model
# main program

# training kNN classifier with k = 3
knn_model = train_knn_classifier(X_train, y_train, k=3)

print(" kNN classifier trained successfully with k = 3")

#A8
def calculate_knn_accuracy(knn_model, X_test, y_test):
    # the function calculates the accuracy of the  kNN classifier(trained)
    accuracy_value = knn_model.score(X_test, y_test)
    return accuracy_value
# mainprogram

# calculating the accuracy of the kNN classifier on test data
knn_accuracy = calculate_knn_accuracy(knn_model, X_test, y_test)

print("A8: Accuracy of kNN classifier (k = 3):", knn_accuracy)


#A9
def get_knn_predictions(knn_model, X_test):
    # The function predicts the class labels for the test feature vectors
    predicted_labels = knn_model.predict(X_test)
    return predicted_labels
#  main program

# Predicting the class labels for the test data
y_predicted_test = get_knn_predictions(knn_model, X_test)

print("A9: Predicted class labels for test dataset:")
print(y_predicted_test)

#A10
import numpy as np
def euclidean_distance(vector_1, vector_2):
    # The function calculates the Euclidean distance between two vectors
    distance = np.sqrt(np.sum((vector_1 - vector_2) ** 2))
    return distance


def custom_knn_single_prediction(X_train, y_train, test_vector, k):
    # The function predicts the class of a single test vector
    distance_label_list = []

    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], test_vector)
        distance_label_list.append((distance, y_train[i]))

    # Sorting based on  the distances
    distance_label_list.sort(key=lambda x: x[0])

    # Select k nearest neighbors
    k_nearest_neighbors = distance_label_list[:k]

    # Extract the labels of the nearest neighbors
    neighbor_labels = [label for _, label in k_nearest_neighbors]

    # Majority voting
    predicted_class = max(set(neighbor_labels), key=neighbor_labels.count)

    return predicted_class


def custom_knn_predict(X_train, y_train, X_test, k):
    # The function predicts the class labels for all test vectors
    predicted_labels = []

    for test_vector in X_test:
        label = custom_knn_single_prediction(X_train, y_train, test_vector, k)
        predicted_labels.append(label)

    return np.array(predicted_labels)

# Useing a smaller subset of data for custom kNN 
X_train_small = X_train[:100]
y_train_small = y_train[:100]
X_test_small = X_test[:20]

# mainprogram

y_predicted_custom_knn = custom_knn_predict(
    X_train_small,
    y_train_small,
    X_test_small,
    k=3
)

y_predicted_package_knn = knn_model.predict(X_test_small)

print("A10: Predictions using custom kNN:")
print(y_predicted_custom_knn)

print("A10: Predictions using package kNN:")
print(y_predicted_package_knn)



#A11
def calculate_accuracy_for_k_values(X_train, y_train, X_test, y_test, k_values):
    # the function calculates the accuracy of the custom kNN for different k values
    accuracy_list = []

    for k in k_values:
        predicted_labels = custom_knn_predict(X_train, y_train, X_test, k)
        accuracy = np.mean(predicted_labels == y_test)
        accuracy_list.append(accuracy)

    return accuracy_list

# main program

# Using the smaller subset for custom kNN 
X_train_small = X_train[:100]
y_train_small = y_train[:100]
X_test_small = X_test[:20]
y_test_small = y_test[:20]

# k values from 1 to 11
k_values = list(range(1, 12))

# Calculate the accuracy for each k
accuracy_values = calculate_accuracy_for_k_values(
    X_train_small,
    y_train_small,
    X_test_small,
    y_test_small,
    k_values
)

# Ploting the accuracy vs k graph
plt.plot(k_values, accuracy_values, marker='o')
plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("A11: Accuracy vs k for Custom kNN Classifier")
plt.grid(True)
plt.show()

# Display the accuracy values
for i in range(len(k_values)):
    print("A11: Accuracy for k =", k_values[i], ":", accuracy_values[i])



#A12
def compute_confusion_matrix(y_actual, y_predicted):
    # The function computes the TP, TN, FP, FN values
    tp = tn = fp = fn = 0

    for i in range(len(y_actual)):
        if y_actual[i] == 1 and y_predicted[i] == 1:
            tp += 1
        elif y_actual[i] == 0 and y_predicted[i] == 0:
            tn += 1
        elif y_actual[i] == 0 and y_predicted[i] == 1:
            fp += 1
        elif y_actual[i] == 1 and y_predicted[i] == 0:
            fn += 1

    return tp, tn, fp, fn


def calculate_accuracy(tp, tn, fp, fn):
    # Accuracy metric
    return (tp + tn) / (tp + tn + fp + fn)


def calculate_precision(tp, fp):
    # Precision metric
    return tp / (tp + fp) if (tp + fp) != 0 else 0


def calculate_recall(tp, fn):
    # Recall metric
    return tp / (tp + fn) if (tp + fn) != 0 else 0


def calculate_f1_score(precision, recall):
    # F1-score metric
    return (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# output

# Using the small subset from A11
y_predicted_custom_knn = custom_knn_predict(
    X_train_small,
    y_train_small,
    X_test_small,
    k=3
)

# Compute the confusion matrix values
tp, tn, fp, fn = compute_confusion_matrix(
    y_test_small,
    y_predicted_custom_knn
)

# Calculate the performance metrics
accuracy = calculate_accuracy(tp, tn, fp, fn)
precision = calculate_precision(tp, fp)
recall = calculate_recall(tp, fn)
f1_score = calculate_f1_score(precision, recall)

# Displaying results
print("A12: Confusion Matrix Values")
print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)

print("A12: Accuracy:", accuracy)
print("A12: Precision:", precision)
print("A12: Recall:", recall)
print("A12: F1 Score:", f1_score)

#A13
def calculate_accuracy(tp, tn, fp, fn):
    #  accuracy from confusion matrix values
    return (tp + tn) / (tp + tn + fp + fn)


def calculate_precision(tp, fp):
    #  precision function
    if (tp + fp) == 0:
        return 0
    return tp / (tp + fp)


def calculate_recall(tp, fn):
    #  recall function
    if (tp + fn) == 0:
        return 0
    return tp / (tp + fn)


def calculate_f_beta_score(precision, recall, beta):
    #  F-beta score
    if (precision + recall) == 0:
        return 0
    return ((1 + beta ** 2) * precision * recall) / ((beta ** 2 * precision) + recall)


#mani code

# Using confusion matrix the values are obtained in A12
accuracy_value = calculate_accuracy(tp, tn, fp, fn)
precision_value = calculate_precision(tp, fp)
recall_value = calculate_recall(tp, fn)
f1_score_value = calculate_f_beta_score(
    precision_value,
    recall_value,
    beta=1
)

print("A13: Performance Metrics using Own Functions")
print("Accuracy:", accuracy_value)
