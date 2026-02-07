## ASSIGNMENT -4
##A1
'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Split dataset into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train kNN classifier
def train_knn(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)  # Create kNN model
    model.fit(X_train, y_train)  # Train model
    return model

# Evaluate model performance
def evaluate_model(model, X, y):
    y_pred = model.predict(X)  # Predict labels
    cm = confusion_matrix(y, y_pred)  # Confusion matrix
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    return cm, precision, recall, f1

# Load dataset
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

# Train-test split
X_train, X_test, y_train, y_test = split_data(X, y)

# Train classifier
model = train_knn(X_train, y_train, k=3)

# Evaluate on training and testing sets
cm_train, p_train, r_train, f1_train = evaluate_model(model, X_train, y_train)
cm_test, p_test, r_test, f1_test = evaluate_model(model, X_test, y_test)

# Print metrics
print("TRAIN CONFUSION MATRIX:\n", cm_train)
print("Train Precision:", p_train)
print("Train Recall:", r_train)
print("Train F1-score:", f1_train)

print("\nTEST CONFUSION MATRIX:\n", cm_test)
print("Test Precision:", p_test)
print("Test Recall:", r_test)
print("Test F1-score:", f1_test)


##A2

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train kNN regressor
def train_knn_regressor(X_train, y_train, k=3):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

# Calculate regression performance metrics
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Percentage error
    r2 = r2_score(y_true, y_pred)  # R² score
    return mse, rmse, mape, r2

# Train regression model
X_train, X_test, y_train, y_test = split_data(X, y)
model = train_knn_regressor(X_train, y_train)
y_pred = model.predict(X_test)

# Print metrics
mse, rmse, mape, r2 = regression_metrics(y_test, y_pred)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE (%):", mape)
print("R2 Score:", r2)

##A3
import numpy as np
import matplotlib.pyplot as plt

# Generate 20 random 2D points and assign classes
def generate_training_data(n=20):
    X = np.random.uniform(1, 10, (n, 2))
    y = np.where(X[:, 0] + X[:, 1] > 10, 1, 0)  # Rule-based labeling
    return X, y

X_train, y_train = generate_training_data()

# Plot data points
plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], color='blue', label='Class 0')
plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], color='red', label='Class 1')
plt.title("Training Data Scatter Plot")
plt.legend()
plt.show()


##A4
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# ---------------- FUNCTION DEFINITIONS ----------------

# Generate 20 random training points with 2 features
def generate_training_data(n=20):
    X = np.random.uniform(1, 10, (n, 2))  # Feature values between 1 and 10
    y = np.where(X[:, 0] + X[:, 1] > 10, 1, 0)  # Class assignment rule
    return X, y

# Train kNN classifier
def train_knn(X, y, k=3):
    model = KNeighborsClassifier(n_neighbors=k)  # k = number of neighbors
    model.fit(X, y)  # Fit model to training data
    return model

# Create test grid (approx 10,000 points)
def generate_test_grid():
    x = np.arange(0, 10, 0.1)
    y = np.arange(0, 10, 0.1)
    xx, yy = np.meshgrid(x, y)  # 2D coordinate grid
    grid = np.c_[xx.ravel(), yy.ravel()]  # Convert to list of points
    return xx, yy, grid

# Plot decision regions and training data
def plot_results(X_train, y_train, xx, yy, Z):
    plt.figure(figsize=(8, 6))

    # Decision region background
    plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.3)

    # Plot training points
    plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1],
                color='blue', label='Class 0')
    plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1],
                color='red', label='Class 1')

    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("kNN Decision Boundary (k=3)")
    plt.legend()
    plt.show()

# ---------------- MAIN PROGRAM ----------------

X_train, y_train = generate_training_data()
model = train_knn(X_train, y_train)  # Default k = 3

xx, yy, grid = generate_test_grid()
Z = model.predict(grid)  # Predict class for each grid point

plot_results(X_train, y_train, xx, yy, Z)


##A5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# ---------------- FUNCTION DEFINITIONS ----------------

def generate_training_data(n=20):
    X = np.random.uniform(1, 10, (n, 2))
    y = np.where(X[:, 0] + X[:, 1] > 10, 1, 0)
    return X, y

def generate_test_grid():
    x = np.arange(0, 10, 0.1)
    y = np.arange(0, 10, 0.1)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid

# Plot boundary for different k values
def plot_knn_boundary(X_train, y_train, k, xx, yy, grid):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    Z = model.predict(grid)

    plt.figure(figsize=(7,6))
    plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.3)

    # Training points
    plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1],
                color='blue', label='Class 0')
    plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1],
                color='red', label='Class 1')

    plt.title(f"kNN Decision Boundary (k={k})")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.legend()
    plt.show()

# ---------------- MAIN PROGRAM ----------------

X_train, y_train = generate_training_data()
xx, yy, grid = generate_test_grid()

# Test multiple k values to observe boundary changes
for k in [1, 3, 5, 9]:
    plot_knn_boundary(X_train, y_train, k, xx, yy, grid)


##A6

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# ---------------- FUNCTION DEFINITIONS ----------------

# Load dataset and keep only two features and two classes
def load_and_filter_data(feature_file, label_file):
    X = np.load(feature_file)  # Load feature matrix
    y = np.load(label_file)    # Load labels

    X = X[:, [0, 1]]  # Select only first two features for 2D visualization

    # Select only first two classes for binary classification
    classes = np.unique(y)
    class0, class1 = classes[0], classes[1]
    mask = (y == class0) | (y == class1)

    return X[mask], y[mask], class0, class1


# Create grid of test points covering feature space
def generate_grid(X):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    # Meshgrid creates coordinate pairs
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    grid = np.c_[xx.ravel(), yy.ravel()]  # Convert grid to list of points
    return xx, yy, grid


# Plot original training data
def plot_training_data(X, y, class0, class1):
    plt.figure(figsize=(7,6))

    # Plot points belonging to class0 and class1
    plt.scatter(X[y==class0][:,0], X[y==class0][:,1],
                color='blue', label=f'Class {class0}')
    plt.scatter(X[y==class1][:,0], X[y==class1][:,1],
                color='red', label=f'Class {class1}')

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Project Data Training Scatter Plot")
    plt.legend()
    plt.show()


# Plot decision boundary for given k
def plot_decision_boundary(X, y, k, xx, yy, grid, class0, class1):
    model = KNeighborsClassifier(n_neighbors=k)  # Initialize kNN
    model.fit(X, y)  # Train model

    Z = model.predict(grid)  # Predict class for all grid points

    plt.figure(figsize=(7,6))

    # Background classification regions
    plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.3)

    # Training points
    plt.scatter(X[y==class0][:,0], X[y==class0][:,1],
                color='blue', label=f'Class {class0}')
    plt.scatter(X[y==class1][:,0], X[y==class1][:,1],
                color='red', label=f'Class {class1}')

    plt.title(f"kNN Decision Boundary (k={k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


# ---------------- MAIN PROGRAM ----------------

# Load and filter project dataset
X, y, class0, class1 = load_and_filter_data("X_features.npy", "y_labels.npy")

# Plot training data distribution (A3-style)
plot_training_data(X, y, class0, class1)

# Generate grid for decision boundary visualization
xx, yy, grid = generate_grid(X)

# A4: k=3 decision boundary
plot_decision_boundary(X, y, k=3, xx=xx, yy=yy, grid=grid, class0=class0, class1=class1)

# A5: Observe boundary change for multiple k values
for k in [1, 5, 11, 21]:
    plot_decision_boundary(X, y, k, xx, yy, grid, class0, class1)
'''

##A7
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# ---------------- FUNCTION DEFINITIONS ----------------

# Load dataset and keep only 2 features and 2 classes
def load_and_filter_data(feature_file, label_file):
    X = np.load(feature_file)  # Load feature matrix
    y = np.load(label_file)    # Load class labels

    X = X[:, [0, 1]]  # Select first two features for 2D classification

    # Keep only first two classes for binary classification
    classes = np.unique(y)
    class0, class1 = classes[0], classes[1]
    mask = (y == class0) | (y == class1)

    return X[mask], y[mask]


# Perform GridSearch to find best k value
def perform_grid_search(X_train, y_train):
    # Define range of k values to test
    param_grid = {'n_neighbors': list(range(1, 31))}

    knn = KNeighborsClassifier()  # Initialize classifier

    # 5-fold cross-validation used to evaluate each k
    grid_search = GridSearchCV(knn, param_grid, cv=5)

    grid_search.fit(X_train, y_train)  # Train models

    return grid_search


# ---------------- MAIN PROGRAM ----------------

# Load filtered project dataset
X, y = load_and_filter_data("X_features.npy", "y_labels.npy")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Perform hyperparameter tuning
grid_search = perform_grid_search(X_train, y_train)

# Display best results
print("Best k value:", grid_search.best_params_['n_neighbors'])
print("Best cross-validation score:", grid_search.best_score_)
