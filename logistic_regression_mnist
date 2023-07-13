import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Fetch the MNIST dataset
print("Fetching data")
mnist = fetch_openml('mnist_784')

print("Data Fetched")
X, y = mnist.data, mnist.target

# Convert string labels to integers
y = y.astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature values to be between 0 and 1
X_train /= 255.0
X_test /= 255.0

# Add bias term to feature matrix
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Define the softmax function
def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

# Define the loss function (cross-entropy)
def loss(y, y_pred):
    return -np.mean(np.sum(y * np.log(y_pred), axis=1))

# Define the gradient descent algorithm
def gradient_descent(X, y, learning_rate, num_iterations):
    num_samples, num_features = X.shape
    num_classes = len(np.unique(y))
    W = np.random.randn(num_features, num_classes)

    for i in tqdm(range(num_iterations)):
        scores = np.dot(X, W)
        probs = softmax(scores)
        grad = (1 / num_samples) * np.dot(X.T, (probs - one_hot_encode(y, num_classes)))
        W -= learning_rate * grad

    return W

# Define the one-hot encoding function
def one_hot_encode(labels, num_classes):
    num_samples = len(labels)
    one_hot = np.zeros((num_samples, num_classes))
    one_hot[np.arange(num_samples), labels] = 1
    return one_hot

# Train the logistic regression model
learning_rate = 0.01
num_iterations = 800
W = gradient_descent(X_train, y_train, learning_rate, num_iterations)

# Make predictions on the testing set
scores = np.dot(X_test, W)
probs = softmax(scores)
predictions = np.argmax(probs, axis=1)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)