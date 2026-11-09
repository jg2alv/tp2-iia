import numpy as np
np.random.seed(42)

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_mnist():
    # Load MNIST data from sklearn
    mnist = fetch_openml('mnist_784', as_frame=False, cache=True, version=1)
    X, y = mnist["data"], mnist["target"].astype(int)

    # Normalize the data
    X = X / 255.0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))

    return X_train, X_test, y_train, y_test

def initialize_parameters(input_size, hidden_size, output_size):
    pass

# Activation functions
def relu(Z):
    pass

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    pass

# Forward pass
def forward(X, parameters):
    pass

# Loss function
def cross_entropy_loss(y_true, y_pred):
    pass

# Backward pass
def backward(X, y, W2, Z1, A1, A2):
    n_samples = X.shape[0]
    
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / n_samples
    db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples
    dA1 = np.dot(dZ2, W2.T)
    
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / n_samples
    db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples
    
    return dW1, db1, dW2, db2

def gradient_descent_step(X, y, parameters, learning_rate=0.01):
    pass

def accuracy(y_true, y_pred):
    pass
    
    

    










