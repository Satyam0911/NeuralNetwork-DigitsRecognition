from math import exp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the MNIST dataset
data = pd.read_csv('mnist_train.csv')

data = np.array(data)
split_index = int(0.8 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]
x_train = train_data[:, 1:]  # Features
y_train = train_data[:, 0]    # Labels
x_test = test_data[:, 1:]      # Features
y_test = test_data[:, 0]       # Labels

HIDDEN_SIZE = 64

def init_params():
    w1 = np.random.randn(HIDDEN_SIZE, 784) * np.sqrt(2. / 784)  # He initialization
    b1 = np.zeros((HIDDEN_SIZE, 1))
    w2 = np.random.randn(10, HIDDEN_SIZE) * np.sqrt(2. / HIDDEN_SIZE)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # stability
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x.T) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)  # correct usage
    return z1, a1, z2, a2

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y.T  # shape (10, m)

def deriv_ReLU(Z):
    return Z > 0

def back_prop(z1, a1, z2, a2, w2, x, y):
    m = y.size
    one_hot_y = one_hot(y)
    dZ2 = a2 - one_hot_y
    dW2 = (1 / m) * dZ2.dot(a1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = w2.T.dot(dZ2) * deriv_ReLU(z1)
    dW1 = (1 / m) * dZ1.dot(x)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2, axis=0)

def get_accuracy(predictions, y):
    return np.mean(predictions == y)

def gradient_descent(x, y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 50 == 0:
            acc = get_accuracy(get_predictions(a2), y)
            print(f"Iteration {i} | Accuracy: {acc:.4f}")
    return w1, b1, w2, b2

x_train = x_train.reshape(x_train.shape[0], 784) / 255.0  # Normalize and flatten
w1, b1, w2, b2 = gradient_descent(x_train, y_train, iterations=200, alpha=0.1)

def get_predict(x, w1, b1, w2, b2):
    """
    Predicts labels for the given input data using trained parameters.
    
    Parameters:
    - x: input data of shape (num_samples, 784)
    - w1, b1, w2, b2: trained weights and biases

    Returns:
    - predictions: array of predicted class labels
    """
    x = x.reshape(x.shape[0], 784) / 255.0  # Flatten and normalize
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, x)
    predictions = get_predictions(a2)
    return predictions

def test_predict(index, x_test, y_test, w1, b1, w2, b2):
    """
    Predict and display one test sample.
    """
    x = x_test[index].reshape(1, 784) / 255.0
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, x)
    pred = get_predictions(a2)[0]
    true = y_test[index]

    # Show image in original shape
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {pred}\nActual: {true}")
    plt.axis('off')
    plt.show()
test_predict(7, x_test, y_test, w1, b1, w2, b2)