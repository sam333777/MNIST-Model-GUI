import numpy as np
import pandas as pd
from matplotlib import pyplot as plt   

data = pd.read_csv('/Users/samuel/Desktop/SUMMER/numpy:math model/train.csv').to_numpy()
m, n = data.shape
np.random.shuffle(data)                    

data_dev  = data[:1000].T                
Y_dev     = data_dev[0]
X_dev     = data_dev[1:n] / 255.0

data_train = data[1000:].T                 
Y_train    = data_train[0]
X_train    = data_train[1:n] / 255.0
_, m_train = X_train.shape                    

def init_params():
    W1 = np.random.randn(10, 784) * np.sqrt(2/784)
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10)  * np.sqrt(2/10)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2


def relu(Z):
    return np.maximum(0, Z)


def relu_deriv(Z):
    return Z > 0


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def one_hot(Y):
    n_classes = int(Y.max() + 1)
    one_hot_Y = np.zeros((n_classes, Y.size))
    one_hot_Y[Y.astype(int), np.arange(Y.size)] = 1
    return one_hot_Y


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2 @ A1.T
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T @ dZ2 * relu_deriv(Z1)
    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, grads, alpha):
    dW1, db1, dW2, db2 = grads
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(preds, Y):
    return np.mean(preds == Y)



def gradient_descent(X, Y, alpha=0.1, iterations=500, print_every=10):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        grads = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, grads, alpha)

        if i % print_every == 0 or i == iterations - 1:
            acc = get_accuracy(get_predictions(A2), Y)
            print(f"iter {i:4d}  |  accuracy: {acc:.4f}")

    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)

def test_prediction(index,W1, b1, W2, b2):
    current_image = X_train[:, index, None]  
    predictions = make_predictions(X_train[:,index,None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction:", predictions)
    print("label:", label)
    
    current_image = current_image.reshape(28, 28)  *255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.1, iterations=3000)

    np.savez('mnist_weights.npz', W1=W1, b1=b1, W2=W2, b2=b2)
