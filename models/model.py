# %%
import cupy as np
import pandas as pd
import time
INPUT_NODES = 784  
OUTPUT_NODES = 10
data = pd.read_csv('train.csv')
gfh =  100
dfh  = 100
epochs = 50
learning_rate = 0.05
batch_size =   32

# %%
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0].astype(int)
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0].astype(int)
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

# %%
def acc_func(Z, alpha=1.0):
    return np.maximum(0, Z)

def acc_func_der(Z, alpha=1.0):
    return (Z > 0).astype(float)

def init_params():
    W1 = np.random.rand(gfh, INPUT_NODES) - 0.5
    b1 = np.random.rand(gfh, 1) - 0.5
    W2 = np.random.rand(dfh, gfh) - 0.5
    b2 = np.random.rand(dfh, 1) - 0.5
    W3 = np.random.rand(OUTPUT_NODES, dfh) - 0.5
    b3 = np.random.rand(OUTPUT_NODES, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

def softmax(Z):
    eZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return eZ / eZ.sum(axis=0, keepdims=True)

def f_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = acc_func(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = acc_func(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def one_hot(Y):
    one_hot_Y = np.zeros((OUTPUT_NODES, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def b_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = W3.T.dot(dZ3) * acc_func_der(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * acc_func_der(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

def change_prms(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

# %%
def getPred(A3):
    return np.argmax(A3, 0)

def getAcc(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations, batch_size=2048):
    W1, b1, W2, b2, W3, b3 = init_params()
    m_train = X.shape[1]

    for i in range(iterations):
        perm = np.random.permutation(m_train) 

        for start in range(0, m_train, batch_size):
            idx = perm[start:start+batch_size] 
            X_batch = X[:, idx]
            Y_batch = Y[idx]

            Z1, A1, Z2, A2, Z3, A3 = f_prop(W1, b1, W2, b2, W3, b3, X_batch)
            dW1, db1, dW2, db2, dW3, db3 = b_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X_batch, Y_batch)
            W1, b1, W2, b2, W3, b3 = change_prms(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        if i % 10 == 0:
            print(f"Iteration {i}")
        if i % 100 == 0:
            np.savez(f"mnist_model_epoch_{i}.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

    return W1, b1, W2, b2, W3, b3




# %%
start_time = time.time()
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, learning_rate, epochs, batch_size)
print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))

# %%
def makePred(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = f_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = getPred(A3)
    return predictions
# %%
dev_pred = makePred(X_dev, W1, b1, W2, b2, W3, b3)
print(getAcc(dev_pred, Y_dev))

# %%
np.savez("mnist_model.npz",
         W1=W1, b1=b1,
         W2=W2, b2=b2,
         W3=W3, b3=b3)