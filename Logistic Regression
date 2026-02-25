'''
linear regression would be useless when we need a probability instead of a number.
Because it can return any number like 2, -1, 1000, which is not a probability.
logistic regression classifies your data based on thresholding between 0 and 1 accountuing for ptobability.
the function used here is either sigmoid or softmax.
the sigmoid function is as follows:
sigmoid(x) = 1 / (1 + e^(-x))
the softmax function is as follows:
softmax(x_i) = e^(x_i) / sum(e^(x_j)) for all j in the output classes 
logistic regression assumes log-odds and features are linear. 
logistic regression gives an s shaped but still linear beacuse the decision boundary is linear.
 which means that the decision boundary is a straight line in 2D yet a plane in 3D.

We will be completing this in several phases as follows:
1. pure python linear algebra like numpy alternative
2. Activation funcytions from scratch
3. defining loss
4. handling weights
5. forward and backward pass
6. a simple mini batch gradient descent 
7. model suite 
8. evaluation on custom data


this might not match exactly with how sklearn deals with logistic regression class yet it's jsut for understanding and practice
'''

def make_matrix(rows, cols, fill=0.0):
    return [[fill for _ in range(cols)] for _ in range(rows)]

def shape(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    return (rows, cols)

def transpose(matrix):
    rows , cols = shape(matrix)
    transposed = make_matrix(cols, rows)
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    return transposed


def matrix_multiplication(A,B):
    r , c = shape(A)
    c1 , d = shape(B)
    assert c == c1
    result = make_matrix(r, d)
    for i in range(r):
        for j in range(d):
            for k in range(c):
                result[i][j] += A[i][k] * B[k][j]
    return result 

def mat_add(A,B):
    row , col = shape(A)
    row1 , col1 = shape(B)
    assert row == row1 and col == col1
    temp = make_matrix(row , col)
    for i in range(row):
        for j in range(col):
            temp[i][j] = A[i][j] + B[i][j]

    return temp


def mat_scale(A, scalar):
    row , col = shape(A)
    return [[A[i][j] * scalar for j in range(col)] for i in range(row)]


def mat_elementwise_mul(A,B):
    r , c = shape(A)
    r1 , c1 = shape(B)
    assert r == r1 and c == c1
    return [[A[i][j] * B[i][j] for j in range(c)] for i in range(r)]

def mat_sub(A,B):
    r,c = shape(A)
    r1 , c1 = shape(B)
    assert r == r1 and c == c1
    return [[A[i][j] - B[i][j] for j in range(c)] for i in range(r)]

def clip(x , max_val , min_val):
    return max(min_val , min(max_val , x))

def apply_elementwise(matrix, func):
    row , col = shape(matrix)
    return[[func(matrix[i][j]) for j in range(col)] for i in range(row)]

def broadcast_add(matrix, bias_row):
    r, c = shape(matrix)
    result = make_matrix(r, c)
    for i in range(r):
        for j in range(c):
            result[i][j] = matrix[i][j] + bias_row[0][j]
    return result

def sum_cols(matrix):
    r, c = shape(matrix)
    result = make_matrix(1, c)
    for j in range(c):
        for i in range(r):
            result[0][j] += matrix[i][j]
    return result

e = 2.718281828459045

def sigmoid(x):
    x = clip(x , 500, -500)
    return 1/1 - e**-x

def sigmoid_derivative(x):
    x = sigmoid(x)
    return 1.0 / (1.0 + e ** (-x))

def sigmoid_matrix(A):
    return apply_elementwise(A ,sigmoid )

def sigmoid_derivative_matrix(A):
    return apply_elementwise(A,sigmoid_derivative )


def softmax(v):
    max_val = max(v)
    exponents = [e ** (val - max_val) for val in v]
    sum_exps = sum(exponents)
    return [i/sum_exps for i in exponents]

def softmax_matrix(A):
    return [softmax(i) for i in A]

from math import log
print(log(10))


def binary_cross_entropy(Y, Y_hat, epsilon=1e-12):
    n = len(Y)
    total = 0.0
    for i in range(n):
        y     = Y[i][0]                                       
        y_hat = clip(Y_hat[i][0], 1.0 - epsilon, epsilon)    
        total += y * log(y_hat) + (1.0 - y) * log(1.0 - y_hat)  #
    return -total / n


def categorical_cross_entropy(Y_onehot, Y_hat_probs, epsilon=1e-12):
    n = len(Y_onehot)
    total = 0.0
    for i in range(n):
        for j in range(len(Y_onehot[i])):
            p = clip(Y_hat_probs[i][j], epsilon, 1.0 - epsilon)
            total += Y_onehot[i][j] * log(p)
    return -total / n



class SimpleLCG:
    def __init__(self, seed=42):
        self.state = seed
        self.m = 2**32
        self.a = 1664525
        self.c = 1013904223

    def rand_int(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def rand_float(self):
        return self.rand_int() / self.m

    def rand_normal(self):
        u1 = max(self.rand_float(), 1e-10)  # avoid log(0)
        u2 = self.rand_float()
        pi = 3.141592653589793
        angle = 2 * pi * u2
        cos_val = 0.0
        term = 1.0
        for k in range(1, 20):
            cos_val += term
            term *= -angle * angle / ((2*k) * (2*k - 1))
        magnitude = (-2.0 * log(u1)) ** 0.5
        return magnitude * cos_val

rng = SimpleLCG(seed=42)

def initialize_weights(rows, cols, scale=0.01):
    result = make_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = rng.rand_normal() * scale
    return result

def initialize_zeros(rows, cols):
    return make_matrix(rows, cols, 0.0)


def forward_binary(X, W, b):
    Z = matrix_multiplication(X, W)              
    Z = broadcast_add(Z, b)      
    A = sigmoid_matrix(Z)         

    return Z, A

def forward_multiclass(X, W, b):
    Z = matrix_multiplication(X, W)
    Z = broadcast_add(Z, b)
    A = softmax_matrix(Z)         
    return Z, A


def backward_binary(X, Y, A):
    n = len(X)

    # dL/dZ = A - Y  (the beautiful collapse of chain rule)
    dZ = mat_sub(A, Y)                          # (n_samples, 1)

    # dL/dW = (1/n) * X^T @ dZ
    X_T = transpose(X)                          # (n_features, n_samples)
    dW  = matrix_multiplication(X_T, dZ)                      # (n_features, 1)
    dW  = mat_scale(dW, 1.0 / n)

    # dL/db = (1/n) * sum(dZ)  — sum over all samples
    db = sum_cols(dZ)                           # (1, 1)
    db = mat_scale(db, 1.0 / n)

    return dW, db

def backward_multiclass(X, Y_onehot, A):
    n = len(X)
    dZ = mat_sub(A, Y_onehot)                   # (n_samples, n_classes)
    X_T = transpose(X)
    dW  = matrix_multiplication(X_T, dZ)
    dW  = mat_scale(dW, 1.0 / n)
    db  = sum_cols(dZ)
    db  = mat_scale(db, 1.0 / n)
    return dW, db


def update_parameters(W, b, dW, db, learning_rate):
    W_scaled = mat_scale(dW, learning_rate)
    b_scaled  = mat_scale(db,  learning_rate)
    W_new = mat_sub(W, W_scaled)
    b_new = mat_sub(b, b_scaled)
    return W_new, b_new


def create_batches(X, Y, batch_size, shuffle=True):
    n = len(X)
    indices = list(range(n))

    if shuffle:
        # Fisher-Yates shuffle — pure Python
        for i in range(n - 1, 0, -1):
            j = rng.rand_int() % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]

    batches = []
    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        X_batch = [X[i] for i in batch_idx]
        Y_batch = [Y[i] for i in batch_idx]
        batches.append((X_batch, Y_batch))
    return batches

# ============================================================
# LOGISTIC REGRESSION MODEL — COMPLETE
# ============================================================

class LogisticRegression:

    def __init__(self, n_features, n_classes=2, learning_rate=0.1,
                 n_epochs=100, batch_size=32, verbose=True):
        self.lr          = learning_rate
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.verbose     = verbose
        self.n_classes   = n_classes
        self.mode        = 'binary' if n_classes == 2 else 'multiclass'

        if self.mode == 'binary':
            self.W = initialize_weights(n_features, 1)
            self.b = initialize_zeros(1, 1)
        else:
            self.W = initialize_weights(n_features, n_classes)
            self.b = initialize_zeros(1, n_classes)

        self.loss_history = []

    def _forward(self, X):
        if self.mode == 'binary':
            return forward_binary(X, self.W, self.b)
        else:
            return forward_multiclass(X, self.W, self.b)

    def _loss(self, Y, A):
        if self.mode == 'binary':
            return binary_cross_entropy(Y, A)
        else:
            return categorical_cross_entropy(Y, A)

    def _backward(self, X, Y, A):
        if self.mode == 'binary':
            return backward_binary(X, Y, A)
        else:
            return backward_multiclass(X, Y, A)

    def fit(self, X, Y):
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            batches = create_batches(X, Y, self.batch_size)

            for X_batch, Y_batch in batches:
                Z, A = self._forward(X_batch)

                batch_loss = self._loss(Y_batch, A)
                epoch_loss += batch_loss

                dW, db = self._backward(X_batch, Y_batch, A)

                self.W, self.b = update_parameters(
                    self.W, self.b, dW, db, self.lr
                )

            avg_loss = epoch_loss / len(batches)
            self.loss_history.append(avg_loss)

            if self.verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                print(f"Epoch {epoch+1:>4}/{self.n_epochs}  |  Loss: {avg_loss:.6f}")

    def predict_proba(self, X):
        _, A = self._forward(X)
        return A

    def predict(self, X, threshold=0.5):
        A = self.predict_proba(X)
        predictions = []
        if self.mode == 'binary':
            for row in A:
                predictions.append(1 if row[0] >= threshold else 0)
        else:
            for row in A:
                max_idx = 0
                for j in range(1, len(row)):
                    if row[j] > row[max_idx]:
                        max_idx = j
                predictions.append(max_idx)
        return predictions

    def accuracy(self, X, Y_true):
        preds = self.predict(X)
        if self.mode == 'binary':
            correct = sum(1 for p, y in zip(preds, Y_true) if p == y[0])
        else:
            correct = sum(1 for p, y in zip(preds, Y_true) if p == y)
        return correct / len(Y_true)
    
def generate_binary_data(n_samples=200, seed=0):
        gen = SimpleLCG(seed)
        X, Y = [], []
        for i in range(n_samples):
            label = i % 2
            cx = 1.0 if label == 1 else -1.0
            cy = 1.0 if label == 1 else -1.0
            x1 = cx + gen.rand_normal() * 0.6
            x2 = cy + gen.rand_normal() * 0.6
            X.append([x1, x2])
            Y.append([float(label)])
        return X, Y

def generate_multiclass_data(n_samples=300, n_classes=3, seed=7):
    gen = SimpleLCG(seed)
    pi  = 3.141592653589793
    centers = [
        [2.0 * (e ** (0)) * 0,   1.5],
        [-1.3, -0.75],
        [1.3,  -0.75],
    ]
    X, Y = [], []
    per_class = n_samples // n_classes
    for c in range(n_classes):
        cx, cy = centers[c]
        for _ in range(per_class):
            x1 = cx + gen.rand_normal() * 0.5
            x2 = cy + gen.rand_normal() * 0.5
            X.append([x1, x2])
            # One-hot encode
            onehot = [0.0] * n_classes
            onehot[c] = 1.0
            Y.append(onehot)
    return X, Y

def train_test_split_pure(X, Y, test_ratio=0.2, seed=1):
    n = len(X)
    indices = list(range(n))
    gen = SimpleLCG(seed)
    for i in range(n-1, 0, -1):
        j = gen.rand_int() % (i+1)
        indices[i], indices[j] = indices[j], indices[i]
    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx  = indices[split:]
    X_train = [X[i] for i in train_idx]
    Y_train = [Y[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    Y_test  = [Y[i] for i in test_idx]
    return X_train, X_test, Y_train, Y_test



print("=" * 50)
print("BINARY LOGISTIC REGRESSION")
print("=" * 50)

X_bin, Y_bin = generate_binary_data(n_samples=400)
X_tr, X_te, Y_tr, Y_te = train_test_split_pure(X_bin, Y_bin)

model_bin = LogisticRegression(
    n_features=2,
    n_classes=2,
    learning_rate=0.3,
    n_epochs=60,
    batch_size=32,
    verbose=True
)
model_bin.fit(X_tr, Y_tr)
acc = model_bin.accuracy(X_te, Y_te)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(f"Sample predictions (first 10): {model_bin.predict(X_te[:10])}")
print(f"Actual labels   (first 10): {[int(y[0]) for y in Y_te[:10]]}")

print("\n" + "=" * 50)
print("MULTICLASS LOGISTIC REGRESSION (SOFTMAX)")
print("=" * 50)

X_mc, Y_mc = generate_multiclass_data(n_samples=600, n_classes=3)
X_tr2, X_te2, Y_tr2, Y_te2 = train_test_split_pure(X_mc, Y_mc)

model_mc = LogisticRegression(
    n_features=2,
    n_classes=3,
    learning_rate=0.5,
    n_epochs=80,
    batch_size=32,
    verbose=True
)
model_mc.fit(X_tr2, Y_tr2)

# For multiclass accuracy, convert one-hot Y_te to class indices
Y_te2_labels = [row.index(max(row)) for row in Y_te2]
acc_mc = model_mc.accuracy(X_te2, Y_te2_labels)
print(f"\nTest Accuracy: {acc_mc*100:.2f}%")
