from utilities import (
    dot, LCG, accuracy, per_class_accuracy,
    train_test_split, gen_blobs, gen_circles, unique,
    mean, absolute, sqrt, log, exp,
    mean_squared_error, mean_absolute_error,
    PI, cos, sin,
)

E = 2.718281828459045


def relu(z):
    return [max(0.0, zi) for zi in z]

def relu_grad(z):
    return [1.0 if zi > 0 else 0.0 for zi in z]

def sigmoid(z):
    return [1.0 / (1.0 + E**(-max(-500.0, min(500.0, zi)))) for zi in z]

def sigmoid_grad(z):
    s = sigmoid(z)
    return [si * (1.0 - si) for si in s]

def tanh_act(z):
    return [(E**zi - E**(-zi)) / (E**zi + E**(-zi))
            if abs(zi) < 300 else (1.0 if zi > 0 else -1.0)
            for zi in z]

def tanh_grad(z):
    t = tanh_act(z)
    return [1.0 - ti**2 for ti in t]

def linear(z):
    return list(z)

def linear_grad(z):
    return [1.0] * len(z)

def softmax(z):
    m = max(z)
    e = [E**(zi - m) for zi in z]
    t = sum(e)
    return [ei / t for ei in e]

ACTIVATIONS = {
    'relu'    : (relu,     relu_grad),
    'sigmoid' : (sigmoid,  sigmoid_grad),
    'tanh'    : (tanh_act, tanh_grad),
    'linear'  : (linear,   linear_grad),
}


class Layer:
    def __init__(self, n_in, n_out, activation='relu', seed=0):
        rng  = LCG(seed)
        # He initialisation for relu, Xavier for others
        scale = sqrt(2.0 / n_in) if activation == 'relu' else sqrt(1.0 / n_in)
        self.W  = [[rng.next_gaussian() * scale for _ in range(n_in)]
                   for _ in range(n_out)]
        self.b  = [0.0] * n_out
        self.act_fn, self.act_grad = ACTIVATIONS[activation]
        self.activation = activation
        # cache for backprop
        self.Z  = None
        self.A  = None
        self.A_prev = None

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = [sum(self.W[j][k] * A_prev[k]
                      for k in range(len(A_prev))) + self.b[j]
                  for j in range(len(self.W))]
        self.A = self.act_fn(self.Z)
        return self.A

    def backward(self, delta):
        '''
        delta : error signal from layer above  shape (n_out,)
        returns dW, db, delta_prev
        '''
        n_in  = len(self.A_prev)
        n_out = len(self.W)
        # apply activation gradient first
        if self.activation != 'softmax':
            g = self.act_grad(self.Z)
            delta_act = [delta[j] * g[j] for j in range(n_out)]
        else:
            delta_act = list(delta)
        dW    = [[delta_act[j] * self.A_prev[k]
                  for k in range(n_in)]
                 for j in range(n_out)]
        db    = list(delta_act)
        # propagate delta back through weights
        delta_prev = [sum(self.W[j][k] * delta_act[j]
                          for j in range(n_out))
                      for k in range(n_in)]
        return dW, db, delta_prev

    def update(self, dW, db, lr):
        for j in range(len(self.W)):
            for k in range(len(self.W[j])):
                self.W[j][k] -= lr * dW[j][k]
            self.b[j] -= lr * db[j]


class MLP:


    def __init__(self, layer_sizes=(64,), activation='relu',
                 task='classification', lr=0.01,
                 epochs=100, batch_size=32, seed=42):
        self.layer_sizes = layer_sizes
        self.activation  = activation
        self.task        = task
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.seed        = seed
        self.layers      = []
        self.loss_history_ = []

    def _build(self, n_in, n_out):
        self.layers = []
        sizes = [n_in] + list(self.layer_sizes) + [n_out]
        for i in range(len(sizes) - 1):
            act = self.activation if i < len(sizes) - 2 else (
                'linear' if self.task == 'regression' else 'sigmoid'
            )
            self.layers.append(Layer(sizes[i], sizes[i+1],
                                     activation=act,
                                     seed=self.seed + i * 100))

    def _forward(self, x):
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def _loss(self, y_pred, y_true):
        if self.task == 'regression':
            return (y_pred[0] - y_true[0])**2
        # binary cross-entropy
        p = max(1e-12, min(1 - 1e-12, y_pred[0]))
        return -(y_true[0] * log(p) + (1 - y_true[0]) * log(1 - p))

    def _output_delta(self, y_pred, y_true):
        if self.task == 'regression':
            return [2.0 * (y_pred[0] - y_true[0])]
        # multiclass or binary: delta = ŷ - y  (works for both)
        return [y_pred[i] - y_true[i] for i in range(len(y_pred))]

    def _backward(self, delta):
        grads = []
        for layer in reversed(self.layers):
            dW, db, delta = layer.backward(delta)
            grads.append((dW, db))
        grads.reverse()
        for layer, (dW, db) in zip(self.layers, grads):
            layer.update(dW, db, self.lr)

    def fit(self, X, Y):
        n   = len(X)
        nf  = len(X[0])
        rng = LCG(self.seed)

        # determine output size and encode Y
        if self.task == 'classification':
            self.classes_   = sorted(unique(Y))
            self.label_map_ = {c: i for i, c in enumerate(self.classes_)}
            n_out = 1 if len(self.classes_) == 2 else len(self.classes_)
            Yn = [[float(self.label_map_[y])] if n_out == 1
                  else [1.0 if self.label_map_[y] == k else 0.0
                        for k in range(n_out)]
                  for y in Y]
        else:
            n_out = 1
            Yn = [[float(y)] for y in Y]

        if self.task == 'classification' and n_out > 2:
            self._build_multiclass(nf, n_out)
        else:
            self._build(nf, n_out)

        self.loss_history_ = []
        bs = self.batch_size or n

        for epoch in range(self.epochs):
            indices = rng.shuffle_indices(n)
            epoch_loss = 0.0
            for start in range(0, n, bs):
                batch = indices[start:start + bs]
                for i in batch:
                    y_pred = self._forward(X[i])
                    epoch_loss += self._loss(y_pred, Yn[i])
                    delta = self._output_delta(y_pred, Yn[i])
                    self._backward(delta)
            self.loss_history_.append(epoch_loss / n)
        return self

    def _build_multiclass(self, n_in, n_out):
        self.layers = []
        sizes = [n_in] + list(self.layer_sizes) + [n_out]
        for i in range(len(sizes) - 1):
            act = self.activation if i < len(sizes) - 2 else 'sigmoid'
            self.layers.append(Layer(sizes[i], sizes[i+1],
                                     activation=act,
                                     seed=self.seed + i * 100))

    def predict_proba(self, X):
        return [self._forward(x) for x in X]

    def predict(self, X):
        probas = self.predict_proba(X)
        if self.task == 'regression':
            return [p[0] for p in probas]
        if len(self.classes_) == 2:
            return [self.classes_[1] if p[0] >= 0.5
                    else self.classes_[0] for p in probas]
        return [self.classes_[p.index(max(p))] for p in probas]

    def accuracy(self, X, Y):
        return accuracy(Y, self.predict(X))

    def score_report(self, X, Y, name=''):
        tag = f' [{name}]' if name else ''
        if self.task == 'classification':
            preds = self.predict(X)
            print(f'  Accuracy{tag}: {accuracy(Y, preds)*100:.2f}%')
            per_class_accuracy(Y, preds)
        else:
            preds = self.predict(X)
            Yf    = [float(y) for y in Y]
            mse   = mean_squared_error(Yf, preds)
            ym    = mean(Yf)
            ss_res = sum((a-b)**2 for a,b in zip(Yf,preds))
            ss_tot = sum((y-ym)**2 for y in Yf)
            r2    = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
            print(f'  MSE{tag}: {mse:.4f}  R2: {r2:.4f}')
        print(f'  Final loss: {self.loss_history_[-1]:.6f}')
        print(f'  Architecture: {[len(X[0])]}'
              f' -> {list(self.layer_sizes)} -> output')

def gen_xor(n=400, seed=2):
    rng = LCG(seed); X, Y = [], []
    for _ in range(n):
        x1 = rng.next_gaussian()
        x2 = rng.next_gaussian()
        Y.append(1 if (x1 > 0) != (x2 > 0) else 0)
        X.append([x1, x2])
    return X, Y

def gen_regression(n=300, seed=5):
    rng = LCG(seed); X, Y = [], []
    def sin_a(x):
        r, t = x, x
        for k in range(1, 15):
            t *= -x*x / ((2*k)*(2*k+1)); r += t
        return r
    for _ in range(n):
        x1 = rng.next_float(-3, 3)
        x2 = rng.next_float(-3, 3)
        X.append([x1, x2])
        Y.append(float(sin_a(x1) + x2**2*0.3 + rng.next_gaussian()*0.2))
    return X, Y


sep = '='*52

print(sep)
print('TEST 1 — MLP on XOR (perceptron failed, MLP solves it)')
print(sep)
Xx, Yx = gen_xor(400)
Xxtr, Xxte, Yxtr, Yxte = train_test_split(Xx, Yx)
mlp_xor = MLP(layer_sizes=(8,), activation='relu',
              lr=0.05, epochs=200, batch_size=32)
mlp_xor.fit(Xxtr, Yxtr)
print()
mlp_xor.score_report(Xxte, Yxte, name='XOR')

print()
print(sep)
print('TEST 2 — MLP on Blobs (3 class)')
print(sep)
X3, Y3 = gen_blobs(450, n_clusters=3, spread=0.7)
Xtr3, Xte3, Ytr3, Yte3 = train_test_split(X3, Y3)
mlp3 = MLP(layer_sizes=(16, 8), activation='relu',
           lr=0.01, epochs=200, batch_size=32)
mlp3.fit(Xtr3, Ytr3)
print()
mlp3.score_report(Xte3, Yte3, name='3-class')

print()
print(sep)
print('TEST 3 — MLP on Concentric Circles')
print(sep)
Xc, Yc = gen_circles(400, n_rings=2)
Xctr, Xcte, Yctr, Ycte = train_test_split(Xc, Yc)
mlp_c = MLP(layer_sizes=(16, 8), activation='tanh',
            lr=0.05, epochs=300, batch_size=32)
mlp_c.fit(Xctr, Yctr)
print()
mlp_c.score_report(Xcte, Ycte, name='circles')

print()
print(sep)
print('TEST 4 — MLP Regression (sine surface)')
print(sep)
Xr, Yr = gen_regression(300)
Xrtr, Xrte, Yrtr, Yrte = train_test_split(Xr, Yr)
mlp_r = MLP(layer_sizes=(32, 16), activation='relu',
            task='regression', lr=0.01, epochs=300, batch_size=32)
mlp_r.fit(Xrtr, Yrtr)
print()
mlp_r.score_report(Xrte, Yrte, name='regression')

print()
print(sep)
print('TEST 5 — Depth comparison: shallow vs deep')
print(sep)
X, Y = gen_xor(400)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y)
print()
for sizes, name in [
    ((4,),       'shallow  [4]'),
    ((8,),       'shallow  [8]'),
    ((8, 4),     'deep     [8,4]'),
    ((16, 8, 4), 'deep     [16,8,4]'),
]:
    m = MLP(layer_sizes=sizes, activation='relu',
            lr=0.05, epochs=200, batch_size=32)
    m.fit(Xtr, Ytr)
    acc = m.accuracy(Xte, Yte)
    print(f'  {name:<22} acc={acc*100:.2f}%  '
          f'final_loss={m.loss_history_[-1]:.4f}')

print()
print(sep)
print('TEST 6 — Activation comparison on XOR')
print(sep)
print()
for act in ['relu', 'tanh', 'sigmoid']:
    m = MLP(layer_sizes=(16,), activation=act,
            lr=0.05, epochs=200, batch_size=32)
    m.fit(Xtr, Ytr)
    acc = m.accuracy(Xte, Yte)
    print(f'  {act:<10}  acc={acc*100:.2f}%  '
          f'final_loss={m.loss_history_[-1]:.4f}')

print()
print(sep)
print('TEST 7 — Loss curve (first 10 and last 5 epochs)')
print(sep)
mlp_lc = MLP(layer_sizes=(16, 8), activation='relu',
             lr=0.05, epochs=50, batch_size=32)
mlp_lc.fit(Xtr, Ytr)
print()
print('  First 10 epochs:')
for i, l in enumerate(mlp_lc.loss_history_[:10]):
    bar = '&' * int(l * 40)
    print(f'  epoch {i+1:>3}: {l:.4f}  {bar}')
print('  ...')
print('  Last 5 epochs:')
for i, l in enumerate(mlp_lc.loss_history_[-5:]):
    bar = '&' * int(l * 40)
    print(f'  epoch {46+i:>3}: {l:.4f}  {bar}')
