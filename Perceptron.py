from utilities import (
    dot, LCG, accuracy, per_class_accuracy,
    train_test_split, gen_blobs, unique,
    mean, absolute, PI, cos, sin,
)

def step(z):
    return 1 if z >= 0 else 0

def sign(z):
    return 1 if z >= 0 else -1

class Perceptron:
    '''
    Binary classifier. Labels must be 0 and 1.

    Parameters:
    lr       : learning rate
    epochs   : max passes over training data
    seed     : for weight initialisation
    '''

    def __init__(self, lr=0.01, epochs=100, seed=42):
        self.lr     = lr
        self.epochs = epochs
        self.seed   = seed
        self.w      = None
        self.b      = 0.0
        self.errors_per_epoch_ = []

    def fit(self, X, Y):
        rng = LCG(self.seed)
        nf  = len(X[0])
        self.w = [rng.next_gaussian() * 0.01 for _ in range(nf)]
        self.b = 0.0
        self.errors_per_epoch_ = []
        Yn = [int(y) for y in Y]

        for epoch in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, Yn):
                z      = dot(self.w, xi) + self.b
                y_hat  = step(z)
                error  = yi - y_hat
                if error != 0:
                    errors += 1
                    self.w = [self.w[j] + self.lr * error * xi[j]
                              for j in range(nf)]
                    self.b += self.lr * error
            self.errors_per_epoch_.append(errors)
            if errors == 0:
                break
        return self

    def predict(self, X):
        return [step(dot(self.w, x) + self.b) for x in X]

    def accuracy(self, X, Y):
        return accuracy([int(y) for y in Y], self.predict(X))

    def score_report(self, X, Y, name=''):
        preds = self.predict(X)
        tag   = f' [{name}]' if name else ''
        print(f'  Accuracy{tag}: {accuracy([int(y) for y in Y], preds)*100:.2f}%')
        print(f'  Weights : {[round(w, 4) for w in self.w]}')
        print(f'  Bias    : {round(self.b, 4)}')
        print(f'  Epochs to converge: {len(self.errors_per_epoch_)}')


class VotedPerceptron:
    def __init__(self, lr=0.1, epochs=100, seed=42):
        self.lr     = lr
        self.epochs = epochs
        self.seed   = seed
        self.voted_weights_ = []   # list of (w, b, vote_count)

    def fit(self, X, Y):
        rng = LCG(self.seed)
        nf  = len(X[0])
        w   = [rng.next_gaussian() * 0.01 for _ in range(nf)]
        b   = 0.0
        c   = 0   # survival count of current weight vector
        Yn  = [int(y) for y in Y]
        self.voted_weights_ = []

        for epoch in range(self.epochs):
            for xi, yi in zip(X, Yn):
                y_hat = step(dot(w, xi) + b)
                if y_hat == yi:
                    c += 1
                else:
                    self.voted_weights_.append((list(w), b, c))
                    w = [w[j] + self.lr * (yi - y_hat) * xi[j]
                         for j in range(nf)]
                    b += self.lr * (yi - y_hat)
                    c  = 0

        self.voted_weights_.append((list(w), b, c))
        return self

    def predict(self, X):
        preds = []
        for x in X:
            score = sum(c * sign(dot(w, x) + b)
                        for w, b, c in self.voted_weights_)
            preds.append(1 if score >= 0 else 0)
        return preds

    def accuracy(self, X, Y):
        return accuracy([int(y) for y in Y], self.predict(X))


class MulticlassPerceptron:
    '''
    Trains one binary Perceptron per class (one-vs-rest).
    Predicts the class whose perceptron fires most confidently.
    '''

    def __init__(self, lr=0.1, epochs=100, seed=42):
        self.lr      = lr
        self.epochs  = epochs
        self.seed    = seed
        self.models  = {}
        self.classes = []

    def fit(self, X, Y):
        self.classes = sorted(unique(Y))
        for c in self.classes:
            Yb = [1 if y == c else 0 for y in Y]
            m  = Perceptron(lr=self.lr, epochs=self.epochs,
                            seed=self.seed + hash(str(c)) % 1000)
            m.fit(X, Yb)
            self.models[c] = m
        return self

    def predict(self, X):
        scores = {c: [dot(self.models[c].w, x) + self.models[c].b
                      for x in X]
                  for c in self.classes}
        return [max(self.classes,
                    key=lambda c: scores[c][i])
                for i in range(len(X))]

    def accuracy(self, X, Y):
        return accuracy(Y, self.predict(X))

    def score_report(self, X, Y, name=''):
        preds = self.predict(X)
        tag   = f' [{name}]' if name else ''
        print(f'  Accuracy{tag}: {accuracy(Y, preds)*100:.2f}%')
        per_class_accuracy(Y, preds)


def gen_linearly_separable(n=200, seed=1):
    rng = LCG(seed); X, Y = [], []
    for _ in range(n):
        x1 = rng.next_gaussian()
        x2 = rng.next_gaussian()
        y  = 1 if x1 + x2 > 0 else 0
        X.append([x1, x2]); Y.append(y)
    return X, Y

def gen_xor(n=200, seed=2):
    rng = LCG(seed); X, Y = [], []
    for _ in range(n):
        x1 = rng.next_gaussian()
        x2 = rng.next_gaussian()
        y  = 1 if (x1 > 0) != (x2 > 0) else 0
        X.append([x1, x2]); Y.append(y)
    return X, Y

sep = '='*52

print(sep)
print('TEST 1 — Perceptron on linearly separable data')
print(sep)
X, Y = gen_linearly_separable(300)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y)
p = Perceptron(lr=0.1, epochs=100)
p.fit(Xtr, Ytr)
print()
p.score_report(Xte, Yte, name='perceptron')
print(f'  Errors per epoch: {p.errors_per_epoch_[:10]}...')

print()
print(sep)
print('TEST 2 — Perceptron on XOR (not linearly separable)')
print('  Perceptron theorem: will never converge on XOR')
print(sep)
Xx, Yx = gen_xor(300)
Xxtr, Xxte, Yxtr, Yxte = train_test_split(Xx, Yx)
px = Perceptron(lr=0.1, epochs=50)
px.fit(Xxtr, Yxtr)
print()
px.score_report(Xxte, Yxte, name='XOR')
print(f'  Errors per epoch (never reaches 0): {px.errors_per_epoch_}')

print()
print(sep)
print('TEST 3 — Learning rate comparison')
print(sep)
X, Y = gen_linearly_separable(300)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y)
print()
for lr in [0.001, 0.01, 0.1, 0.5, 1.0]:
    p_lr = Perceptron(lr=lr, epochs=100)
    p_lr.fit(Xtr, Ytr)
    acc  = p_lr.accuracy(Xte, Yte)
    conv = len(p_lr.errors_per_epoch_)
    print(f'  lr={lr:<6}  acc={acc*100:.2f}%  epochs_to_converge={conv}')

print()
print(sep)
print('TEST 4 — Voted Perceptron vs Vanilla Perceptron')
print(sep)
X, Y = gen_linearly_separable(400)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y)
print()
p_v  = Perceptron(lr=0.1, epochs=50)
p_vp = VotedPerceptron(lr=0.1, epochs=50)
p_v.fit(Xtr, Ytr);  p_vp.fit(Xtr, Ytr)
print(f'  Vanilla Perceptron : {p_v.accuracy(Xte, Yte)*100:.2f}%')
print(f'  Voted Perceptron   : {p_vp.accuracy(Xte, Yte)*100:.2f}%')
print(f'  Vote vectors kept  : {len(p_vp.voted_weights_)}')

print()
print(sep)
print('TEST 5 — Multiclass Perceptron (one-vs-rest)')
print(sep)
X3, Y3 = gen_blobs(450, n_clusters=3, spread=0.6)
Xtr3, Xte3, Ytr3, Yte3 = train_test_split(X3, Y3)
mp = MulticlassPerceptron(lr=0.1, epochs=100)
mp.fit(Xtr3, Ytr3)
print()
mp.score_report(Xte3, Yte3, name='3-class OvR')

print()
print(sep)
print('TEST 6 — Decision boundary visualization (ASCII)')
print('  + = class 1,  . = class 0,  boundary where w·x+b=0')
print(sep)
X, Y = gen_linearly_separable(300)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y)
p_vis = Perceptron(lr=0.1, epochs=100)
p_vis.fit(Xtr, Ytr)
print()
w1, w2, b = p_vis.w[0], p_vis.w[1], p_vis.b
rows, cols = 15, 40
for r in range(rows):
    line = ''
    x2 = 2.5 - r * (5.0 / rows)
    for c in range(cols):
        x1  = -2.5 + c * (5.0 / cols)
        z   = w1*x1 + w2*x2 + b
        line += '+' if z >= 0 else '.'
    print('  ' + line)
print(f'  Boundary: {w1:.3f}*x1 + {w2:.3f}*x2 + {b:.3f} = 0')
