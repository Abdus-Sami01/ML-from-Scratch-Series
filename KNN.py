from utilities import (
    norm, mean, std, sqrt, log, exp, absolute, dot, vec_sub,
    LCG, train_test_split, unique, accuracy, per_class_accuracy,
    confusion_matrix, mean_squared_error, mean_absolute_error, PI, cos
)

rng = LCG(42)

def euclidean(a, b): 
    return norm(vec_sub(a, b), p=2)
def manhattan(a, b): 
    return norm(vec_sub(a, b), p=1)
def cosine_dist(a, b):
    d = (norm(a,2) * norm(b,2))
    return 1.0 - max(-1.0, min(1.0, dot(a,b)/d)) if d else 1.0

METRICS = {'euclidean': euclidean, 'manhattan': manhattan, 'cosine': cosine_dist}

def get_neighbours(X_train, x, k, dist_fn):
    dists = []
    for i, xt in enumerate(X_train):
        d = dist_fn(x, xt)
        dists.append((d , i))
    dists.sort()
    return dists[:k]

def idw(neighbours):
    for d, _ in neighbours:
        if d == 0.0:
            return [1.0 if di == 0.0 else 0.0 for di, _ in neighbours]
    w = [1.0 / d**2 for d, _ in neighbours]
    t = sum(w)
    return [wi/t for wi in w]

def z_normalise(Xtr, Xte=None):
    nf = len(Xtr[0])
    mu = [mean([r[j] for r in Xtr]) for j in range(nf)]
    sd = [std([r[j]  for r in Xtr]) or 1.0 for j in range(nf)]
    scale = lambda X: [[(X[i][j]-mu[j])/sd[j] for j in range(nf)] for i in range(len(X))]
    return (scale(Xtr), scale(Xte)) if Xte else scale(Xtr)


class KNNClassifier:
    def __init__(self, k=5, metric='euclidean', weighted=False):
        self.k, self.weighted = k, weighted
        self.dist_fn = METRICS[metric]

    def fit(self, X, Y):
        self.X, self.Y = X, Y
        self.classes = sorted(unique(Y))
        return self

    def _pred_one(self, x):
        nb = get_neighbours(self.X, x, self.k, self.dist_fn)
        if not self.weighted:
            counts = {}
            for d, i in nb:
                counts[self.Y[i]] = counts.get(self.Y[i], 0) + 1
            return max(counts, key=lambda l: (counts[l], l))
        else:
            w = idw(nb)
            scores = {}
            for wi, (d, i) in zip(w, nb):
                scores[self.Y[i]] = scores.get(self.Y[i], 0.0) + wi
            return max(scores, key=lambda l: scores[l])

    def predict(self, X):      return [self._pred_one(x) for x in X]
    def accuracy(self, X, Y):  return accuracy(Y, self.predict(X))


class KNNRegressor:
    def __init__(self, k=5, metric='euclidean', weighted=False):
        self.k, self.weighted = k, weighted
        self.dist_fn = METRICS[metric]

    def fit(self, X, Y):
        self.X, self.Y = X, Y
        return self

    def _pred_one(self, x):
        nb = get_neighbours(self.X, x, self.k, self.dist_fn)
        if not self.weighted:
            return mean([self.Y[i] for _, i in nb])
        w = idw(nb)
        return sum(wi * self.Y[i] for wi, (_, i) in zip(w, nb))

    def predict(self, X): return [self._pred_one(x) for x in X]

    def r2(self, X, Y):
        p = self.predict(X); ym = mean(Y)
        return 1 - sum((a-b)**2 for a,b in zip(Y,p)) / sum((y-ym)**2 for y in Y)

X_train = [
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 1.0],
    [6.0, 5.0],
    [7.0, 4.0],
    [8.0, 6.0],
]

Y_train = [0, 0, 0, 1, 1, 1]

X_test = [
    [2.5, 2.5],   
    [7.0, 5.0],   
]

X_train_n, X_test_n = z_normalise(X_train, X_test)

clf = KNNClassifier(k=3, metric='euclidean', weighted=False)
clf.fit(X_train_n, Y_train)
print(clf.predict(X_test_n))   


X_reg_train = [[1.0], [2.0], [3.0], [4.0], [5.0]]
Y_reg_train  = [2.0,   4.0,   6.0,   8.0,  10.0]  

X_reg_test   = [[2.5], [3.5]]

X_reg_n, X_reg_te_n = z_normalise(X_reg_train, X_reg_test)

reg = KNNRegressor(k=2, weighted=False)
reg.fit(X_reg_n, Y_reg_train)
print(reg.predict(X_reg_te_n))   
